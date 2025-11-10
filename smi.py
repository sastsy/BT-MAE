import torch
from tqdm import tqdm, trange
import csv
import time
# import matplotlib.pyplot as plt
import os
import logging
import math

logger2 = logging.getLogger(__name__)
logger2.setLevel(logging.INFO)

import numpy as np


def chebyshev_dist(a, b, compute_dtype=torch.float16):
    aa = a.to(compute_dtype)
    bb = b.to(compute_dtype)
    out = (aa.unsqueeze(1) - bb.unsqueeze(0)).abs().amax(dim=-1)
    return out.to(torch.float32)


class KSG:
    def __init__(self, k_neighbors: int = 3, device="cuda", topk=False):
        """
        KSG estimator for mutual information using Chebyshev distance.
        
        Args:
            k_neighbors (int): number of neighbors for estimation
            device (str): 'cuda' or 'cpu'
        """
        self.k = k_neighbors
        self.device = device
        self.topk = topk
        logger2.info(f"Using topk {topk}")

    @torch.inference_mode()
    def __call1__(self, x, y, std=False):
        """
        Estimate mutual information between x and y.
        
        Args:
            x, y: arrays (torch.Tensor or np.ndarray) of shape (n, d)
            std (bool): to return standard error
            
        Returns:
            Mutual information estimate (and standard error if std=True)
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        n = x.shape[0]
        k = min(self.k, n - 1)

        xy = torch.cat([x, y], dim=1)  # (n, dx+dy)

        d_xy = chebyshev_dist(xy, xy)  # (n, n)
        d_x  = chebyshev_dist(x, x)
        d_y  = chebyshev_dist(y, y)

        eps = 1e-12
        if self.topk:
            values, _ = torch.topk(d_xy, k+1, largest=False, dim=1)
            r = values[:, -1] - eps
        else:
            d_xy_sorted, _ = torch.sort(d_xy, dim=1)
            r = d_xy_sorted[:, k] - eps

        nx = (d_x <= r.unsqueeze(1)).sum(dim=1) - 1  
        ny = (d_y <= r.unsqueeze(1)).sum(dim=1) - 1

        psi = torch.special.digamma
        base = psi(torch.tensor(k, device=self.device, dtype=torch.float32)) \
            + psi(torch.tensor(n, device=self.device, dtype=torch.float32))
        vals = psi(nx.to(torch.float32) + 1) + psi(ny.to(torch.float32) + 1)

        mi = (base - vals.mean()).clamp_min(0.0).item() 
        se = (vals.std(unbiased=True) / math.sqrt(n)).item()
        if std:
            return mi, se
        else:
            return mi
        
    # streaming
    @torch.inference_mode()
    def __call__(self, x, y, device="cuda", row_chunk=4096, dist_dtype=torch.float16, std=False,  debug=False):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device)
        n = x.shape[0]
        k = min(self.k, n-1)

        xy = torch.cat([x, y], dim=1)
        eps = torch.finfo(torch.float32).eps

        r = torch.empty(n, device=device, dtype=torch.float32)
        for i in range(0, n, row_chunk):
            j = min(i + row_chunk, n)
            d = chebyshev_dist(xy[i:j], xy)
            r[i:j] = torch.topk(d, k + 1, largest=False, dim=1).values[:, -1] - eps

        nx = torch.empty(n, device=device, dtype=torch.int32)
        ny = torch.empty(n, device=device, dtype=torch.int32)
        for i in range(0, n, row_chunk):
            j = min(i + row_chunk, n)
            dxi = chebyshev_dist(x[i:j], x)
            dyi = chebyshev_dist(y[i:j], y)
            ri  = r[i:j].unsqueeze(1)
            nx[i:j] = (dxi <= ri).sum(1) - 1
            ny[i:j] = (dyi <= ri).sum(1) - 1
        
        # print(f"nx min={nx.min().item()}, max={nx.max().item()}, ny min={ny.min().item()}, max={ny.max().item()}")

        psi = torch.special.digamma
        base = psi(torch.tensor(k, device=device, dtype=torch.float32)) \
            + psi(torch.tensor(n, device=device, dtype=torch.float32))
        vals = psi(nx.to(torch.float32) + 1) + psi(ny.to(torch.float32) + 1)

        mi = (base - vals.mean()).item() 

        se = (vals.std(unbiased=True) / math.sqrt(n)).item()

        if std:
            return mi, se 
        return mi 


@torch.inference_mode()
def whiten_torch(X, eps=torch.finfo(torch.float32).eps, device="cuda"):
    """
    PCA whitening on input data.
    
    Args:
        X: input np.ndarray or torch.Tensor
        eps: constant
        device:
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    else:
        X = X.to(dtype=torch.float32, device=device)
    
    print(f"[DEBUG] Whitening input: mean={X.mean().item():.4f}, std={X.std().item():.4f}")

    Xc = X - X.mean(dim=0, keepdim=True)
    n = Xc.shape[0]
    cov = (Xc.t() @ Xc) / (n - 1)  # (d, d)

    U, S, Vh = torch.linalg.svd(cov) 
    print(f"[DEBUG] Whitening eigenvalues min={S.min().item():.6f}, max={S.max().item():.6f}")
    inv_sqrt = torch.diag(1.0 / torch.sqrt(S + eps))
    W = U @ inv_sqrt @ U.t()
    Xw = Xc @ W
    return Xw


@torch.inference_mode()
def random_orthonormal_matrices_torch(d, k, batch, device="cuda"):
    """
    Generate random orthonormal matrices using Haar measure.
    
    Args:
        d: original dimension
        k: projection dimension
        batch: number of matrices to generate
        device:
        
    Returns:
        Tensor of shape (batch, d, k)
    """
    A = torch.randn((batch, d, k), device=device, dtype=torch.float32)
    Q, R = torch.linalg.qr(A, mode='reduced')  # Q: (batch,d,k), R: (batch,k,k)
    diag = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))  # (batch, k)
    diag[diag == 0] = 1.0
    Q = Q * diag.unsqueeze(1)  # broadcast -> (batch, d, k)
    return Q


def make_projection_matrices(dX, dY, proj_k, n_proj, device, seed=0, aligned=False):
    """
    Generate pairs of random projection matrices.
    
    Args:
        dX, dY: input dimensions
        proj_k: projection dimension
        n_proj: number of projection pairs
        device: 
        seed: 
        
    Returns:
        projection matrices (U_list, V_list)
    """
    torch.manual_seed(seed)
    U_list = random_orthonormal_matrices_torch(dX, proj_k, n_proj, device=device)  # (n_proj,dX,k)
    if aligned:
        V_list = U_list
    else:
        V_list = random_orthonormal_matrices_torch(dY, proj_k, n_proj, device=device)  # (n_proj,dY,k)
    return U_list, V_list


@torch.inference_mode()
def sliced_mi_with_fixed_proj_batched(X_np, Y_np, U_list, V_list, knn=5, device="cuda", batch_size=10000, topk=False, collect_raw=True, debug=False):
    """
    Compute sliced mutual information using pre-computed projection matrices.
    
    Args:
        X_np, Y_np: input arrays
        U_list, V_list: projection matrices
        knn: number of neighbors for KSG
        device: 
        batch_size: size for processing
        
    Returns:
        Tuple of (mean, standard error, all MI values between projections)
    """
    Xw = whiten_torch(X_np, device=device)  # normalize_torch
    Yw = whiten_torch(Y_np, device=device) # normalize_torch
    n_proj = U_list.shape[0]

    ksg = KSG(k_neighbors=knn, topk=topk, device=device)

    mis = []

    for start in trange(0, n_proj, batch_size, desc='Number of projections'):
        end = min(start + batch_size, n_proj)

        U_batch = U_list[start:end]
        V_batch = V_list[start:end]

        X_proj_batch = torch.einsum('nd,bdk->bnk', Xw, U_batch) 
        Y_proj_batch = torch.einsum('nd,bdk->bnk', Yw, V_batch)
        if torch.isnan(X_proj_batch).any() or torch.isnan(Y_proj_batch).any():
            print("NaNs detected in projections!")
            return 0.0, 0.0, np.zeros(1)

        for i in trange(X_proj_batch.shape[0], leave=False):
            mi = ksg(X_proj_batch[i], Y_proj_batch[i])  # X_proj_batch[i] size (n, k)
            mis.append(float(mi))

        torch.cuda.empty_cache()

    mis = np.array(mis, dtype=float)

    return mis.mean(), mis.std() / np.sqrt(len(mis)), mis


def compute_smi(X1, Y1, proj_k=5, n_proj=5000, knn=3, seed=0, batch_size=512, device="cuda", topk=False, aligned=False, Us=None, run_name='smi', epoch_name ='0'):
    """
    Compare mutual information between two dataset pairs using sliced MI.
    
    Args:
        X1, Y1: first dataset pair
        proj_k: projection dimension
        n_proj: number of projections
        knn: number of neighbors for KSG
        seed: 
        batch_size: size for processing
        device: 
        topk: using torch.topk or torch.sort
        aligned: U_list = V_list or not
        Us: if not None U_list = V_list = Us
        
    Returns:
        tuple of results
    """
    dX, dY = X1.shape[1], Y1.shape[1]
    n_samples = X1.shape[0]
    if Us is None:
        U_list, V_list = make_projection_matrices(dX, dY, proj_k, n_proj, seed=seed, device=device, aligned=aligned)
    else:
        logger2.info("Orthogonal matrices are given")
        U_list, V_list = Us

    logger2.info(f"[INFO] Computing smi ({run_name}) for {n_proj} projections in batches of {batch_size} with proj_k {proj_k}, knn {knn}, seed {seed}...")

    t0 = time.perf_counter()

    mean1, se1, mis1 = sliced_mi_with_fixed_proj_batched(
        X1, Y1, U_list, V_list, knn=knn, device=device, batch_size=batch_size, topk=topk,
    )

    # if device.startswith("cuda"):
    #     torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0

    logger2.info(f"[INFO] Computation done: mean={mean1:.4f}, stderr={se1:.4f} during {elapsed:.2f} sec")

    os.makedirs(run_name, exist_ok=True)
    out_dir = run_name

    # fname = os.path.join(out_dir, f"smi_hist_{epoch_name}.png")

    # plt.figure(figsize=(8, 4))
    # plt.hist(mis1, bins=30, alpha=0.5, label=f'{epoch_name}')
    # plt.axvline(mean1, color='blue', linestyle='--')
    
    # plt.axvline(mean1, color='blue', linestyle='--', label=f'mean={mean1:.4f}')
    # plt.axvline(mean1 - se1, color='blue', linestyle=':', label=f'std={se1:.4f}')
    # plt.axvline(mean1 + se1, color='blue', linestyle=':')
    
    # plt.legend()
    # plt.xlabel("Sliced MI")
    # plt.ylabel("Count")
    # plt.title(f"Distribution of SMI on {epoch_name} epoch")
    # plt.tight_layout()
    # plt.savefig(fname, dpi=150)
    # plt.close()

    out_csv = os.path.join(out_dir, "results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch","n_samples", "n_proj", "proj_k", "knn_k","batch_size", "time_seconds",
            "mean_smi","std"
        ])
    
    row = (
                epoch_name, n_samples,n_proj, proj_k, knn, batch_size, elapsed,
                mean1, se1
            )

    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return (mean1, se1, mis1)


@torch.inference_mode()
def compute_and_log_mi(
    model,
    data_loader,
    U_list,
    V_list,
    mask1_fixed,
    mask2_fixed,
    args,
    epoch,
    log_writer=None,
):
    model.eval()

    device = next(model.parameters()).device

    X_embs = []
    Y_embs = []

    for samples, _ in tqdm(data_loader, desc=f"Collecting MI embeddings (epoch {epoch})"):
        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            # mi_view arg is for fixing the mask
            emb1 = model(samples, mask_ratio=args.mask_ratio, mi_view=1)["cls_feats"]
            emb2 = model(samples, mask_ratio=args.mask_ratio, mi_view=2)["cls_feats"]

        emb1_pool = emb1.detach().cpu()
        emb2_pool = emb2.detach().cpu()
        
        X_embs.append(emb1_pool)
        Y_embs.append(emb2_pool)

    X_embs = torch.cat(X_embs, dim=0).numpy()
    Y_embs = torch.cat(Y_embs, dim=0).numpy()
    
    X_var = X_embs.var(axis=0)
    Y_var = Y_embs.var(axis=0)

    print("X_embs variance: min =", X_var.min(), ", max =", X_var.max())
    print("Y_embs variance: min =", Y_var.min(), ", max =", Y_var.max())

    
    print(f"X_embs shape: {X_embs.shape}, Y_embs shape: {Y_embs.shape}")
    print(f"X_embs mean={X_embs.mean():.4f}, std={X_embs.std():.4f}")
    print(f"Y_embs mean={Y_embs.mean():.4f}, std={Y_embs.std():.4f}")
    print(f"Any NaNs? X={np.isnan(X_embs).any()}, Y={np.isnan(Y_embs).any()}")
    print(f"Any infs? X={np.isinf(X_embs).any()}, Y={np.isinf(Y_embs).any()}")

    mean_mi, se_mi, mis = compute_smi(
        X_embs, Y_embs,
        proj_k=5,
        n_proj=1000,
        knn=3,
        seed=0,
        batch_size=512,
        device=device,
        topk=False,
        aligned=False,
        Us=(U_list, V_list),
        run_name='./smi_logs',
        epoch_name=str(epoch)
    )

    if log_writer is not None:
        log_writer.add_scalar('mutual_info/mean', mean_mi, epoch)
        log_writer.add_scalar('mutual_info/std_error', se_mi, epoch)

    print(f"[Epoch {epoch}] Mean MI = {mean_mi:.4f} ± {se_mi:.4f}")