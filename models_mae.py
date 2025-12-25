# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from loss_func import GatherLayer


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 global_pool=False, num_classes=100,
                 bt_variant=None, bt_lambda=0.005, bt_weight=0.05, gather_layer=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        # new 
        self.global_pool = global_pool
        self.fc_norm = norm_layer(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes, bias=True)
        
        self.bt_variant = bt_variant
        self.bt_lambda = bt_lambda
        self.bt_weight = bt_weight
        self.gather_layer = gather_layer
        
        self.fixed_mask1, self.fixed_ids_restore1, \
        self.fixed_mask2, self.fixed_ids_restore2 = self._make_two_orthogonal_masks()
        
    def _make_two_orthogonal_masks(self):
        L = self.patch_embed.num_patches
        len_keep = int(L * (1 - 0.75))  # ratio = 0.25 keep

        # One random shuffle for both masks
        noise = torch.rand(L)
        ids_sorted = torch.argsort(noise)

        # Split into two disjoint sets
        ids_keep_1 = ids_sorted[:len_keep]
        ids_keep_2 = ids_sorted[len_keep:2*len_keep]  # next chunk, guaranteed disjoint

        # --- Build mask1 ---
        mask1 = torch.ones(L)
        mask1[ids_keep_1] = 0

        ids_restore1 = torch.argsort(torch.argsort(noise))  # same restore order

        mask1 = mask1[ids_restore1]

        # --- Build mask2 ---
        mask2 = torch.ones(L)
        mask2[ids_keep_2] = 0

        ids_restore2 = ids_restore1.clone()  # same reorder

        mask2 = mask2[ids_restore2]

        return mask1, ids_restore1, mask2, ids_restore2
    
    def make_orthogonal_masks(self, N, L, len_keep, device):
        """
        Returns:
            mask1, ids_restore1
            mask2, ids_restore2
        """

        noise = torch.rand(N, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep1 = ids_shuffle[:, :len_keep]
        ids_keep2 = ids_shuffle[:, len_keep:2 * len_keep]

        mask1 = torch.ones(N, L, device=device)
        mask2 = torch.ones(N, L, device=device)

        mask1.scatter_(1, ids_keep1, 0)
        mask2.scatter_(1, ids_keep2, 0)

        return mask1, ids_restore, mask2, ids_restore
    
    def compute_bt_loss_per_image(self, latent):
        B, N, d = latent.shape

        bt_losses = []
        on_diags = []
        off_diags = []
        for z_img in latent:
            z_tokens = z_img[1:]
            z_img = (z_img - z_img.mean(0)) / (z_img.std(0) + 1e-6)
            
            num_samples = z_tokens.shape[0]

            c = (z_img.T @ z_img) / num_samples
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()

            on_diags.append(on_diag)
            off_diags.append(off_diag)

            bt_losses.append(on_diag + self.bt_lambda * off_diag)
        
        bt_loss = torch.stack(bt_losses).mean()
        on_diag_mean = torch.stack(on_diags).mean()
        off_diag_mean = torch.stack(off_diags).mean()

        return bt_loss, on_diag_mean, off_diag_mean

    def compute_bt_loss_per_batch(self, latent):
        B, N, d = latent.shape

        z = latent[:, 1:, :].reshape(B * (N - 1), d)
        if self.gather_layer:
            if dist.is_initialized() and dist.get_world_size() > 1:
                z_global = torch.cat(GatherLayer.apply(z), dim=0)
        else:
            z_global = z

        # z_global = F.normalize(z_global, dim=-1)
        z_global = (z_global - z_global.mean(0)) / (z_global.std(0) + 1e-6)

        c = (z_global.T @ z_global) / z_global.shape[0]
        on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
        off_diag = off_diagonal(c).pow_(2).mean()
        
        bt_loss = on_diag + self.bt_lambda * off_diag

        return bt_loss, on_diag, off_diag
    
    def compute_bt_loss_cls(self, cls_feats):
        if self.gather_layer:
            if dist.is_initialized() and dist.get_world_size() > 1:
                cls_feats = torch.cat(GatherLayer.apply(cls_feats), dim=0)
    
        # cls_feats = F.normalize(cls_feats, dim=-1)
        cls_feats = (cls_feats - cls_feats.mean(0)) / (cls_feats.std(0) + 1e-5)

        B, D = cls_feats.shape
        c = (cls_feats.T @ cls_feats) / B

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        bt_loss = on_diag + self.bt_lambda * off_diag
        return bt_loss, on_diag, off_diag
    
    def compute_bt_loss_cross(self, z1, z2):
        z1_flat = z1[:, 1:, :].reshape(-1, z1.shape[-1])
        z2_flat = z2[:, 1:, :].reshape(-1, z2.shape[-1])
        
        if self.gather_layer:
            if dist.is_initialized() and dist.get_world_size() > 1:
                z1_flat = torch.cat(GatherLayer.apply(z1_flat), dim=0)
                z2_flat = torch.cat(GatherLayer.apply(z2_flat), dim=0)

        z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
        z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)

        num_samples = z1_norm.shape[0]
        c = (z1_norm.T @ z2_norm) / num_samples

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        
        off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()

        bt_loss = on_diag + self.bt_lambda * off_diag
        
        return bt_loss, on_diag, off_diag
    
    def compute_bt_loss_cross_cls(self, cls1, cls2):
        """
        cls1, cls2: [B, D] CLS features from two views
        """
        if self.gather_layer:
            if dist.is_initialized() and dist.get_world_size() > 1:
                cls1 = torch.cat(GatherLayer.apply(cls1), dim=0)
                cls2 = torch.cat(GatherLayer.apply(cls2), dim=0)

        cls1 = (cls1 - cls1.mean(0)) / (cls1.std(0) + 1e-5)
        cls2 = (cls2 - cls2.mean(0)) / (cls2.std(0) + 1e-5)

        B, D = cls1.shape
        c = (cls1.T @ cls2) / B

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        bt_loss = on_diag + self.bt_lambda * off_diag
        return bt_loss, on_diag, off_diag
    
    def _get_masks(self, imgs, mask_ratio, mi_view=None, two_views=False):
        device = imgs.device
        N = imgs.size(0)

        if mi_view is not None:
            if mi_view == 1:
                mask = self.fixed_mask1.to(device).expand(N, -1)
                ids_restore = self.fixed_ids_restore1.to(device).expand(N, -1)
            elif mi_view == 2:
                mask = self.fixed_mask2.to(device).expand(N, -1)
                ids_restore = self.fixed_ids_restore2.to(device).expand(N, -1)
            else:
                raise ValueError("mi_view must be 1, 2 or None")
            return [(mask, ids_restore)]

        if two_views:
            # build orthogonal masks
            with torch.no_grad():
                _, mask1, _ = self.forward_encoder(imgs, mask_ratio)
                N, L = mask1.shape
                len_keep = int((mask1 == 0).sum(dim=1).min().item())

            return self.make_orthogonal_masks(N, L, len_keep, device)

        return [(None, None)]
    
    def _encode_view(self, imgs, mask_ratio, mask, ids_restore):
        latent, used_mask, used_ids_restore = self.forward_encoder(
            imgs, mask_ratio, mask=mask, ids_restore=ids_restore
        )
        return latent, used_mask, used_ids_restore
    
    def _compute_bt(self, latent1, latent2=None, cls1=None, cls2=None):
        if self.bt_variant == "per_image":
            return self.compute_bt_loss_per_image(latent1)

        if self.bt_variant == "per_batch":
            return self.compute_bt_loss_per_batch(latent1)

        if self.bt_variant == "cls":
            return self.compute_bt_loss_cls(cls1)

        if self.bt_variant == "cls_cross":
            return self.compute_bt_loss_cross_cls(cls1, cls2)

        if self.bt_variant == "per_image_cross":
            return self.compute_bt_loss_cross(latent1, latent2)

        return None, None, None

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, mask=None, ids_restore=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        if mask is None:
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            N, L, D = x.shape
            len_keep = (mask == 0).sum(dim=1).min().item()
            ids_keep = torch.argsort(mask, dim=1)[:, :len_keep]
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, mask=None, mi_view=None):
        two_views = self.bt_variant in {"cls_cross", "per_image_cross"} and mi_view is None

        masks = self._get_masks(imgs, mask_ratio, mi_view=mi_view, two_views=two_views)

        if two_views:
            mask1, ids_restore1, mask2, ids_restore2 = masks
            latent1, _, _ = self._encode_view(imgs, mask_ratio, mask1, ids_restore1)
            latent2, _, _ = self._encode_view(imgs, mask_ratio, mask2, ids_restore2)
        else:
            (mask1, ids_restore1), = masks
            latent1, mask1, ids_restore1 = self._encode_view(
                imgs, mask_ratio, mask1, ids_restore1
            )
            latent2 = None

        pred = self.forward_decoder(latent1, ids_restore1)
        mae_loss = self.forward_loss(imgs, pred, mask1)

        if self.global_pool:
            cls1 = latent1[:, 1:, :].mean(dim=1)
            cls1 = self.fc_norm(cls1)
            cls2 = (
                self.fc_norm(latent2[:, 1:, :].mean(dim=1))
                if latent2 is not None else None
            )
        else:
            cls1 = latent1[:, 0]
            cls2 = latent2[:, 0] if latent2 is not None else None

        outputs = self.fc(cls1.detach())

        bt_loss = None
        on_diag = None
        off_diag = None
        if mi_view is None:
            bt_loss, on_diag, off_diag = self._compute_bt(
                latent1, latent2, cls1, cls2
            )

        total_loss = mae_loss + (self.bt_weight * bt_loss if bt_loss is not None else 0.0)

        return {
            "loss": total_loss,
            "mae_loss": mae_loss,
            "bt_loss": bt_loss,
            "on_diag": on_diag,
            "off_diag": off_diag,
            "pred": pred,
            "mask": mask1,
            "cls_feats": cls1,
            "outputs": outputs,
        }



def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
