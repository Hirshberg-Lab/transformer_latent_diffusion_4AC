"""transformer based denoiser"""

import torch
from einops.layers.torch import Rearrange
from torch import nn

from tld.transformer_blocks import DecoderBlock, MLPSepConv, SinusoidalEmbedding


class DenoiserTransBlock(nn.Module):
    def __init__(
        self,
        patch_size: int,
        x_points: int, # changed "img_size" to "x_points" - the number of points in C(\omega)
        embed_dim: int,
        dropout: float,
        n_layers: int,
        mlp_multiplier: int = 4,
        n_channels: int = 1
    ):
        super().__init__()

        self.patch_size = patch_size
        self.x_points = x_points
        self.n_channels = n_channels # We have only "one channel" when we work with regular spectra
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        seq_len = int((self.x_points / self.patch_size) ) # * (self.x_points / self.patch_size))
        patch_dim = self.n_channels * self.patch_size # * self.patch_size # I removed it because now patch_dim=patch_size

        self.patchify_and_embed = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_channels, # We have one channel
                out_channels=patch_dim,      # out channels are equal to the patch_dim when n_channels=1
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("bs d l -> bs l d"), # h*w=l
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.rearrange2 = Rearrange(
            "b l (c p) -> b c (l p)", # I removed p1 and p2 and combined it to a single p
            # l=int(self.x_points / self.patch_size),
            p=self.patch_size,
        )

        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer("precomputed_pos_enc", torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=self.embed_dim,
                    mlp_multiplier=self.mlp_multiplier,
                    # note that this is a non-causal block since we are
                    # denoising the entire image no need for masking
                    is_causal=False,
                    dropout_level=self.dropout,
                    mlp_class=MLPSepConv,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, patch_dim), self.rearrange2)

    def forward(self, x, cond):
        x = self.patchify_and_embed(x)
        pos_enc = self.precomputed_pos_enc[: x.size(1)].expand(x.size(0), -1)
        x = x + self.pos_embed(pos_enc)

        for block in self.decoder_blocks:
            x = block(x, cond)

        return self.out_proj(x)


class Denoiser(nn.Module):
    def __init__(
        self,
        x_points: int, # changed "img_size" to "x_points" - the number of points in C(\omega)
        noise_embed_dims: int,
        patch_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        y_points: int = 99, # changed "text_emb_size" to "y_points" - the number of points in G(\tau)
        mlp_multiplier: int = 4,
        n_channels: int = 1
    ):
        super().__init__()

        self.x_points = x_points
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim
        self.n_channels = n_channels # We have only "one channel" when we work with regular spectra

        self.fourier_feats = nn.Sequential(
            SinusoidalEmbedding(embedding_dims=noise_embed_dims),
            nn.Linear(noise_embed_dims, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.denoiser_trans_block = DenoiserTransBlock(patch_size, x_points, embed_dim, dropout, n_layers, mlp_multiplier,n_channels)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.label_proj = nn.Linear(y_points, self.embed_dim)

    def forward(self, x, noise_level, label):
        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = self.label_proj(label) # .unsqueeze(1) # I removed this extra dim

        noise_label_emb = torch.cat([noise_level, label], dim=1)  # bs, 2, d
        noise_label_emb = self.norm(noise_label_emb)

        x = self.denoiser_trans_block(x, noise_label_emb)

        return x
