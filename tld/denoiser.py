"""transformer based denoiser"""

import torch
from einops.layers.torch import Rearrange
from torch import nn

from tld.transformer_blocks import DecoderBlock, MLPSepConv, SinusoidalEmbedding


class DenoiserTransBlock(nn.Module):
    def __init__(
        self,
        patch_size: int,
        x_points: int, # changed "img_size" to "x_points" - the number of points in C(\omega) (the "time" domain of x)
        freq_x_points: int, # the number of points in the freq. domain of x
        embed_dim: int,
        dropout: float,
        n_layers: int,
        mlp_multiplier: int = 4,
        n_channels: int = 2
    ):
        super().__init__()

        self.patch_size = patch_size
        self.x_points = x_points
        self.freq_x_points = freq_x_points
        self.n_channels = n_channels 
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        seq_len = int((self.x_points / self.patch_size) * (self.freq_x_points / self.patch_size))
        patch_dim = self.n_channels * self.patch_size * self.patch_size 

        self.patchify_and_embed = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_channels, 
                out_channels=patch_dim,      
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("bs d h w -> bs (h w) d"), 
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.rearrange2 = Rearrange(
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)", 
            h=int(self.freq_x_points / self.patch_size),
            p1=self.patch_size,
            p2=self.patch_size,
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
                    freq_x_points = int(self.freq_x_points / self.patch_size)
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
        x_points: int, # changed "img_size" to "x_points" - the number of points in C(\omega) (the "time" domain of x)
        freq_x_points: int, # the number of points in the freq. domain of x
        noise_embed_dims: int,
        patch_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        y_points: int, # changed "text_emb_size" to "y_points" - the number of points in G(\tau) (the "time" domain of y)
        freq_y_points: int, # the number of points in the freq. domain of y
        mlp_multiplier: int = 4,
        n_channels: int = 2
    ):
        super().__init__()

        self.x_points = x_points
        self.freq_x_points = freq_x_points
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim
        self.n_channels = n_channels # We have only "one channel" when we work with regular spectra

        self.fourier_feats = nn.Sequential(
            SinusoidalEmbedding(embedding_dims=noise_embed_dims),
            nn.Linear(noise_embed_dims, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.denoiser_trans_block = DenoiserTransBlock(patch_size, x_points, freq_x_points, embed_dim, dropout, n_layers, mlp_multiplier,n_channels)
        self.norm = nn.LayerNorm(self.embed_dim)
        # self.label_proj = nn.Linear(y_points, self.embed_dim)
        patch_dim = self.n_channels * patch_size * patch_size 
        self.patchify_and_embed_label = nn.Sequential(
            nn.Conv2d(
                in_channels = self.n_channels, 
                out_channels = patch_dim,      
                kernel_size = patch_size,
                stride = patch_size,
            ),
            Rearrange("bs d h w -> bs (h w) d"), 
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        seq_len = int((y_points / patch_size) * (freq_y_points / patch_size))
        self.pos_embed_label = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer("precomputed_pos_enc_label", torch.arange(0, seq_len).long())

    def forward(self, x, noise_level, label):
        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = self.patchify_and_embed_label(label)
        pos_enc_label = self.precomputed_pos_enc_label[: label.size(1)].expand(label.size(0), -1)
        label = label + self.pos_embed_label(pos_enc_label)

        # label = self.label_proj(label) 

        noise_label_emb = torch.cat([noise_level, label], dim=1)  # bs, 2, d
        noise_label_emb = self.norm(noise_label_emb)

        x = self.denoiser_trans_block(x, noise_label_emb)

        return x
