"""transformer based deterministic model"""


from torch import nn

from tld.denoiser import DenoiserTransBlock


class Deterministic_NN(nn.Module):
    def __init__(
        self,
        x_points: int, 
        patch_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        y_points: int = 99, 
        mlp_multiplier: int = 4,
        n_channels: int = 1,
        n_channels_y: int = 4
    ):
        super().__init__()

        self.x_points = x_points
        self.embed_dim = embed_dim
        self.n_channels = n_channels 

        self.denoiser_trans_block = DenoiserTransBlock(patch_size, x_points, embed_dim, dropout, n_layers, mlp_multiplier,n_channels)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.label_proj = nn.Linear(y_points, self.embed_dim)
        self.label_to_x = nn.Sequential(
            nn.Linear(y_points, y_points),
            nn.Conv1d(
                in_channels=n_channels_y,
                out_channels=n_channels,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.LayerNorm( y_points//patch_size ),
            nn.Linear( y_points//patch_size, x_points ),
            nn.LayerNorm(x_points),
        )

    def forward(self, label):
        x = self.label_to_x(label)
        label = self.norm(self.label_proj(label)) 

        x = self.denoiser_trans_block(x, label)

        return x
