from dataclasses import dataclass, field
import torch

# @dataclass
# class DataDownloadConfig:
#     """config for downloading and processing latents"""
#     data_link: str
#     caption_col: str = "caption"
#     url_col: str = "url"
#     latent_save_path: str = "latents_folder"
#     raw_imgs_save_path: str = "raw_imgs_folder"
#     use_drive: bool = False
#     initial_csv_path: str = "imgs.csv"
#     number_sample_per_shard: int = 10000
#     image_size: int = 256
#     batch_size: int = 64
#     download_data: bool = True
#     first_n_rows: int = 1000000
#     use_wandb: bool = False

@dataclass
class SpectralDataConfig:
    omega_domain: tuple[float,float] = (0, 80.0)
    num_bumps_range: tuple[int,int] = (1,2) 
    bump_widths_fraction_range: tuple[float,float] = (0.13, 0.45)
    bump_centers_fraction_range: tuple[float,float] = (0.0, 0.6)

@dataclass
class DenoiserConfig:
    x_points: int = 1024
    noise_embed_dims: int = 256
    patch_size: int = 2
    embed_dim: int = 128
    dropout: float = 0
    n_layers: int = 3
    y_points: int = 99 
    mlp_multiplier: int = 4 
    n_channels: int = 1

@dataclass
class DeterministicConfig:
    x_points: int = 1024
    patch_size: int = 2
    embed_dim: int = 128
    dropout: float = 0
    n_layers: int = 3
    y_points: int = 99
    mlp_multiplier: int = 4
    n_channels: int = 1
    n_channels_y: int = 4

@dataclass
class DenoiserLoad:
    dtype: torch.dtype = torch.float32
    file_url: str | None = None
    local_filename: str | None = None

@dataclass
class VaeConfig:
    vae_scale_factor: float = 8
    vae_name: str = "madebyollin/sdxl-vae-fp16-fix"
    vae_dtype: torch.dtype = torch.float32

@dataclass
class ClipConfig:
    clip_model_name: str = "ViT-L/14"
    clip_dtype: torch.dtype = torch.float16

@dataclass
class DataConfig:
    """where is the latent data stored"""
    latent_path: str  
    text_emb_path: str
    val_path: str

@dataclass
class TrainConfig:
    batch_size: int = 200
    dataset_size: int = 20000 
    lr: float = 3e-4
    n_epoch: int = 100
    alpha: float = 0.999
    from_scratch: bool = True
    ##betas determine the distribution of noise seen during training
    beta_a: float = 1  
    beta_b: float = 2.5
    save_and_eval_every_iters: int = 1000
    run_id: str = ""
    model_name: str = ""
    compile: bool = True
    save_model: bool = False
    use_wandb: bool = False


@dataclass
class LTDConfig:
    """main config for inference"""
    denoiser_cfg: DenoiserConfig = field(default_factory=DenoiserConfig)
    denoiser_load: DenoiserLoad = field(default_factory=DenoiserLoad)
    # vae_cfg: VaeConfig = field(default_factory=VaeConfig)
    # clip_cfg: ClipConfig = field(default_factory=ClipConfig)


@dataclass
class ModelConfig:
    """main config for getting data, training and inference"""
    # data_config: DataConfig 
    data_config: SpectralDataConfig = field(default_factory=SpectralDataConfig)
    denoiser_config: DenoiserConfig = field(default_factory=DenoiserConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    # vae_cfg: VaeConfig = field(default_factory=VaeConfig)
    # clip_cfg: ClipConfig = field(default_factory=ClipConfig)


@dataclass
class DeterministicModelConfig:
    """main config for getting data, training and inference for the deterministic model"""
    data_config: SpectralDataConfig = field(default_factory=SpectralDataConfig)
    deterministic_config: DeterministicConfig = field(default_factory=DeterministicConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)


if __name__=='__main__':
    cfg = LTDConfig()
    print(cfg)
