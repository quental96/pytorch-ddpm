import os
import gc
import torch
from torch import nn
from torch.cuda import amp
from dataclasses import dataclass
from ddpm.config import TrainingConfig, BaseConfig
from ddpm.unet import UNet
from ddpm.diffusion import SimpleDiffusion
from ddpm.data import get_dataloader
from ddpm.helper import setup_log_directory
from ddpm.train import train_one_epoch
from ddpm.sample import reverse_diffusion


@dataclass
class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 512
    BASE_CH_MULT = (1, 2, 4, 8)  # 32, 16, 8, 4
    APPLY_ATTENTION = (False, False, True, False)
    DROPOUT_RATE = 0.3
    TIME_EMB_MULT = 2  # 128


if __name__ == '__main__':

    model = UNet(
        input_channels=TrainingConfig.IMG_SHAPE[0],
        output_channels=TrainingConfig.IMG_SHAPE[0],
        base_channels=ModelConfig.BASE_CH,
        base_channels_multiples=ModelConfig.BASE_CH_MULT,
        apply_attention=ModelConfig.APPLY_ATTENTION,
        dropout_rate=ModelConfig.DROPOUT_RATE,
        time_multiple=ModelConfig.TIME_EMB_MULT,
    )
    model.to(BaseConfig.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)

    dataloader = get_dataloader(
        dataset_name=BaseConfig.DATASET,
        batch_size=TrainingConfig.BATCH_SIZE,
        device=BaseConfig.DEVICE,
        pin_memory=True,
        num_workers=TrainingConfig.NUM_WORKERS,
    )

    loss_fn = nn.MSELoss()

    sd = SimpleDiffusion(
        num_diffusion_timesteps=TrainingConfig.TIMESTEPS,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
    )

    scaler = amp.GradScaler()

    total_epochs = TrainingConfig.NUM_EPOCHS + 1
    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

    generate_video = False
    ext = ".mp4" if generate_video else ".png"

    for epoch in range(1, total_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        # Algorithm 1: Training
        train_one_epoch(model, sd, dataloader, optimizer, scaler, loss_fn, epoch=epoch)

        if epoch % 100 == 0:
            save_path = os.path.join(log_dir, f"{epoch}{ext}")

            # Algorithm 2: Sampling
            reverse_diffusion(model, sd, timesteps=TrainingConfig.TIMESTEPS, num_images=8,
                              generate_video=generate_video,
                              save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE)

            # clear_output()
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model": model.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
            del checkpoint_dict
