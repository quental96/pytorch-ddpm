import os
import torch
from dataclasses import dataclass
from ddpm.config import TrainingConfig, BaseConfig
from ddpm.unet import UNet
from ddpm.diffusion import SimpleDiffusion
from ddpm.sample import reverse_diffusion


@dataclass
class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 512
    BASE_CH_MULT = (1, 2, 4, 8)  # 32, 16, 8, 4
    APPLY_ATTENTION = (False, False, True, False)
    DROPOUT_RATE = 0.1
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

    checkpoint_dir = BaseConfig.root_checkpoint_dir
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "ckpt.tar"), map_location='cpu')['model'])
    model.to(BaseConfig.DEVICE)

    sd = SimpleDiffusion(
        num_diffusion_timesteps=TrainingConfig.TIMESTEPS,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
    )

    gen_dir = BaseConfig.root_samples_dir
    os.makedirs(gen_dir, exist_ok=True)

    generation_card = 100
    for i in range(generation_card):
        save_path = os.path.join(gen_dir, str(i+1) + '.png')
        reverse_diffusion(model,
                          sd,
                          timesteps=TrainingConfig.TIMESTEPS,
                          num_images=1,
                          generate_video=False,
                          save_path=save_path,
                          img_shape=TrainingConfig.IMG_SHAPE,
                          device=BaseConfig.DEVICE)
