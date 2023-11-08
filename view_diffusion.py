import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from ddpm.config import BaseConfig, TrainingConfig
from ddpm.data import get_dataloader
from ddpm.data import inverse_transform
from ddpm.diffusion import forward_diffusion, SimpleDiffusion

if __name__ == '__main__':

    sd = SimpleDiffusion(num_diffusion_timesteps=TrainingConfig.TIMESTEPS, device='cpu')

    loader = iter(
        get_dataloader(
            dataset_name=BaseConfig.DATASET,
            batch_size=6,
            device='cpu',
        )
    )

    x0s, _ = next(loader)

    noisy_images = []
    specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long)

        xts, _ = forward_diffusion(sd, x0s, timestep)
        xts = inverse_transform(xts) / 255.0
        xts = make_grid(xts, nrow=1, padding=1)

        noisy_images.append(xts)

    # Plot and see samples at different timesteps

    _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor='white')

    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.9)
    plt.axis("off")
    plt.show()
