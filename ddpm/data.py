import torchvision.transforms as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from .helper import DeviceDataLoader


def get_dataset(dataset_name=''):

    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize((64, 64),
                      interpolation=TF.InterpolationMode.BICUBIC,
                      antialias=True),
            TF.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
        ]
    )

    if dataset_name == '':
        dataset = datasets.ImageFolder(root='/path/to/dataset', transform=transforms)

    return dataset


def get_dataloader(dataset_name='',
                   batch_size=32,
                   pin_memory=False,
                   shuffle=True,
                   num_workers=0,
                   device='cpu'):

    dataset = get_dataset(dataset_name=dataset_name)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle)

    device_dataloader = DeviceDataLoader(dataloader, device)

    return device_dataloader


def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0
