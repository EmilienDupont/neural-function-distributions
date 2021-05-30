from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def mnist(path_to_data, batch_size=16, size=28, train=True, download=False):
    """MNIST dataloader.

    Args:
        path_to_data (string): Path to MNIST data files.
        batch_size (int):
        size (int): Size (height and width) of each image. Default is 28 for no resizing. 
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(path_to_data, train=train, download=download,
                             transform=all_transforms)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def celebahq(path_to_data, batch_size=16, size=256):
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(path_to_data, transform=all_transforms)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
