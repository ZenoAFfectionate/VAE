import torch
import torchvision
from torchvision import transforms


def get_dataset(dataset_name="MNIST", batch_size=None):
    print(f'> Loading the dataset {dataset_name}')

    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten images
    ])

    if dataset_name == "MNIST":
        n_classes = 10
        train_set = torchvision.datasets.MNIST(
            root='./data', train=True, transform=transform)
        test_set = torchvision.datasets.MNIST(
            root='./data', train=False, transform=transform)

    if dataset_name == "CIFAR10":
        n_classes = 10
        train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, transform=transform)

    if (dataset_name == 'CIFAR100'):
        n_classes = 100
        train_set = torchvision.datasets.CIFAR100(
            root='./data', train=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(
            root='./data', train=False, transform=transform)

    # create data loader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True)

    return train_loader, test_loader, n_classes

