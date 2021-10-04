import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset


def n_class_alt_line(num_classes=2, num_points=10, save_plot=False):
    """N class alternating line dataset."""
    points = np.linspace(0, num_points - 1, num_points).reshape((-1, 1))
    labels = [i % num_classes for i in range(num_points)]
    if save_plot:
        plt.scatter(points, np.zeros(num_points), s=22**2, c=labels)
        plt.show()
        plt.savefig('{}CAL.png'.format(num_classes))
    points = torch.tensor(points.astype(np.float32))
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(points, labels)


def load_mnist():
    """Get MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return (datasets.MNIST('data', train=True, download=True, transform=transform),
            datasets.MNIST('data', train=False, download=True, transform=transform))


def load_cifar10():
    """Get CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return (datasets.CIFAR10('data', train=True, download=True, transform=transform),
            datasets.CIFAR10('data', train=False, download=True, transform=transform))


def load_cifar100():
    """Get CIFAR-100 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    return (datasets.CIFAR100('data', train=True, download=True, transform=transform),
            datasets.CIFAR100('data', train=False, download=True, transform=transform))
