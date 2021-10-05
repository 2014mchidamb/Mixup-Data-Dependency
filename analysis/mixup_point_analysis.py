#!/home/users/mc696/anaconda3/bin/python3
#SBATCH --job-name=analysis
#SBATCH -t 12:00:00
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
import numpy as np
import os
import sys
import torch

sys.path.append(os.getcwd())

from scipy.spatial.distance import cdist
from utils.data_utils import load_mnist, load_cifar10, load_cifar100


def compute_min_mixup_distance(out_file, task, train_dataset, test_dataset, subset_prop, simulation_epochs, alpha):
    """Computes min/avg angular distance between test/train data points and mixed up train data points."""
    # We will use these indices when comparing with other points to check for collisions.
    mixup_indices = {i : [] for i in range(len(train_dataset.classes))}
    mixup_points = []
    if subset_prop < 1:
        train_data = torch.utils.data.Subset(train_dataset, 
                np.random.choice(list(range(len(train_dataset))), size=int(subset_prop * len(train_dataset)), replace=False))
        test_data = torch.utils.data.Subset(test_dataset, 
                np.random.choice(list(range(len(test_dataset))), size=int(subset_prop * len(test_dataset)), replace=False))
    else:
        train_data, test_data = train_dataset, test_dataset
    print('Size of subsampled train data: {}'.format(len(train_data)), file=out_file)
    print('Size of subsampled test data: {}'.format(len(test_data)), file=out_file)

    # Generate mixup points.
    for i in range(simulation_epochs):
        for j in range(len(train_data)):
            rand_ind = np.random.randint(0, len(train_data))
            point_1, class_1 = train_data[j]
            point_2, class_2 = train_data[rand_ind]
            lam = np.random.beta(alpha, alpha)
            # This mixup point can only collide with points that are not class_1 or class_2.
            mixup_point = lam * torch.flatten(point_1).numpy() + (1 - lam) * torch.flatten(point_2).numpy()
            mixup_points.append(mixup_point)
            for key in mixup_indices.keys():
                if key != class_1 and key != class_2:
                    mixup_indices[key].append(len(mixup_points) - 1) 
    mixup_points = np.array(mixup_points)
    print('Size of Mixup array: {}'.format(len(mixup_points)))

    # Angular distance.
    def min_ang_dists(data):
        min_dists = []
        for example, label in data:
            min_dists.append(np.amin(np.arccos(1 - cdist(torch.flatten(example).numpy().reshape(1, -1), 
                mixup_points[mixup_indices[label]], 'cosine')) / np.pi))
        return min_dists

    # Compute angular distances between mixup points and train/test points.
    min_train_dists = min_ang_dists(train_data)
    min_test_dists = min_ang_dists(test_data)

    # Min and avg dists.
    min_mixup_train_dist, avg_mixup_train_dist = np.amin(min_train_dists), np.mean(min_train_dists)
    min_mixup_test_dist, avg_mixup_test_dist = np.amin(min_test_dists), np.mean(min_test_dists)

    print('{} Average Angular Distance Between Train/Mixup Points With Class Collisions: {}'.format(task, avg_mixup_train_dist), file=out_file)
    print('{} Smallest Angular Distance Between Train/Mixup Points With Class Collisions: {}'.format(task, min_mixup_train_dist), file=out_file)
    print('{} Average Angular Distance Between Test/Mixup Points With Class Collisions: {}'.format(task, avg_mixup_test_dist), file=out_file)
    print('{} Smallest Angular Distance Between Test/Mixup Points With Class Collisions: {}\n'.format(task, min_mixup_test_dist), file=out_file)


subset_prop = 0.5 # How much to subsample the data.
alpha = 1024 # Mixing parameter.
num_epochs = 1
out_file= open('runs/datasets_subset_{}_alpha_{}_epochs_{}_analysis.out'.format(subset_prop, alpha, num_epochs), 'w')

mnist_train, mnist_test = load_mnist()
compute_min_mixup_distance(out_file, 'MNIST', mnist_train, mnist_test, subset_prop=subset_prop, simulation_epochs=num_epochs, alpha=alpha)

cifar10_train, cifar10_test = load_cifar10()
compute_min_mixup_distance(out_file, 'CIFAR10', cifar10_train, cifar10_test, subset_prop=subset_prop, simulation_epochs=num_epochs, alpha=alpha)

cifar100_train, cifar100_test = load_cifar100()
compute_min_mixup_distance(out_file, 'CIFAR100', cifar100_train, cifar100_test, subset_prop=subset_prop, simulation_epochs=num_epochs, alpha=alpha)
