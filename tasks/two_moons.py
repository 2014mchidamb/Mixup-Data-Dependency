#!/home/users/mc696/anaconda3/bin/python3
#SBATCH --job-name=moons
#SBATCH -t 4:00:00
#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

# This code is forked from https://github.com/mohammadpz/Gradient_Starvation/blob/main/Figure_1/moon_fig_1.py.
# Thanks to the excellent authors for making their code readily available and easy to understand/use!
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch import FloatTensor as FT
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons

from torch.utils.data import DataLoader, TensorDataset
from utils.manifold_mixup import ManifoldMixupDataset, ManifoldMixupModel, ManifoldMixupLoss, MixupModule


def two_moons_dataset(seed, margin=0.0, rotation=0.0):
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
    y_pm1 = (2.0 * y - 1.0)[:, None]
    move_along_x = 0.5 * np.ones((n_samples, 1))
    move_along_y = y_pm1 * 0.3
    X = X - np.concatenate([move_along_x, move_along_y], 1)
    X[:, 1] = X[:, 1] - X[:, 1].mean()
    alter_sign = np.sign(-X[:, 1] * y_pm1[:, 0])
    X[:, 1] = X[:, 1] * alter_sign
    X[:, 1] = X[:, 1] - y_pm1[:, 0] * margin
    # Rotate data by 90 degrees
    X = X[:, ::-1].copy()

    if rotation != 0.0:
        theta = np.radians(rotation)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        X = np.dot(X, R)
    return X, y

# training for 10 different seeds and then averaging over all runs
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# A two-layer neural network with the following number of hidden units
hidden_dim = 500
n_samples = 300
epochs = 1000
delta = 0.1
alpha_1 = 128
alpha_2 = 512

experiments = [
    {'name': 'Base Model',
     'dataset': lambda seed: two_moons_dataset(seed, margin=+delta)},
    {'name': 'Mixup Model (alpha = {})'.format(alpha_1),
     'dataset': lambda seed: two_moons_dataset(seed, margin=+delta)},
    {'name': 'Mixup Model (alpha = {})'.format(alpha_2),
     'dataset': lambda seed: two_moons_dataset(seed, margin=+delta)}]

# Network architecture
class Net(nn.Module):
    def __init__(self, hidden_dim, exp=None):
        super(Net, self).__init__()

        if 'deeper' in exp['name']:
            self.fc1 = nn.Linear(2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, hidden_dim)
            self.fc5 = nn.Linear(hidden_dim, 2)

        else:
            if 'larger' in exp['name']:
                hidden_dim = hidden_dim * 10
            elif 'smaller' in exp['name']:
                hidden_dim = hidden_dim // 10

            self.fc1 = nn.Linear(2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 2)
            if 'dropout' in exp['name']:
                self.dropout_layer = nn.Dropout(p=0.7)
            elif 'batchnorm' in exp['name']:
                self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        if 'deeper' in exp['name']:
            return self.fc5(F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))))
        else:
            if 'dropout' in exp['name']:
                return self.fc2(F.relu(self.dropout_layer(self.fc1(x))))
            elif 'batchnorm' in exp['name']:
                return self.fc2(F.relu(self.bn(self.fc1(x))))
            else:
                return self.fc2(F.relu(self.fc1(x)))

# An n by n grid for the heatmap
n = 100
d1_min = -2
d1_max = 2
d2_min = -2
d2_max = 2
d1, d2 = torch.meshgrid([
    torch.linspace(d1_min, d1_max, n),
    torch.linspace(d2_min, d2_max, n)])
heatmap_plane = torch.stack((d1.flatten(), d2.flatten()), dim=1)
heatmap_avg = np.zeros((heatmap_plane.shape[0], len(experiments)))

for seed in seeds:
    print('Seed: ' + str(seed))

    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    for exp_idx, exp in enumerate(experiments):
        print('Experiment: ' + exp['name'])
        X, y = exp['dataset'](seed)
        X, y = FT(X), torch.LongTensor(y)

        net = Net(hidden_dim, exp)
        if 'Mixup' in exp['name']:
            mixup_alpha = float(exp['name'].split()[-1][:-1]) # Just get numerical value and ignore paren.
            net = ManifoldMixupModel(net, alpha=mixup_alpha) # Input mixup.
        if 'weight_decay' in exp['name']:
            optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=exp['coef'])
        elif 'adam' in exp['name']:
            optimizer = optim.Adam(net.parameters(), lr=exp['lr'])
        else:
            optimizer = optim.Adam(net.parameters(), lr=1e-3)

        if 'longer' in exp['name']:
            epochs_ = 10 * epochs
        else:
            epochs_ = epochs

        loss_fn = torch.nn.CrossEntropyLoss()
        if 'Mixup' in exp['name']:
            mixup_loss_fn = ManifoldMixupLoss(loss_fn)
            mixup_dataset = ManifoldMixupDataset(TensorDataset(X, y), same_class_only=False, num_classes=2, disclude_erm=False)
            train_loader = DataLoader(mixup_dataset, batch_size=len(X), shuffle=True)

            for epoch in range(epochs_):
                for _, (data, target) in enumerate(train_loader):
                    y_hat = net(data)
                    loss = mixup_loss_fn(y_hat, target)
                    loss.backward()
                    optimizer.step()
            print('Final loss: ', loss.item())
        else:
            for epoch in range(epochs_):
                y_hat = net(X)
                loss = loss_fn(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Final loss: ', loss.item())

        # Average heatmaps over seeds
        net.eval()
        heatmap_avg[:, exp_idx] += torch.argmin(net(heatmap_plane), dim=1).data.cpu().numpy() / len(seeds)

# Plotting
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
cm = ListedColormap(['#0365C0', '#C82506'])
figure = plt.figure(figsize=(9, 4)) #plt.figure(figsize=((len(experiments)) * 2.7, 6))
hmp_x = heatmap_plane[:, 0].data.numpy().reshape(n, n)
hmp_y = heatmap_plane[:, 1].data.numpy().reshape(n, n)

for exp_idx, exp in enumerate(experiments):
    ax = plt.subplot(1, len(experiments), exp_idx + 1)
    # plot only one of the seeds
    X, y = exp['dataset'](seeds[0])
    hma = heatmap_avg[:, exp_idx].reshape(n, n)
    ax.contourf(hmp_x, hmp_y, hma, 1, cmap=plt.cm.RdBu, alpha=.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm, edgecolors='k', s=18)
    ax.axhline(y=0, ls='--', lw=0.7, color='k', alpha=0.5)
    ax.axvline(x=0, ls='--', lw=0.7, color='k', alpha=0.5)
    ax.set_title(exp['name'])
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

plt.tight_layout()
plt.savefig('plots/mixup_moons_alpha_{}_{}_delta_{}.png'.format(alpha_1, alpha_2, delta))
