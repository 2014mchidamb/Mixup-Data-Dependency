#!/home/users/mc696/anaconda3/bin/python3
#SBATCH --job-name=train_models
#SBATCH -t 12:00:00
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
import argparse
import numpy as np
import os
import sys
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
from utils.data_utils import load_mnist, load_cifar10, load_cifar100, n_class_alt_line
from utils.manifold_mixup import ManifoldMixupDataset, ManifoldMixupModel, ManifoldMixupLoss, MixupModule
from utils.resnet import ResNet18
from utils.training_utils import full_train_test_loop
from utils.visualization_utils import plot_mixup_error


# Set up commandline arguments.
parser = argparse.ArgumentParser(description='Hyperparameters for model training.')
parser.add_argument('--task-name', dest='task_name', default='CIFAR10', type=str)
parser.add_argument('--alpha', dest='mixup_alpha', default=16, type=float)
parser.add_argument('--no-erm', dest='no_erm', action='store_true')
parser.add_argument('--num-runs', dest='num_runs', default=1, type=int)
parser.add_argument('--subsample', dest='subsample', default=5, type=int)
parser.set_defaults(no_erm=False)
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device != 'cpu':
    print('Device count: ', torch.cuda.device_count())
    print('GPU being used: {}'.format(torch.cuda.get_device_name(0)))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

with_erm = 'with_erm'
if args.no_erm:
    with_erm = 'no_erm'
perf_file = open('runs/{}_alpha_{}_{}.out'.format(args.task_name, args.mixup_alpha, with_erm), 'w')

out_dim = 10
not_NCAL = True # There are a lot of alternating line specific settings.
if args.task_name == 'MNIST':
    train_data, test_data = load_mnist()
elif args.task_name == 'CIFAR10':
    train_data, test_data = load_cifar10()
elif args.task_name == 'CIFAR100':
    train_data, test_data = load_cifar100()
    out_dim = 100
elif args.task_name == 'NCAL':
    train_data = n_class_alt_line(num_classes=2, num_points=3)
    test_data = train_data
    out_dim = 2
    not_NCAL = False
else:
    sys.exit('Invalid task name.')
in_dim = np.reshape(train_data.data, (train_data.data.shape[0], -1)).shape[1] if not_NCAL else 1

# Model/training parameters.
mixup_alpha = args.mixup_alpha
h_dim = 512
num_hidden = 1
lr = 1e-3
epochs = 50 if not_NCAL else 1000
batch_size = 128
num_runs = args.num_runs
test_interval = 0 if not_NCAL else 10 # How often to append error.

print('The following model/training parameters were used for this run: \n', file=perf_file)
print('batch_size = ', batch_size, file=perf_file)
print('mixup_alpha = ', mixup_alpha, file=perf_file)
print('h_dim = ', h_dim, file=perf_file)
print('num_hidden = ', num_hidden, file=perf_file)
print('lr = ', lr, file=perf_file)
print('epochs = ', epochs, file=perf_file)
print('num_runs = ', num_runs, file=perf_file)
print('-------------------------------------------------\n', file=perf_file)

# Subsample as necessary.
if args.subsample > 0:
    train_data = torch.utils.data.Subset(train_data, 
            np.random.choice(list(range(len(train_data))), size=args.subsample, replace=False))
    test_data = torch.utils.data.Subset(test_data, 
            np.random.choice(list(range(len(test_data))), size=args.subsample, replace=False))
mixup_train = ManifoldMixupDataset(train_data, same_class_only=False, num_classes=10, disclude_erm=args.no_erm)

# Prepare mixup and same class mixup data.
base_dl = DataLoader(train_data, batch_size=batch_size, shuffle=not_NCAL)
mixup_dl = DataLoader(mixup_train, batch_size=batch_size, shuffle=not_NCAL)
test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Set up models.
if not_NCAL:
    base_model = ResNet18(out_dim, is_mnist=(args.task_name == 'MNIST')).to(device)
else:
    layers = [torch.nn.Flatten(), torch.nn.Linear(in_dim, h_dim), torch.nn.ReLU()]
    for i in range(num_hidden - 1):
        layers.append(torch.nn.Linear(h_dim, h_dim))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(h_dim, out_dim))
    base_model = torch.nn.Sequential(*layers).to(device)
mixup_model = ManifoldMixupModel(base_model, alpha=mixup_alpha).to(device) # Input mixup.

criterion = torch.nn.CrossEntropyLoss()
mixup_criterion = ManifoldMixupLoss(criterion)

base_optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)
mixup_optimizer = torch.optim.Adam(mixup_model.parameters(), lr=lr)

mixup_avg_errors, mixup_error_std, mixup_evals = full_train_test_loop(mixup_model, test_dl, criterion, mixup_dl, mixup_criterion, mixup_optimizer,
                     epochs, batch_size, 'Mixup', perf_file, base_dl, 0, device, True, num_runs, evals=(not not_NCAL), test_interval=test_interval)

base_avg_errors, base_error_std, base_evals = full_train_test_loop(base_model, test_dl, criterion, base_dl, criterion, base_optimizer,
                     epochs, batch_size, 'Base', perf_file, base_dl, 0, device, True, num_runs, evals=(not not_NCAL), test_interval=test_interval)

# Create error plot.
plot_mixup_error(args.task_name, mixup_alpha, num_runs, epochs, 
        mixup_avg_errors, mixup_error_std, base_avg_errors, base_error_std, args.no_erm, test_interval)

# Print evaluations for alternating line.
if not not_NCAL:
    print('Mixup Model Evaluations: ', file=perf_file)
    print(mixup_evals, file=perf_file)
