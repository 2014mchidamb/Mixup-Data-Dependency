import numpy as np
import pandas as pd
import torch


def get_grad_norm(model):
    grad_norm = 0
    for p in model.parameters():
        grad_norm += p.grad.data.norm(2).item() ** 2
    return grad_norm ** 0.5


def reset_weights(m):
    reset_parameters = getattr(m, 'reset_parameters', None)
    if callable(reset_parameters):
        m.reset_parameters()


def get_model_param_tensor(model):
    flattened_params = []
    for param_tensor in model.parameters():
        flattened_params.append(torch.flatten(param_tensor))
    return torch.cat(flattened_params)


def get_model_evaluations(model, data_loader, device='cpu'):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    output = None
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = softmax(model(data))
    return output


def train(model, train_loader, loss_fn, optimizer, epoch, batch_size, out_file, log_interval=20, device='cpu'):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if isinstance(data, list): 
            # This means we're mixing two examples.
            for i in range(len(data)):
                data[i] = data[i].to(device)
                target[i] = target[i].to(device)
        else:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if log_interval > 0 and batch_idx % log_interval == 0:
            # Compute average gradient norm for parameters as well.
            grad_norm = get_grad_norm(model)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Avg Gradient Norm: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), grad_norm), file=out_file)


def test(model, test_loader, loss_fn, out_file, device='cpu', return_error=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).sum().item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if return_error:
        return test_loss, 1 - (correct / len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), file=out_file)


def get_conf_matrix(model, test_loader, num_classes, device='cpu'):
    model.eval()
    conf_matrix = [[0 for i in range(num_classes)] for j in range(num_classes)]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(len(target)):
                conf_matrix[int(target[i])][int(pred[i])] += 1

    return pd.DataFrame(conf_matrix)


def full_train_test_loop(
        model, test_loader, test_loss_fn, train_loader, loss_fn, optimizer, num_epochs, batch_size,
        model_name, out_file, val_loader=None, log_interval=20, device='cpu', return_errors=True, num_runs=10, test_interval=0, evals=False):
    full_training_errors = []
    full_model_evals = []
    avg_test_error = 0
    print('{} model training loss results: '.format(model_name), file=out_file)
    for i in range(num_runs):
        training_errors = []
        for j in range(1, num_epochs + 1):
            train(model, train_loader, loss_fn, optimizer, j, batch_size, out_file, log_interval, device)
            if test_interval == 0 or j % test_interval == 0:
                _, err = test(model, val_loader, test_loss_fn, out_file, device, return_error=True)
                training_errors.append(err)
        full_training_errors.append(training_errors)
        avg_test_error += 100 * test(model, test_loader, test_loss_fn, out_file, device, return_error=True)[1] / num_runs 
        if evals:
            model_evals = get_model_evaluations(model, test_loader, device=device).cpu().numpy()
            full_model_evals.append(model_evals)
        model.apply(reset_weights)
    print('-------------------------------------------------\n', file=out_file)

    # Final test performance.
    print('{} average test error over {} runs: {:.2f}%'.format(model_name, num_runs, avg_test_error), file=out_file)
    print('-------------------------------------------------\n', file=out_file)

    return np.mean(full_training_errors, axis=0), np.std(full_training_errors, axis=0), np.mean(full_model_evals, axis=0)

