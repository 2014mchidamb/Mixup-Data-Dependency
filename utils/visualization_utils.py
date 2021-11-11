import matplotlib
import matplotlib.pyplot as plt


def plot_mixup_error(task, mixup_alpha, num_runs, num_epochs, mixup_avg_errors, mixup_error_std, 
        base_avg_errors, base_error_std, no_erm=False, test_interval=0):
    full_task_name = task
    if task == 'NCAL':
        full_task_name = 'Alternating Line'
    if no_erm:
        plot_title = r"Error Curves, {} ($\alpha$ = {}, {} Runs, No Same Point)".format(full_task_name, mixup_alpha, num_runs)
        image_title = 'plots/{}_alpha_{}_runs_{}_no_erm.png'.format(task, mixup_alpha, num_runs)
    else:
        plot_title = r"Error Curves, {} ($\alpha$ = {}, {} Runs)".format(full_task_name, mixup_alpha, num_runs)
        image_title = 'plots/{}_alpha_{}_runs_{}.png'.format(task, mixup_alpha, num_runs)

    # Create error plot.
    plt.figure(figsize=(9, 7))
    plt.rc('axes', titlesize=18, labelsize=18)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=18)
    plt.rc('figure', titlesize=20)

    plt.title(plot_title)
    plt.xlabel('Epochs')
    plt.ylabel('Training Error')

    x_epochs = [i for i in range(1, num_epochs + 1) if (test_interval == 0 or i % test_interval == 0)]
    plt.plot(x_epochs, base_avg_errors, label='ERM', color='C0')
    plt.plot(x_epochs, mixup_avg_errors, label='Mixup', color='C1')
    plt.fill_between(x_epochs, base_avg_errors - base_error_std, base_avg_errors + base_error_std, facecolor='C0', alpha=0.3)
    plt.fill_between(x_epochs, mixup_avg_errors - mixup_error_std, mixup_avg_errors + mixup_error_std, facecolor='C1', alpha=0.3)

    plt.legend(loc="lower left")

    plt.savefig(image_title)
