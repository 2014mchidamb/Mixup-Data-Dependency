# Mixup-Data-Dependency
Code associated with the paper "Towards Understanding the Data Dependency of Mixup-style Training".

## Running Alternating Line Experiments
In order to generate the plots found in Section 2.3 ("A Mixup Failure Case"), one can run the following command 
for different values of alpha.

```
python3 tasks/train_models.py --task-name NCAL --alpha 128 --num-runs 10
```

If running using slurm, it is also possible to just run:

```
./tasks/run_task_with_erm.sh NCAL 128 10 0
```

The generated output files can be found under `runs/` and `plots/` with file names based on the provided parameters.

## Running Image Classification Experiments
In order to generate the plots found in Section 2.4 ("Sufficient Conditions for Minimizing the Original Risk"), one can run
the following commands for different values of alpha.

```
python3 tasks/train_models.py --task-name MNIST --alpha 1024 --num-runs 5
python3 tasks/train_models.py --task-name CIFAR10 --alpha 1024 --num-runs 5
python3 tasks/train_models.py --task-name CIFAR100 --alpha 1024 --num-runs 5
```

Once again, if running using slurm it is possible to instead run `./tasks/run_task_with_erm.sh` with the
same arguments as above and an additional fourth argument set to 0. As before, output files can be found in `runs/`
and `plots/`.

## Running Angular Distance Analysis
To recreate the approximate epsilon computation found in Section 2.4 (in the discussion of application of sufficient conditions), one
can run the following command after manually setting `subset_prop` and `alpha` in `analysis/mixup_point_analysis.py`.

```
python3 analysis/mixup_point_analysis.py
```

## Running Two Moons Experiments
To recreate the two moons experiments found in Section 3.1 ("The Margin of Mixup Classifiers"), set `alpha_1` and `alpha_2` 
in `tasks/two_moons/py` to the mixing parameters to be compared and then run the following command.

```
python3 tasks/two_moons.py
```
