#!/bin/bash
# First arg is task, second is mixup alpha, third is num runs, and fourth is subsampling.

sbatch tasks/train_models.py --task-name $1 --alpha $2 --num-runs $3 --subsample $4
