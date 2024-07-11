#! /bin/bash
#SBATCH --job-name=MLFF
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1

python main.py simulate -c /home/gridsan/wlluo/src/MLFF/scan_physnet_1000_atome_50k_missing2/scan.yaml -o /home/gridsan/wlluo/src/MLFF/test_physnet_1000_atome_50k_missing2