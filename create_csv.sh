#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=gpu
#
#SBATCH --mail-type='FAIL'
#SBATCH --mail-user='manon.dausort@uclouvain.be'
#SBATCH --output='/CECI/home/users/m/d/mdausort/Cytology/slurm/slurmJob_csv_creation.out'
#SBATCH --error='/CECI/home/users/m/d/mdausort/Cytology/slurm/slurmJob_csv_creation.err'

#source tl/bin/activate


python3 DBTA_csv_creation.py --csv_dir /CECI/home/users/m/d/mdausort/Cytology/Training/ --images_dir /CECI/home/users/m/d/mdausort/Cytology/Training/ -m 20 -pw 224 -ph 224