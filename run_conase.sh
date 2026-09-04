#!/bin/bash
#SBATCH --job-name=conase
#SBATCH --partition=amd-512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=conase_%j.log
#SBATCH --error=conase_%j.err

# Load Python module & activate environment
module load softwares/python/3.10.5-gnu8
source ~/mara/bin/activate

# Move to script directory and run
cd ~/conase
python process_conase.py
