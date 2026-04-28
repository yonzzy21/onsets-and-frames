#!/bin/bash
#SBATCH --job-name=guitarset_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1          # Request 1 GPU (e.g. RTX 8000 or V100)
#SBATCH --mem=32GB            # Request 32GB memory
#SBATCH --time=24:00:00       # Time limit (24 hours)
#SBATCH --output=slurm_%j.out # Standard output and error log

# Load Anaconda module (standard for NYU Greene HPC)
module purge
module load anaconda3/2020.07

# Initialize conda
eval "$(conda shell.bash hook)"

# Create a conda environment named 'onsets' if it doesn't exist
conda create -n onsets python=3.9 -y
conda activate onsets

# Install dependencies
pip install -r requirements.txt

# Run the training script. 
# Because this project uses the `sacred` library, arguments are passed with the `with` keyword.
echo "Starting training..."
# You can also pass the dataset path directly if it's stored elsewhere on the HPC!
# Replace '/path/to/your/hpc/GuitarSet' with your actual dataset folder.
# 'sequence_length=32000' enforces exactly 2-second chunks (16,000 Hz * 2 seconds)
python train.py with train_on="GuitarSet" audio_features="cqt" dataset_dir="/path/to/your/hpc/GuitarSet" sequence_length=32000
