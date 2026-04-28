#!/bin/bash
#SBATCH --job-name=guitarset_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1          # Request 1 GPU (e.g. RTX 8000 or V100)
#SBATCH --mem=32GB            # Request 32GB memory
#SBATCH --time=6:00:00        # Time limit (6 hours as requested)
#SBATCH --output=slurm_%j.out # Standard output and error log

# Start the Singularity container and pass the commands to run inside it
singularity exec --nv \
  --overlay /scratch/yd3245/training_project/overlay-25GB.ext3:ro \
  --bind /scratch/yd3245:/scratch \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    # 1. Initialize Conda from the overlay
    source /ext3/miniconda3/bin/activate
    conda activate base

    # 3. Run the training script. 
    # Because this project uses the `sacred` library, arguments are passed with the `with` keyword.
    echo 'Starting training...'
    python train.py with \
      train_on='GuitarSet' \
      audio_features='mel' \
      dataset_dir='/scratch/yd3245/coloqui/Guitarset_FromMirdata' \
      sequence_length=32000
"
