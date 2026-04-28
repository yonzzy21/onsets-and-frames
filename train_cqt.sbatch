#!/bin/bash

#SBATCH --job-name=guitar-cqt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB
#SBATCH --time=6:00:00
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_1003_general
#SBATCH --gres=gpu:1

module purge

singularity exec --nv \
    --bind /scratch:/scratch \
    --overlay /scratch/yd3245/coloqui/overlay-25GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
    source /ext3/miniconda3/bin/activate
    python train.py with \
      train_on='GuitarSet' \
      audio_features='cqt' \
      dataset_dir='/scratch/yd3245/coloqui/Guitarset_FromMirdata' \
      sequence_length=32000
"
