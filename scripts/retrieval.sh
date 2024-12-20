#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 100GB
#SBATCH --gpus=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 2-00:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/EEG_Image_decode/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/EEG_Image_decode/logs/%J_slurm.err

cd /proj/rep-learning-robotics/users/x_nonra/EEG_Image_decode/

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate BCI
export PYTHONPATH="$PYTHONPATH:$(realpath ./Retrieval)"

python Retrieval/ATMS_retrieval.py --logger True --gpu cuda:0 --output_dir ./outputs/ --average_repetitions \
--subjects sub-09 --precompute_img --img_model_type ViT-L-14 --precompute_model_name gLocal_vit-l-14_noalign --seed 42 &
python Retrieval/ATMS_retrieval.py --logger True --gpu cuda:0 --output_dir ./outputs/ --average_repetitions \
--subjects sub-10 --precompute_img --img_model_type ViT-L-14 --precompute_model_name gLocal_vit-l-14_noalign --seed 42 &
wait