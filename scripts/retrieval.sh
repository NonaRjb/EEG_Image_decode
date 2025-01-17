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

data_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/Preprocessed_data_250Hz"
precompute_model_name="gLocal_openclip_vit-l-14_laion2b_s32b_b82k"
model_type="ViT-L-14"

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate BCI
export PYTHONPATH="$PYTHONPATH:$(realpath ./Retrieval)"

python Retrieval/ATMS_retrieval.py --data_path "$data_path" --logger True --gpu cuda:0 --output_dir ./outputs/ \
--subjects sub-02 --insubject True --precompute_img --img_model_type "$model_type" --precompute_model_name "$precompute_model_name" --seed 42 &
python Retrieval/ATMS_retrieval.py --data_path "$data_path" --logger True --gpu cuda:0 --output_dir ./outputs/ \
--subjects sub-02 --insubject True --precompute_img --img_model_type "$model_type" --precompute_model_name "${precompute_model_name}_noalign" --seed 42 &
wait