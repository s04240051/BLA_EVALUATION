#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --output=flamingo_incontext_AP.out

module load 2022
module load Anaconda3/2022.05

source activate openflamingo
cd /home/hzhu/project/EMNLP_BLA/BLIP/

echo "Active Passive"
python test_blip2_bla.py --file_path /home/hzhu/project/EMNLP_BLA/datasets/BLA/original/active_passive_captions_gruen_strict.json --dataset_type whole --in_context_learning