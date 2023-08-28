#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --output=blip2_prompt_test.out

module load 2022
module load Anaconda3/2022.05

source activate openflamingo
cd /home/hzhu/project/EMNLP_BLA/BLIP/

echo "Active Passive"
# python test_blip2_bla.py --file_path /home/xchen/datasets/BLA/original/finetune/finetune_random/active_passive/test.json

# echo "Coordination"
# python test_blip2_bla.py --file_path /home/xchen/datasets/BLA/original/finetune/finetune_random/coord/test.json

echo "Relative Clause"
python test_blip2_bla.py --file_path /home/hzhu/project/EMNLP_BLA/datasets/BLA/original/finetune/finetune_random/rc/test.json
