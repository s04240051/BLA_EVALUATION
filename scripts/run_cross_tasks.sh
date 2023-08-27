#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=blip2_cross_task.out

module load 2022
module load Anaconda3/2022.05

source activate blip2022
cd /home/xchen/BLIP/

echo "Cross task test on coordination"
python test_blip2_bla.py  --in_context_learning --cross_dataset_example  \
                          --file_path /home/xchen/datasets/BLA/original/finetune/finetune_random/coord/test.json \
                          --example_file_path /home/xchen/datasets/BLA/original/finetune/finetune_random/rc/test.json

echo "Cross task test on relative clause"
python test_blip2_bla.py  --in_context_learning --cross_dataset_example  \
                          --file_path /home/xchen/datasets/BLA/original/finetune/finetune_random/rc/test.json \
                          --example_file_path /home/xchen/datasets/BLA/original/finetune/finetune_random/coord/test.json