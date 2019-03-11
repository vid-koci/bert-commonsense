#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --time=144:00:00
#SBATCH -J WikiLong
#SBATCH --gres=gpu:8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vid.kocijan@gmail.com

#      --do_train \

python wsc_train.py \
      --task_name Dpr \
      --do_eval \
      --eval_batch_size 10 \
      --data_dir "../data/Test/" \
      --bert_model bert-large-uncased \
      --max_seq_length 128 \
      --train_batch_size 64 \
      --learning_rate 2.0e-5 \
      --num_train_epochs 50.0 \
      --output_dir dpr_results/ \
      --tolerance_param 0.1 \
      --penalty_param 5 
