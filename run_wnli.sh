#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --time=144:00:00
#SBATCH -J WikiLong
#SBATCH --gres=gpu:8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vid.kocijan@gmail.com

python wnli_eval.py \
      --task_name Wnli \
      --eval_batch_size 1 \
      --data_dir "../data/GLUE/WNLI/" \
      --bert_model bert-large-uncased \
      --max_seq_length 128 \
      --output_dir WnliResults/
