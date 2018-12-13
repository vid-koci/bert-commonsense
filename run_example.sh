#!/bin/bash
#SBATCH --partition=big
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -J DPR1
#SBATCH --gres=gpu:8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vid.kocijan@gmail.com

export GLUE_DIR="../data/GLUE/"

#            --do_train \
python train.py \
      --task_name Wiki \
          --do_eval \
            --eval_batch_size 128 \
                --data_dir "../data/Wiki_bert/" \
                  --bert_model bert-large-uncased \
                    --max_seq_length 128 \
                      --train_batch_size 32 \
                        --learning_rate 2.0e-5 \
                          --num_train_epochs 1.0 \
                            --output_dir dpr_1/ \
                              --tolerance_param 0.4
