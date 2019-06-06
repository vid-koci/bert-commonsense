# A Surprisingly Robust Trick for Winograd Schema Challenge

This code contains models and experiments for the paper [A Surprisingly Robust Trick for Winograd Schema Challenge](https://arxiv.org/abs/1905.06290).

The MaskedWiki datasets and pre-trained models can be downloaded from [this webpage](...). The link contains two datasets, MaskedWiki\_Sample (~2.4M examples) and MaskedWiki\_Full (~130M examples). All the experiments were conducted with the MaskedWiki\_Sample only.

The following libraries are needed to run the code: numpy, pytorch (0.4.1 or later), tqdm, boto3, nltk

To evaluate BERT, use the following script:
```
python main.py \
      --task_name wscr \
      --do_eval \
      --eval_batch_size 10 \
      --data_dir "data/" \
      --bert_model bert-large-uncased \
      --max_seq_length 128 \
      --output_dir model_output/ ```

To evaluate one of the downloaded pre-trained models, use the following code:
```
python main.py \
      --task_name wscr \
      --do_eval \
      --eval_batch_size 10 \
      --data_dir "data/" \
      --bert_model bert-large-uncased \
      --max_seq_length 128 \
      --output_dir model_output/ \
      --load_from_file models/BERT_Wiki_WscR ```

To train the BERT\_Wiki model, use the following code:
```
python main.py \
      --task_name maskedwiki \
      --do_eval \
      --do_train \
      --eval_batch_size 10 \
      --data_dir "data/" \
      --bert_model bert-large-uncased \
      --max_seq_length 128 \
      --train_batch_size 64 \
      --alpha_param 20 \
      --beta_param 0.2 \
      --learning_rate 5.0e-6 \
      --num_train_epochs 1.0 \
      --output_dir model_output/ ```

To train the BERT\_Wiki\_WscR model, download the MaskedWiki\_sample into the `data` folder and use the following code:
```
python main.py \
      --task_name wscr \
      --do_eval \
      --do_train \
      --eval_batch_size 10 \
      --data_dir "data/" \
      --bert_model bert-large-uncased \
      --max_seq_length 128 \
      --train_batch_size 64 \
      --learning_rate 1.0e-5 \
      --alpha_param 5 \
      --beta_param 0.2 \
      --num_train_epochs 30.0 \
      --output_dir $outfolder/ \
      --load_from_file models/BERT_Wiki ```

