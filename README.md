# A Surprisingly Robust Trick for Winograd Schema Challenge and WikiCREM: A Large Unsupervised Corpus for Coreference Resolution

This code contains models and experiments for the paper [A Surprisingly Robust Trick for Winograd Schema Challenge](https://arxiv.org/abs/1905.06290) and [WikiCREM: A Large Unsupervised Corpus for Coreference Resolution](https://arxiv.org/abs/1908.08025).

The MaskedWiki datasets and pre-trained models can be downloaded from [this webpage](https://ora.ox.ac.uk/objects/uuid:9b34602b-c982-4b49-b4f4-6555b5a82c3d). The link contains two datasets, MaskedWiki\_Sample (~2.4M examples) and MaskedWiki\_Full (~130M examples). All the experiments were conducted with the MaskedWiki\_Sample only.

The WikiCREM datasets and BERT\_WikiCREM model can be downloaded from [this webpage](https://ora.ox.ac.uk/objects/uuid:c83e94bb-7584-41a1-aef9-85b0e764d9e3).

The following libraries are needed to run the code: Python 3 (version 3.6 or later),  numpy (version 1.14 was used), pytorch (version 0.4.1 was used), tqdm, boto3, nltk (version 3.3 was used), requests, Spacy (version 2.0.13 was used), Spacy en\_core\_web\_lg model.

To evaluate BERT on all datasets, use the following script:
```
python main.py \
      --task_name wscr \
      --do_eval \
      --eval_batch_size 10 \
      --data_dir "data/" \
      --bert_model bert-large-uncased \
      --max_seq_length 128 \
      --output_dir model_output/
```

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
      --load_from_file models/BERT_Wiki_WscR 
```

To train the BERT\_Wiki model, use the code below.
To reproduce the exact results from the paper, use the versions of the libraries as listed in the conda environment `wsc_env.yml`.
Please note that re-training the models with different version of the libraries may yield different results. Running a full hyper-parameter search is recommended in this case.
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
      --output_dir model_output/ 
```

To train the BERT\_Wiki\_WscR model, download the MaskedWiki\_sample into the `data` folder and BERT\_Wiki model into the `models` folder. Then use the following code:
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
      --output_dir model_output/ \
      --load_from_file models/BERT_Wiki 
```
## References

```
@inproceedings{kocijan19acl,
    title     = {A Surprisingly Robust Trick for Winograd Schema Challenge},
    author    = {Vid Kocijan and
               Ana-Maria Cretu and
               Oana-Maria Camburu and
               Yordan Yordanov and
               Thomas Lukasiewicz},
    booktitle = {The 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
    address = {Florence, Italy},
    month = {July},
    year = {2019}
}
```
```
@inproceedings{kocijan19emnlp,
    title     = {WikiCREM: A Large Unsupervised Corpus for Coreference Resolution},
    author    = {Vid Kocijan and
               Ana-Maria Cretu and
               Oana-Maria Camburu and
               Yordan Yordanov and
               Phil Blunsom and
               Thomas Lukasiewicz},
    booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    address = {Hong Kong},
    month = {November},
    year = {2019}
}
```
