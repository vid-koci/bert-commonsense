# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team., 2019 Intelligent Systems Lab, University of Oxford
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.getcwd()) 

import csv
import json
import logging
import argparse
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel, BertOnlyMLMHead
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from data_reader import InputExample,DataProcessor
from scorer import scorer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class BertForMaskedLM(PreTrainedBertModel):
    """BERT model with the masked language modeling head.
    
    The code is taken from pytorch_pretrain_bert/modeling.py, but the loss function has been changed to return
    loss for each example separately.
    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1,reduction='none')
            masked_lm_loss = loss_fct(prediction_scores.permute(0,2,1), masked_lm_labels)
            return torch.mean(masked_lm_loss,1)
        else:
            return prediction_scores

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, type_1, type_2, masked_lm_1, masked_lm_2):
        self.input_ids_1=input_ids_1
        self.attention_mask_1=attention_mask_1
        self.type_1=type_1
        self.masked_lm_1=masked_lm_1
        #These are only used for train examples
        self.input_ids_2=input_ids_2
        self.attention_mask_2=attention_mask_2
        self.type_2=type_2
        self.masked_lm_2=masked_lm_2

def convert_examples_to_features_train(examples, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_sent = tokenizer.tokenize(example.text_a)
        tokens_a = tokenizer.tokenize(example.candidate_a)
        tokens_b = tokenizer.tokenize(example.candidate_b)
        tokens_1, type_1, attention_mask_1, masked_lm_1 = [],[],[],[]
        tokens_2, type_2, attention_mask_2, masked_lm_2 = [],[],[],[]
        tokens_1.append("[CLS]")
        tokens_2.append("[CLS]")
        for token in tokens_sent:
            if token=="_":
                tokens_1.extend(["[MASK]" for _ in range(len(tokens_a))])
                tokens_2.extend(["[MASK]" for _ in range(len(tokens_b))])
            else:
                tokens_1.append(token)
                tokens_2.append(token)
        tokens_1 = tokens_1[:max_seq_len-1]#-1 because of [SEP]
        tokens_2 = tokens_2[:max_seq_len-1]
        if tokens_1[-1]!="[SEP]":
            tokens_1.append("[SEP]")
        if tokens_2[-1]!="[SEP]":
            tokens_2.append("[SEP]")

        type_1 = max_seq_len*[0]#We do not do any inference.
        type_2 = max_seq_len*[0]#These embeddings can thus be ignored

        attention_mask_1 = (len(tokens_1)*[1])+((max_seq_len-len(tokens_1))*[0])
        attention_mask_2 = (len(tokens_2)*[1])+((max_seq_len-len(tokens_2))*[0])

        #sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        input_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)
        #replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

        for token in tokens_1:
            if token=="[MASK]":
                if len(input_ids_a)<=0:
                    continue#broken case
                masked_lm_1.append(input_ids_a[0])
                input_ids_a = input_ids_a[1:]
            else:
                masked_lm_1.append(-1)
        while len(masked_lm_1)<max_seq_len:
            masked_lm_1.append(-1)

        for token in tokens_2:
            if token=="[MASK]":
                if len(input_ids_b)<=0:
                    continue#broken case
                masked_lm_2.append(input_ids_b[0])
                input_ids_b = input_ids_b[1:]
            else:
                masked_lm_2.append(-1)
        while len(masked_lm_2)<max_seq_len:
            masked_lm_2.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        while len(input_ids_2) < max_seq_len:
            input_ids_2.append(0)
        assert len(input_ids_1) == max_seq_len
        assert len(input_ids_2) == max_seq_len
        assert len(attention_mask_1) == max_seq_len
        assert len(attention_mask_2) == max_seq_len
        assert len(type_1) == max_seq_len
        assert len(type_2) == max_seq_len
        assert len(masked_lm_1) == max_seq_len
        assert len(masked_lm_2) == max_seq_len
        features.append(
                InputFeatures(input_ids_1=input_ids_1,
                              input_ids_2=input_ids_2,
                              attention_mask_1=attention_mask_1,
                              attention_mask_2=attention_mask_2,
                              type_1=type_1,
                              type_2=type_2,
                              masked_lm_1=masked_lm_1,
                              masked_lm_2=masked_lm_2))
    return features


def convert_examples_to_features_evaluate(examples, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.candidate_a)
        tokens_sent = tokenizer.tokenize(example.text_a)
        
        tokens_1, type_1, attention_mask_1, masked_lm_1 = [],[],[],[]
        tokens_1.append("[CLS]")
        for token in tokens_sent:
            if token=="_":
                tokens_1.extend(["[MASK]" for _ in range(len(tokens_a))])
            else:
                tokens_1.append(token)
        tokens_1 = tokens_1[:max_seq_len-1]#-1 because of [SEP]
        if tokens_1[-1]!="[SEP]":
            tokens_1.append("[SEP]")

        type_1 = max_seq_len*[0]
        attention_mask_1 = (len(tokens_1)*[1])+((max_seq_len-len(tokens_1))*[0])
        #sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        #replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)

        for token in tokens_1:
            if token=="[MASK]":
                if len(input_ids_a)<=0:
                    continue#broken case
                masked_lm_1.append(input_ids_a[0])
                input_ids_a = input_ids_a[1:]
            else:
                masked_lm_1.append(-1)
        while len(masked_lm_1)<max_seq_len:
            masked_lm_1.append(-1)
        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        assert len(input_ids_1) == max_seq_len
        assert len(attention_mask_1) == max_seq_len
        assert len(type_1) == max_seq_len
        assert len(masked_lm_1) == max_seq_len

        features.append(
                InputFeatures(input_ids_1=input_ids_1,
                              input_ids_2=None,
                              attention_mask_1=attention_mask_1,
                              attention_mask_2=None,
                              type_1=type_1,
                              type_2=None,
                              masked_lm_1=masked_lm_1,
                              masked_lm_2=None))
    return features

def test(processor, args, tokenizer, model, device, global_step = 0, tr_loss = 0, test_set = "wscr-test"):
    eval_examples = processor.get_examples(args.data_dir,test_set)
    eval_features = convert_examples_to_features_evaluate(
        eval_examples, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids_1 = torch.tensor([f.input_ids_1 for f in eval_features], dtype=torch.long)
    all_attention_mask_1 = torch.tensor([f.attention_mask_1 for f in eval_features], dtype=torch.long)
    all_segment_ids_1 = torch.tensor([f.type_1 for f in eval_features], dtype=torch.long)
    all_masked_lm_1 = torch.tensor([f.masked_lm_1 for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids_1, all_attention_mask_1, all_segment_ids_1, all_masked_lm_1)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    ans_stats=[]
    for batch in tqdm(eval_dataloader,desc="Evaluation"):
        input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = (tens.to(device) for tens in batch)
        with torch.no_grad():
            loss = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = input_mask_1, masked_lm_labels = label_ids_1)

        eval_loss = loss.to('cpu').numpy()
        for loss in eval_loss:
            curr_id = len(ans_stats)
            ans_stats.append((eval_examples[curr_id].guid,eval_examples[curr_id].ex_true,loss))
    if test_set=="gap-test":
        return scorer(ans_stats,test_set,output_file=os.path.join(args.output_dir, "gap-answers.tsv"))
    elif test_set=="wnli":
        return scorer(ans_stats,test_set,output_file=os.path.join(args.output_dir, "WNLI.tsv"))
    else:
        return scorer(ans_stats,test_set)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the files for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--alpha_param",
                        default=10,
                        type=float,
                        help="Discriminative penalty hyper-parameter.")
    parser.add_argument("--beta_param",
                        default=0.4,
                        type=float,
                        help="Discriminative intolerance interval hyper-parameter.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")                       
    parser.add_argument('--load_from_file',
                        type=str,
                        default=None,
                        help="Path to the file with a trained model. Default means bert-model is used. Size must match bert-model.")                       
            
    args = parser.parse_args()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    processor = DataProcessor()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_name = {"gap":"gap-train",
                "wikicrem":"wikicrem-train",
                "dpr":"dpr-train-small",
                "wscr":"wscr-train",
                "all":"all",
                "maskedwiki":"maskedwiki",
                }[task_name]
        
        if task_name=="all":
            train_examples = processor.get_examples(args.data_dir, "dpr-train")+processor.get_examples(args.data_dir, "gap-train")
        else:
            train_examples = processor.get_examples(args.data_dir, train_name)

        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if args.load_from_file is None:
        model = BertForMaskedLM.from_pretrained(args.bert_model, 
                    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    else:
        model = BertForMaskedLM.from_untrained(args.bert_model, 
                    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    else:
        model = torch.nn.DataParallel(model)

    if not args.load_from_file is None:
        model_dict = torch.load(args.load_from_file)
        model.load_state_dict(model_dict)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    global_step = 0
    tr_loss,nb_tr_steps = 0, 1
    if args.do_train:
        train_features = convert_examples_to_features_train(
            train_examples, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids_1 = torch.tensor([f.input_ids_1 for f in train_features], dtype=torch.long)
        all_input_ids_2 = torch.tensor([f.input_ids_2 for f in train_features], dtype=torch.long)
        all_attention_mask_1 = torch.tensor([f.attention_mask_1 for f in train_features], dtype=torch.long)
        all_attention_mask_2 = torch.tensor([f.attention_mask_2 for f in train_features], dtype=torch.long)
        all_segment_ids_1 = torch.tensor([f.type_1 for f in train_features], dtype=torch.long)
        all_segment_ids_2 = torch.tensor([f.type_2 for f in train_features], dtype=torch.long)
        all_masked_lm_1 = torch.tensor([f.masked_lm_1 for f in train_features], dtype=torch.long)
        all_masked_lm_2 = torch.tensor([f.masked_lm_2 for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids_1, all_input_ids_2, all_attention_mask_1, all_attention_mask_2, all_segment_ids_1, all_segment_ids_2, all_masked_lm_1, all_masked_lm_2)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        validation_name = {"gap":"gap-dev",
                "wikicrem":"wikicrem-dev",
                "dpr":"dpr-dev-small",
                "all":"all",
                "maskedwiki":"wscr-test",
                "wscr":"wscr-test",
                }[task_name]

        model.train()
        try:#This prevents overwriting if several scripts are running at the same time (for hyper-parameter search)
            best_accuracy = float(list(open(os.path.join(args.output_dir,"best_accuracy.txt"),'r'))[0])
        except:
            best_accuracy = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            tr_accuracy = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                input_ids_1,input_ids_2,input_mask_1,input_mask_2, segment_ids_1, segment_ids_2, label_ids_1, label_ids_2 = (tens.to(device) for tens in batch)
                           
                loss_1 = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = input_mask_1, masked_lm_labels = label_ids_1)
                loss_2 = model.forward(input_ids_2, token_type_ids = segment_ids_2, attention_mask = input_mask_2, masked_lm_labels = label_ids_2)
                loss = loss_1 + args.alpha_param * torch.max(torch.zeros(loss_1.size(),device=device),torch.ones(loss_1.size(),device=device)*args.beta_param + loss_1 - loss_2.mean())
                loss = loss.mean()
                tr_accuracy += len(np.where(loss_1.detach().cpu().numpy()-loss_2.detach().cpu().numpy()<0.0)[0])
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids_1.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1
                if not (task_name in ["wscr","gap","dpr","all"]) and global_step % 200 == 0 and (step + 1) % args.gradient_accumulation_steps == 0:#testing during an epoch
                    acc = test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps if nb_tr_steps>0 else 0, test_set=validation_name)
                    logger.info("{}\t{}\n".format(nb_tr_steps,acc))
                    model.train()
                    try:#If several processes are running in parallel this avoids overwriting results.
                        updated_accuracy = float(list(open(os.path.join(args.output_dir,"best_accuracy.txt"),'r'))[0])
                    except:
                        updated_accuracy = 0
                    best_accuracy = max(best_accuracy,updated_accuracy)
                    if acc>best_accuracy:
                        best_accuracy = acc
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model"))
                        with open(os.path.join(args.output_dir,"best_accuracy.txt"),'w') as f1_report:
                            f1_report.write("{}".format(best_accuracy))
            if validation_name=="all":
                acc = (test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps if nb_tr_steps>0 else 0, test_set = "gap-dev") +\
                        test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps if nb_tr_steps>0 else 0, test_set = "winobias-dev"))/2
            else:
                acc = test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps if nb_tr_steps>0 else 0, test_set = validation_name)
            logger.info("{}\t{}\n".format(nb_tr_steps,acc))
            model.train()
            try:
                updated_accuracy = float(list(open(os.path.join(args.output_dir,"best_accuracy.txt"),'r'))[0])
            except:
                updated_accuracy = 0
            best_accuracy = max(best_accuracy,updated_accuracy)
            if acc>best_accuracy:
                best_accuracy = acc
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model"))
                with open(os.path.join(args.output_dir,"best_accuracy.txt"),'w') as f1_report:
                    f1_report.write("{}".format(best_accuracy))
        #reload the best model
        logger.info("Best dev acc {}".format(best_accuracy))
        model_dict = torch.load(os.path.join(args.output_dir, "best_model"))
        model.load_state_dict(model_dict)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        print("GAP-test: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="gap-test"))
        print("DPR/WSCR-test: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="dpr-test"))
        print("WSC: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="wsc"))
        print("WinoGender: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="winogender"))
        _=test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="wnli")
        print("PDP: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="pdp"))
        print("WinoBias Anti Stereotyped Type 1: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="winobias-anti1"))
        print("WinoBias Pro Stereotyped Type 1: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="winobias-pro1"))
        print("WinoBias Anti Stereotyped Type 2: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="winobias-anti2"))
        print("WinoBias Pro Stereotyped Type 2: ",test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="winobias-pro2"))

if __name__ == "__main__":
    main()
