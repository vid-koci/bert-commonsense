# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, candidate_a, candidate_b):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. Sentence analysed with pronoun replaced for #
            candidate_a: string, correct candidate
            candidate_b: string, incorrect candidate
        """
        self.guid = guid
        self.text_a = text_a
        self.candidate_a = candidate_a
        self.candidate_b = candidate_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, type_1, type_2, masked_lm_1, masked_lm_2):
        self.input_ids_1=input_ids_1
        self.input_ids_2=input_ids_2
        self.attention_mask_1=attention_mask_1
        self.attention_mask_2=attention_mask_2
        self.type_1=type_1
        self.type_2=type_2
        self.masked_lm_1=masked_lm_1
        self.masked_lm_2=masked_lm_2


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir, set_name):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()
 
    @classmethod
    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for id_x,(sent,pronoun,candidates,candidate_a,_) in enumerate(zip(lines[0::5],lines[1::5],lines[2::5],lines[3::5],lines[4::5])):
            guid = id_x
            sent = sent.strip()
            text_a = sent.replace(' '+pronoun.strip()+' '," { ",1)
            cnd = candidates.split(",")
            cnd = (cnd[0].strip().lstrip(),cnd[1].strip().lstrip())
            candidate_a = candidate_a.strip().lstrip()
            if cnd[0]==candidate_a:
                candidate_b = cnd[1]
            else:
                candidate_b = cnd[0]
            examples.append(
                InputExample(guid, text_a, candidate_a, candidate_b))
        return examples


class DprProcessor(DataProcessor):
    """Processor for the DPR data set."""

    def get_examples(self, data_dir, set_name):
        """See base class."""
        file_names = {
                "train": "train.c.txt",
                "dev": "test.c.txt",
                "test": "wsc273.txt"
                }
        source = os.path.join(data_dir,file_names[set_name])
        logger.info("LOOKING AT {}".format(source))
        return self._create_examples(list(open(source,'r')))

class WikiProcessor(DataProcessor):
    """Processor for the Wiki data set."""

    def get_examples(self, data_dir, set_name):
        """See base class."""
        file_names = {
                "train": "train.txt",
                "dev": "dev.txt",
                "test": "test.txt"
                }
        source = os.path.join(data_dir,file_names[set_name])
        logger.info("LOOKING AT {}".format(source))
        return self._create_examples(list(open(source,'r')))


def convert_examples_to_features(examples, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.candidate_a)
        tokens_b = tokenizer.tokenize(example.candidate_b)
        tokens_sent = tokenizer.tokenize(example.text_a)
        
        tokens_1, type_1, attention_mask_1, masked_lm_1 = [],[],[],[]
        tokens_2, type_2, attention_mask_2, masked_lm_2 = [],[],[],[]
        tokens_1.append("[CLS]")
        tokens_2.append("[CLS]")
        for token in tokens_sent:
            if token in [".","!","?"]:
                tokens_1.extend([token,"[SEP]"])
                tokens_2.extend([token,"[SEP]"])
            elif token=="{":
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

        n_sep = 0
        for token in tokens_1:
            type_1.append(n_sep)
            if token=="[SEP]":
                n_sep=1#There should not be more than 2 sentences overall. If there are, ooops.
        while len(type_1)<max_seq_len:
            type_1.append(0)
        n_sep = 0
        for token in tokens_2:
            type_2.append(n_sep)
            if token=="[SEP]":
                n_sep=1#There should not be more than 2 sentences overall. If there are, ooops.
        while len(type_2)<max_seq_len:
            type_2.append(0)

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
                masked_lm_1.append(input_ids_a[0])
                input_ids_a = input_ids_a[1:]
            else:
                masked_lm_1.append(-1)
        while len(masked_lm_1)<max_seq_len:
            masked_lm_1.append(-1)

        for token in tokens_2:
            if token=="[MASK]":
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

#        if ex_index < 5:
#            logger.info("*** Example ***")
#            logger.info("guid: %s" % (example.guid))
#            logger.info("tokens_1: %s" % " ".join(
#                    [str(x) for x in tokens_1]))
#            logger.info("tokens_2: %s" % " ".join(
#                    [str(x) for x in tokens_2]))
#            logger.info("input_ids_1: %s" % " ".join([str(x) for x in input_ids_1]))
#            logger.info("input_ids_2: %s" % " ".join([str(x) for x in input_ids_2]))
#            logger.info("attention_mask_1: %s" % " ".join([str(x) for x in attention_mask_1]))
#            logger.info("attention_mask_2: %s" % " ".join([str(x) for x in attention_mask_2]))
#            logger.info("type_1: %s" % " ".join([str(x) for x in type_1]))
#            logger.info("type_2: %s" % " ".join([str(x) for x in type_2]))
#            logger.info("masked_lm_1: %s" % " ".join([str(x) for x in masked_lm_1]))
#            logger.info("masked_lm_2: %s" % " ".join([str(x) for x in masked_lm_2]))

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

def test(processor, args, tokenizer, model, device, global_step = 0, tr_loss = 0, test_set = "dev"):
    eval_examples = processor.get_examples(args.data_dir,test_set)
    eval_features = convert_examples_to_features(
        eval_examples, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids_1 = torch.tensor([f.input_ids_1 for f in eval_features], dtype=torch.long)
    all_input_ids_2 = torch.tensor([f.input_ids_2 for f in eval_features], dtype=torch.long)
    all_attention_mask_1 = torch.tensor([f.attention_mask_1 for f in eval_features], dtype=torch.long)
    all_attention_mask_2 = torch.tensor([f.attention_mask_2 for f in eval_features], dtype=torch.long)
    all_segment_ids_1 = torch.tensor([f.type_1 for f in eval_features], dtype=torch.long)
    all_segment_ids_2 = torch.tensor([f.type_2 for f in eval_features], dtype=torch.long)
    all_masked_lm_1 = torch.tensor([f.masked_lm_1 for f in eval_features], dtype=torch.long)
    all_masked_lm_2 = torch.tensor([f.masked_lm_2 for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids_1, all_input_ids_2, all_attention_mask_1, all_attention_mask_2, all_segment_ids_1, all_segment_ids_2, all_masked_lm_1, all_masked_lm_2)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids_1, input_ids_2, input_mask_1, input_mask_2, segment_ids_1, segment_ids_2, label_ids_1, label_ids_2 in eval_dataloader:
        input_ids_1 = input_ids_1.to(device)
        input_ids_2 = input_ids_2.to(device)
        input_mask_1 = input_mask_1.to(device)
        input_mask_2 = input_mask_2.to(device)
        segment_ids_1 = segment_ids_1.to(device)
        segment_ids_2 = segment_ids_2.to(device)
        label_ids_1 = label_ids_1.to(device)
        label_ids_2 = label_ids_2.to(device)

        with torch.no_grad():
            loss_1 = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = input_mask_1, masked_lm_labels = label_ids_1)
            loss_2 = model.forward(input_ids_2, token_type_ids = segment_ids_2, attention_mask = input_mask_2, masked_lm_labels = label_ids_2)
            loss = loss_1-loss_2

        tmp_eval_loss = loss.to('cpu').numpy()
        tmp_eval_accuracy = len(np.where(tmp_eval_loss<0.0)[0])

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids_1.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'global_step': global_step,
              'loss': tr_loss}

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return eval_accuracy

def filter_data(processor, args, tokenizer, model, device, test_set = "dev"):
    eval_examples = processor.get_examples(args.data_dir,test_set)
    eval_features = convert_examples_to_features(
        eval_examples, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    filtered_file = open("filtered_data.txt","w")
    all_input_ids_1 = torch.tensor([f.input_ids_1 for f in eval_features], dtype=torch.long)
    all_input_ids_2 = torch.tensor([f.input_ids_2 for f in eval_features], dtype=torch.long)
    all_attention_mask_1 = torch.tensor([f.attention_mask_1 for f in eval_features], dtype=torch.long)
    all_attention_mask_2 = torch.tensor([f.attention_mask_2 for f in eval_features], dtype=torch.long)
    all_segment_ids_1 = torch.tensor([f.type_1 for f in eval_features], dtype=torch.long)
    all_segment_ids_2 = torch.tensor([f.type_2 for f in eval_features], dtype=torch.long)
    all_masked_lm_1 = torch.tensor([f.masked_lm_1 for f in eval_features], dtype=torch.long)
    all_masked_lm_2 = torch.tensor([f.masked_lm_2 for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids_1, all_input_ids_2, all_attention_mask_1, all_attention_mask_2, all_segment_ids_1, all_segment_ids_2, all_masked_lm_1, all_masked_lm_2)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    cnt = 0
    n_filtered = 0
    for input_ids_1, input_ids_2, input_mask_1, input_mask_2, segment_ids_1, segment_ids_2, label_ids_1, label_ids_2 in eval_dataloader:
        input_ids_1 = input_ids_1.to(device)
        input_ids_2 = input_ids_2.to(device)
        input_mask_1 = input_mask_1.to(device)
        input_mask_2 = input_mask_2.to(device)
        segment_ids_1 = segment_ids_1.to(device)
        segment_ids_2 = segment_ids_2.to(device)
        label_ids_1 = label_ids_1.to(device)
        label_ids_2 = label_ids_2.to(device)

        with torch.no_grad():
            loss_1 = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = input_mask_1, masked_lm_labels = label_ids_1)
            loss_2 = model.forward(input_ids_2, token_type_ids = segment_ids_2, attention_mask = input_mask_2, masked_lm_labels = label_ids_2)
            loss = loss_1-loss_2

        tmp_eval_loss = loss.to('cpu').numpy()
        tmp_eval_accuracy = len(np.where(tmp_eval_loss<0.0)[0])

        cnt2 = 0
        for diff in tmp_eval_loss:
            num_words = len(eval_examples[cnt].text_a.split())
            num_tokens = sum(0 if token in [0,100,101,102,103] else 1 for token in input_ids_1[cnt2].cpu().data)
            if -0.05<=diff<=0.3 and float(num_words)/num_tokens>=0.8:
                filtered_file.write("{}\n[MASK]\n{},{}\n{}\n\n".format(str(eval_examples[cnt].text_a),str(eval_examples[cnt].candidate_a),str(eval_examples[cnt].candidate_b),str(eval_examples[cnt].candidate_a)))
                n_filtered+=1
            cnt+=1
            cnt2+=1
            
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids_1.size(0)
        nb_eval_steps += 1

    filtered_file.close()
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    logger.info("Size of filtered dataset: {}\n".format(n_filtered))
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return eval_accuracy




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
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
    parser.add_argument("--tolerance_param",
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
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
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
                        help="Number of updates steps to accumualte before performing a backward/update pass.")                       

    args = parser.parse_args()

    processors = {
        "dpr": DprProcessor,
        "wiki": WikiProcessor,
    }

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

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_examples(args.data_dir, "train")
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model, 
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

#TODO throw out following 2 lines or implement some better model loading
    model_dict = torch.load("dpr_x/best_model")
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
        train_features = convert_examples_to_features(
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

        model.train()
        best_accuracy = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            acc = test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps if nb_tr_steps>0 else 0)
            if acc>best_accuracy:
                best_accuracy = acc
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model"))
            for step, batch in enumerate(train_dataloader):#tqdm(train_dataloader, desc="Iteration")):#Do I really want this?
                batch = tuple(t.to(device) for t in batch)
                input_ids_1, input_ids_2, input_mask_1, input_mask_2, segment_ids_1, segment_ids_2, label_ids_1, label_ids_2 = batch
                loss_1 = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = input_mask_1, masked_lm_labels = label_ids_1)
                loss_2 = model.forward(input_ids_2, token_type_ids = segment_ids_2, attention_mask = input_mask_2, masked_lm_labels = label_ids_2)
                loss = torch.max(torch.zeros(loss_1.size(),device=device),torch.ones(loss_1.size(),device=device)*args.tolerance_param + loss_1 - loss_2)
                loss = loss.mean()
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
        #reload the best model
        model_dict = torch.load(os.path.join(args.output_dir, "best_model"))
        model.load_state_dict(model_dict)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        #test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="dev")
        #test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="test")
        filter_data(processor, args, tokenizer, model, device, test_set="train")

if __name__ == "__main__":
    main()
