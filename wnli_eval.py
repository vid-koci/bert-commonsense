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

#Parameters for the filtering model: trained on DPR dataset with parameters tolerance param 0.4 and penalty param 20 and lr 2.0e-5

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
import pickle

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel, BertOnlyMLMHead
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import wnli_utils

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

    def __init__(self, guid, text_a, candidate_a, candidates_b):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. Sentence analysed with pronoun replaced for #
            candidate_a: string, correct candidate
            candidates_b: list of strings, incorrect candidates
        """
        self.guid = guid
        self.text_a = text_a
        self.candidate_a = candidate_a
        self.candidates_b = candidates_b


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
            text_a = sent.replace(' '+pronoun.strip()+' '," _ ",1)
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

class WnliProcessor(DataProcessor):
    """Processor for the Wiki data set."""

    def get_examples(self, data_dir, set_name):
        """See base class."""
        file_names = {
                "train": "train.tsv",
                "dev": "dev.tsv",
                "test": "test.tsv"
                }
        source = os.path.join(data_dir,file_names[set_name])
        checkpoint = os.path.join(data_dir,set_name+"_checkpoint.pickle")
        if os.path.isfile(checkpoint):
            return pickle.load(open(checkpoint,'rb'))
        logger.info("LOOKING AT {}".format(source))
        examples=[]
        for line in tqdm(list(open(source,'r'))[1:]):
            tokens = line.strip().split('\t')
            guid = tokens[0]
            premise = tokens[1]
            hypothesis = tokens[2]
            if len(tokens)>3:
                label = tokens[3]
            else:
                label=None
            premise,candidates = wnli_utils.transform_wnli(premise,hypothesis)
            if premise==None:
                if set_name!="train":
                    if label=="0":
                        guid="-"+guid
                    examples.append(InputExample("#"+guid,"","",[]))
                continue
            candidate_a = candidates[0]
            candidates_b = candidates[1:]
            if label=="1":
                examples.append(InputExample(guid,premise,candidate_a,candidates_b))
            elif set_name!="train":
                if label is None:
                    examples.append(InputExample(guid,premise,candidate_a,candidates_b))
                else:
                    examples.append(InputExample("-"+guid,premise,candidate_a,candidates_b))
        pickle.dump(examples,open(checkpoint,'wb'))
        return examples

def convert_examples_to_features(examples, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.candidate_a)
        tokens_b = [tokenizer.tokenize(ex) for ex in example.candidates_b]
        tokens_sent = tokenizer.tokenize(example.text_a.replace("[MASK]","_"))
        
        tokens_1, type_1, attention_mask_1, masked_lm_1 = [],[],[],[]
        tokens_2, type_2, attention_mask_2, masked_lm_2 = [[] for _ in range(len(example.candidates_b))],[[] for _ in range(len(example.candidates_b))],[[] for _ in range(len(example.candidates_b))],[[] for _ in range(len(example.candidates_b))]
        tokens_1.append("[CLS]")
        for i in range(len(tokens_2)):
            tokens_2[i].append("[CLS]")
        for token in tokens_sent:
            if token in [".","!","?"]:
                tokens_1.extend([token,"[SEP]"])
                for i in range(len(example.candidates_b)):
                    tokens_2[i].extend([token,"[SEP]"])
            elif token=="_":
                tokens_1.extend(["[MASK]" for _ in range(len(tokens_a))])
                for i in range(len(example.candidates_b)):
                    tokens_2[i].extend(["[MASK]" for _ in range(len(tokens_b[i]))])
            else:
                tokens_1.append(token)
                for i in range(len(example.candidates_b)):
                    tokens_2[i].append(token)
        tokens_1 = tokens_1[:max_seq_len-1]#-1 because of [SEP]
        tokens_2 = [seq[:max_seq_len-1] for seq in tokens_2]
        if tokens_1[-1]!="[SEP]":
            tokens_1.append("[SEP]")
        for i in range(len(tokens_2)):
            if tokens_2[i][-1]!="[SEP]":
                tokens_2[i].append("[SEP]")

        n_sep = 0
        for token in tokens_1:
            type_1.append(n_sep)
            if token=="[SEP]":
                n_sep=1#There should not be more than 2 sentences overall. If there are, ooops.
        while len(type_1)<max_seq_len:
            type_1.append(0)
        for i in range(len(tokens_2)):
            n_sep = 0
            for token in tokens_2[i]:
                type_2[i].append(n_sep)
                if token=="[SEP]":
                    n_sep=1#There should not be more than 2 sentences overall. If there are, ooops.
            while len(type_2[i])<max_seq_len:
                type_2[i].append(0)

        attention_mask_1 = (len(tokens_1)*[1])+((max_seq_len-len(tokens_1))*[0])
        attention_mask_2 = [(len(tokens_2[i])*[1])+((max_seq_len-len(tokens_2[i]))*[0]) for i in range(len(tokens_2))]

        #sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        input_ids_2 = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_2]
        #replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_b]

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

        for i in range(len(tokens_2)):
            for token in tokens_2[i]:
                if token=="[MASK]":
                    if len(input_ids_b[i])<=0:
                        continue#broken case
                    masked_lm_2[i].append(input_ids_b[i][0])
                    input_ids_b[i] = input_ids_b[i][1:]
                else:
                    masked_lm_2[i].append(-1)
            while len(masked_lm_2[i])<max_seq_len:
                masked_lm_2[i].append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        for i in range(len(input_ids_2)):
            while len(input_ids_2[i]) < max_seq_len:
                input_ids_2[i].append(0)
        assert len(input_ids_1) == max_seq_len
        for ii2 in input_ids_2:
            assert len(ii2) == max_seq_len
        assert len(attention_mask_1) == max_seq_len
        for am2 in attention_mask_2:
            assert len(am2) == max_seq_len
        assert len(type_1) == max_seq_len
        for t2 in type_2:
            assert len(t2) == max_seq_len
        assert len(masked_lm_1) == max_seq_len
        for ml2 in masked_lm_2:
            assert len(ml2) == max_seq_len
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
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    if test_set =="test":
        result_file = open(os.path.join(args.output_dir, "WNLI.tsv"),'w')
        result_file.write("index\tprediction\n")
    for step in range(len(eval_features)):
        if eval_examples[step].guid[0]=="#":#Could not read the example, will guess 0 (majority baseline)
            if test_set=="test":
                result_file.write("{}\t0\n".format(eval_examples[step].guid[1:]))
            if eval_examples[step].guid[1]=="-":
                eval_accuracy+=1
            nb_eval_steps+=1
            continue
        input_ids_1 = torch.tensor([eval_features[step].input_ids_1],dtype=torch.long)
        input_ids_1.to(device)
        input_ids_2 = torch.tensor(eval_features[step].input_ids_2,dtype=torch.long)
        input_ids_2.to(device)
        input_mask_1 = torch.tensor([eval_features[step].attention_mask_1],dtype=torch.long)
        input_mask_1.to(device)
        input_mask_2 = torch.tensor(eval_features[step].attention_mask_2,dtype=torch.long)
        input_mask_2.to(device)
        segment_ids_1 = torch.tensor([eval_features[step].type_1],dtype=torch.long)
        segment_ids_1.to(device)
        segment_ids_2 = torch.tensor(eval_features[step].type_2,dtype=torch.long)
        segment_ids_2.to(device)
        label_ids_1 = torch.tensor([eval_features[step].masked_lm_1],dtype=torch.long)
        label_ids_1.to(device)
        label_ids_2 = torch.tensor(eval_features[step].masked_lm_2,dtype=torch.long)
        label_ids_2.to(device)

        with torch.no_grad():
            loss_1 = model.forward(input_ids_1, token_type_ids = segment_ids_1, attention_mask = input_mask_1, masked_lm_labels = label_ids_1)
            loss_2 = model.forward(input_ids_2, token_type_ids = segment_ids_2, attention_mask = input_mask_2, masked_lm_labels = label_ids_2)
            loss = loss_1.min()-loss_2.min()

        tmp_eval_loss = loss.to('cpu').numpy()
        tmp_eval_accuracy=0
        if test_set=="test":
            if eval_examples[step].guid[0]=="-":
                ex_id = eval_examples[step].guid[1:]
            else:
                ex_id = eval_examples[step].guid
            if tmp_eval_loss > 0:
                prediction = 0
            else:
                prediction = 1
            result_file.write("{}\t{}\n".format(ex_id,prediction))
            continue
        if eval_examples[step].guid[0]=="-":
            if tmp_eval_loss>0.0:
                tmp_eval_accuracy=1
        else:
            if tmp_eval_accuracy<0.0:
                tmp_eval_accuracy+=1

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids_1.size(0)
        nb_eval_steps += 1

    if test_set=="test":
        result_file.close()
        return
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
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    processors = {
        "wnli": WnliProcessor
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None

    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model, 
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    else:#if n_gpu > 1:
        model = torch.nn.DataParallel(model)

#TODO throw out following 2 lines or implement some better model loading
    model_dict = torch.load("WikiWscrModel/best_model")
    model.load_state_dict(model_dict)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    global_step = 0
    tr_loss,nb_tr_steps = 0, 1
    test(processor, args, tokenizer, model, device, global_step = global_step, tr_loss = tr_loss/nb_tr_steps, test_set="test")

if __name__ == "__main__":
    main()
