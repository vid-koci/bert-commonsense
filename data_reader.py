import os
from tqdm import trange,tqdm
import wnli_utils

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, candidate_a, candidate_b, ex_true=True):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. Sentence analysed with pronoun replaced for _
            candidate_a: string, correct candidate
            candidate_b: string, incorrect candidate
        """
        self.guid = guid
        self.text_a = text_a
        self.candidate_a = candidate_a
        self.candidate_b = candidate_b #only used for train
        self.ex_true = ex_true
        #ex_true only matters for testing and has following string values:
        #"true" - LM has to pick this over others,
        #"false" - LM should not pick this over others
        #"other" - not known, not important, this is "other" candidate
        #"err_false" - Wnli_utils failed to parse the sentence. Automatically false

class DataProcessor(object):
    """Processor for the Wiki data set."""

    def wnli_test(self,source):
        examples=[]
        for line in tqdm(list(open(source,'r'))[1:],desc="parsing WNLI"):
            tokens = line.strip().split('\t')
            guid = tokens[0]
            premise = tokens[1]
            hypothesis = tokens[2]
            premise,candidates = wnli_utils.transform_wnli(premise,hypothesis)
            if premise==None:
                examples.append(InputExample(guid,"","",None,ex_true="err_false"))
                continue
            candidate_a = candidates[0]
            candidates_b = candidates[1:]
            examples.append(InputExample(guid,premise,candidate_a,None,ex_true="true"))#we don't know if it's true, but as long as it's not "other" the code will process it correctly
            for cand in candidates_b:
                examples.append(InputExample(guid,premise,cand,None,ex_true="other"))
        return examples

    def read_wscr_format_train(self,source):
        examples = []
        lines = list(open(source,'r'))
        for id_x,(sent,pronoun,candidates,candidate_a,_) in enumerate(zip(lines[0::5],lines[1::5],lines[2::5],lines[3::5],lines[4::5])):
            guid = id_x
            sent = sent.strip()
            text_a = sent.replace(' '+pronoun.strip()+' '," _ ",1)
            cnd = candidates.split(",")
            cnd = (cnd[0].strip().lstrip(),cnd[1].strip().lstrip())
            candidate_a = candidate_a.strip().lstrip()
            if candidate_a.casefold()==cnd[0].casefold():
                candidate_b = cnd[1]
            else:
                candidate_b=cnd[0]
            examples.append(InputExample(guid, text_a, candidate_a, candidate_b, ex_true="true"))
        return examples

    def read_wscr_format_test(self,source):
        examples = []
        lines = list(open(source,'r'))
        for id_x,(sent,pronoun,candidates,candidate_a,_) in enumerate(zip(lines[0::5],lines[1::5],lines[2::5],lines[3::5],lines[4::5])):
            guid = id_x
            sent = sent.strip()
            text_a = sent.replace(' '+pronoun.strip()+' '," _ ",1)
            candidate_a = candidate_a.strip().lstrip()
            cnd = candidates.strip().split(",")
            cnd = (candidate.strip().lstrip() for candidate in cnd if candidate.strip().lstrip().casefold()!= candidate_a.casefold())
            examples.append(InputExample(guid, text_a, candidate_a, None, ex_true="true"))
            for candidate in cnd:
                examples.append(InputExample(guid, text_a, candidate, None,ex_true="other"))
        return examples


    def get_examples(self, data_dir, set_name):#works for differently for train!
        """See base class."""
        file_names = {
                "maskedwiki":"MaskedWiki_downsampled.txt",
                "wscr-train": "train.c.txt",
                "wscr-test": "test.c.txt",
                "wsc": "wsc273.txt",
                "wnli":"wnli-test.tsv"
                }
        source = os.path.join(data_dir,file_names[set_name])
        if set_name in ["wscr-train","maskedwiki"]:
            return self.read_wscr_format_train(source)
        elif set_name in ["wscr-test","wsc"]:
            return self.read_wscr_format_test(source)
        elif set_name=="wnli":
            return self.wnli_test(source)
        else:
            print("Unknown set_name: ",set_name)
