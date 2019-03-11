import os
import csv
import nltk
from nltk.tag.stanford import StanfordPOSTagger

def transform_wnli(premise,hypothesis):
    cased_premise=premise
    premise=[w.lower() for w in nltk.word_tokenize(premise)]
    hypothesis = [w.lower() for w in nltk.word_tokenize(hypothesis)]
    best_target=["ggg","gg","g"]
    best_masked_s=[]
    for l in range(len(hypothesis)):
        for r in range(l+1,l+3):
            left_part = hypothesis[:l]
            right_part = hypothesis[r:]
            pattern = left_part + ["[MASK]"]+ right_part
            for s in range(len(premise)):
                ok=True
                if s+len(pattern)>len(premise):
                    break
                for a,b in zip(pattern,premise[s:s+len(pattern)]):
                    if a=="[MASK]":
                        continue
                    if a==b:
                        continue
                    ok=False
                    break
                if ok and len(hypothesis[l:r])<=len(best_target):
                    best_target = hypothesis[l:r]
                    best_masked_s = premise[:s]+pattern+premise[s+len(pattern):]
                    #print ("Found:\n{}\n{}".format(premise[:s]+pattern+premise[s+len(pattern):],hypothesis[l:r]))
    if len(best_masked_s)==0:
        return None,None
    os.environ['STANFORD_MODELS'] = "stanford-postagger-2018-10-16/models"
    os.environ['CLASSPATH'] = "stanford-postagger-2018-10-16"
    POS_tagger = StanfordPOSTagger("stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger")
    tagged_premise = POS_tagger.tag(premise)
    candidates = []
    current=[]
    for word,tag in tagged_premise:
        if tag in ["NN","NNS","NNP","NNPS"]:
            current.append(word)
        else:
            if current!=[]:
                candidates.append(" ".join(current).lower())
                current=[]
    if current!=[]:
        candidates.append(" ".join(current).lower())
    best_target=" ".join(best_target)
    candidates=[c for c in candidates if c.find(best_target)==-1 and best_target.find(c)==-1]
    candidates = [best_target]+candidates
    return " ".join(best_masked_s),candidates



#data=list(open("../data/GLUE/WNLI/dev.tsv"))[1:]
#
#for line in data[:10]:
#    l = line.split("\t")
#    print("Analyzing: {}\t{}".format(l[1],l[2]))
#    print(transform_wnli(l[1],l[2]))


