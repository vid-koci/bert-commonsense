import os
import csv
import nltk
from nltk.tag.stanford import StanfordPOSTagger

POS_tagger=None

def transform_wnli(premise,hypothesis):
    cased_premise=premise
    premise=[w.lower() for w in nltk.word_tokenize(premise)]

    #transform WNLI examples back into WSC format
    hypothesis = [w.lower() for w in nltk.word_tokenize(hypothesis)]
    best_target=["","","","","",""]#should get overwritten
    best_masked_s=[]
    for l in range(len(hypothesis)):
        for r in range(l+1,l+6):
            left_part = hypothesis[:l]
            right_part = hypothesis[r:]
            pattern = left_part + ["_"]+ right_part
            for s in range(len(premise)):
                ok=True
                if s+len(pattern)>len(premise):
                    break
                for a,b in zip(pattern,premise[s:s+len(pattern)]):
                    if a=="_":
                        continue
                    if a==b:
                        continue
                    if a in [',','.','?','!'] and b in [',','.','?','!']:#punctuation is ignored
                        continue
                    ok=False
                    break
                if ok and len(hypothesis[l:r])<=len(best_target):
                    best_target = hypothesis[l:r]
                    best_masked_s = premise[:s]+pattern+premise[s+len(pattern):]
    if len(best_masked_s)==0:#We failed
        return None,None
    #We extracted the masked sentence from the premise.
    global POS_tagger
    if POS_tagger is None:
        os.environ['STANFORD_MODELS'] = "stanford-postagger-2018-10-16/models"
        os.environ['CLASSPATH'] = "stanford-postagger-2018-10-16"
        POS_tagger = StanfordPOSTagger("stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger")
    tagged_premise = POS_tagger.tag(nltk.word_tokenize(cased_premise))
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
    found_sentence = " ".join(best_masked_s).replace(" n't","n't").replace(" 's","'s")#Sorry nltk
    return found_sentence,candidates


