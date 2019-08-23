
def compute_f1(ans_stats, gap_output=None):
    other_by_guid={}
    target_by_guid={}
    gold_by_guid={}
    for guid,ex_true,loss in ans_stats:
        if ex_true=="other":
            if guid in other_by_guid:
                other_by_guid[guid]=min(other_by_guid[guid],loss)
            else:
                other_by_guid[guid]=loss
        else:
            target_by_guid[guid]=loss
            gold_by_guid[guid]=ex_true
    tp,fp,tn,fn = 0,0,0,0
    output=[]
    for guid,gold in gold_by_guid.items():
        if gold=="err_true":
            fn+=1
            output.append((guid,"FALSE"))
        elif gold=="err_false":
            tn+=1
            output.append((guid,"FALSE"))
        elif gold=="true":
            if (guid in other_by_guid) and other_by_guid[guid] < target_by_guid[guid]:
                fn+=1
                output.append((guid,"FALSE"))
            else:
                tp+=1
                output.append((guid,"TRUE"))
        else:
            if (guid in other_by_guid) and other_by_guid[guid] < target_by_guid[guid]:
                tn+=1
                output.append((guid,"FALSE"))
            else:
                fp+=1
                output.append((guid,"TRUE"))
    if not gap_output is None:
        out_file = open(gap_output,'w')
        output.sort()
        for out in output:
            if out[0][-1]=='A':
                out_file.write("{}\t{}".format(out[0][:-1],out[1]))
            else:
                out_file.write("\t{}\n".format(out[1]))
        out_file.close()
    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    return 2*(precision*recall)/(precision+recall)

def compute_accuracy(ans_stats, wnli_output=None):
    other_by_guid={}
    target_by_guid={}
    gold_by_guid={}
    for guid,ex_true,loss in ans_stats:
        if ex_true=="other":
            if guid in other_by_guid:
                other_by_guid[guid]=min(other_by_guid[guid],loss)
            else:
                other_by_guid[guid]=loss
        else:
            target_by_guid[guid]=loss
            gold_by_guid[guid]=ex_true
    n_correct=0
    n_overall=0
    output=[]
    for guid,gold in gold_by_guid.items():
        if gold=="err_true" or gold=="err_false":
            n_overall+=1
            output.append((guid,"0"))
        else:
            if (guid in other_by_guid) and other_by_guid[guid] < target_by_guid[guid]:
                n_overall+=1
                output.append((guid,"0"))
            else:
                n_correct+=1
                n_overall+=1
                output.append((guid,"1"))
    if not wnli_output is None:
        out_file = open(wnli_output,'w')
        output.sort(key=lambda x:int(x[0]))
        for out in output:
            out_file.write("{}\t{}\n".format(out[0],out[1]))
        out_file.close()
    return float(n_correct)/n_overall

def scorer(ans_stats, test_set, output_file=None):
    if test_set=="gap-dev":
        return compute_f1(ans_stats)
    elif test_set=="gap-test":
        return compute_f1(ans_stats, gap_output=output_file)
    elif test_set in ["dpr-test","dpr-dev-small","wsc","wscr-test","pdp","winogender","winobias-pro1","winobias-pro2","winobias-anti1","winobias-anti2","winobias-dev","wikicrem-dev"]:
        return compute_accuracy(ans_stats)
    elif test_set=="wnli":
        return compute_accuracy(ans_stats,wnli_output=output_file)

