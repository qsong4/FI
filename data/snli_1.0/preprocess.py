import os,sys
import json

def readJsonFile(inputfile, outputfile):
    with open(inputfile, 'r') as fr, open(outputfile,"w") as fw:
        for i in fr:
            content = i.strip()
            res = json.loads(content)
            sentence1 = res["sentence1"].lower()
            sentence2 = res["sentence2"].lower()
            gold_label = res["gold_label"]
            if gold_label not in ['neutral', 'entailment', 'contradiction']:
                continue
            fw.write(sentence1 + '\t' + sentence2 + '\t' + gold_label + '\n')
    print("**********Done**********")

readJsonFile("./snli_1.0_train.jsonl", "./snli_1.0_train.tsv")