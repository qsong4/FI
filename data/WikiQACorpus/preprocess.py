def readJsonFile(inputfile, outputfile):
    with open(inputfile, 'r') as fr, open(outputfile,"w") as fw:
        for i in fr:
            content = i.strip()
            res = content.split("\t")
            Question = res[1].lower()
            sentence = res[5].lower()
            gold_label = res[-1]
            if gold_label not in ['0', '1']:
                continue
            fw.write(Question + '\t' + sentence + '\t' + gold_label + '\n')
    print("**********Done**********")

readJsonFile("./WikiQA-dev.tsv", "./WikiQA-dev-sample.tsv")