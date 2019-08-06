import re
def prepro(train_file, dev_file, vocab_file):
    vocab_set = set()
    label_set = set()
    for f in (train_file, dev_file):
        with open(f, "r") as fr:
            for line in fr:
                content = line.strip().split()
                label = content[2]
                sent1 = content[0]
                sent2 = content[1]
                for i in sent1+sent2:
                    vocab_set.add(i)
                label_set.add(label)
    #0: <pad>, 1: <unk>, 2: <s>, 3: </s>
    with open(vocab_file, "w") as fw:
        fw.write("<pad>"+"\n")
        fw.write("<unk>"+"\n")
        for i in vocab_set:
            fw.write(i+'\n')
    print("labels: ", label_set)

def removePunc(inputStr):
    string = re.sub(r"\W+", "", inputStr)
    return string.strip()

def genvocabGlove(inputfile, vocab_file):
    vocab = []
    vocab.append('<pad>')
    vocab.append('<unk>')
    file = open(inputfile, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
    print('Loaded GloVe!')
    file.close()
    with open(vocab_file, "w") as fw:
        fw.write("<pad>"+"\n")
        fw.write("<unk>"+"\n")
        for i in vocab:
            fw.write(i+'\n')
    return vocab

def prepro_snli(train_file, dev_file, vocab_file, char_file):
    vocab_set = set()
    label_set = set()
    char_set = set()
    for f in (train_file, dev_file):
        with open(f, "r") as fr:
            for line in fr:
                content = line.strip().split("\t")
                label = content[2]
                sent1 = content[0]
                sent2 = content[1]
                sent = sent1 + ' ' + sent2
                for i in re.split(r"\W+", sent):
                    i = i.strip()
                    i = removePunc(i)
                    i = i.lower()
                    if i == "":
                        continue
                    vocab_set.add(i)
                    for char in i:
                        char_set.add(char.strip())
                label_set.add(label)
    #0: <pad>, 1: <unk>, 2: <s>, 3: </s>
    with open(vocab_file, "w") as fw:
        fw.write("<pad>"+"\n")
        fw.write("<unk>"+"\n")
        for i in vocab_set:
            fw.write(i+'\n')

    with open(char_file, "w") as fw:
        fw.write("<pad>"+"\n")
        fw.write("<unk>"+"\n")
        for i in char_set:
            i.strip()
            fw.write(i+'\n')

    print("labels: ", label_set)



if __name__ == '__main__':
    train_file = "./data/snli_train.tsv"
    dev_file = "./data/snli_dev.tsv"
    vocab_file = "./data/snli.vocab"
    char_file = "./data/snli.char.vocab"
    #prepro(train_file, dev_file, vocab_file)
    prepro_snli("./data/snli_train.tsv", "./data/snli_dev.tsv", vocab_file, char_file)
    #genvocabGlove("./data/vec/glove.840B.300d.txt", vocab_file)
