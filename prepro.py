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
    with open(vocab_file, "w") as fw:
        for i in vocab_set:
            fw.write(i+'\n')
    print("labels: ", label_set)



if __name__ == '__main__':
    prepro(hp)
    logging.info("Done")