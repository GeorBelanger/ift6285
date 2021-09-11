import pandas as pd
#import argparse
import nltk
from collections import Counter

if __name__=="__main__":
    #import pdb; pdb.set_trace()
    #parser = argparse.ArgumentParser(description='Count words')
    #parser.add_argument('filename', metavar="fn", type=str, help='file to analyze')
    lines = []
    filename = "/u/demorali/corpora/1g-word-lm-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100"
    #df = pd.read_csv(filename, sep="\t")
    #for line in pd.read_csv(filename, encoding='utf-8', header=None, chunksize=1):
    #    lines.append(line.iloc[0,0])
    with open(filename, 'r') as f:
        lines = f.readlines()
    num_tokens = []
    num_types = []
    types = set()
    lines=lines[:10]
    #Tokenize with nltk
    use_nltk = False
    
    for line in lines:
        #print(line)
        if use_nltk:
            tokens = nltk.word_tokenize(line)
            #print(sorted(tokens))
            num_tokens.append(len(tokens))
        else:
            tokens = Counter(line.split())
            #print(tokens)
            #import pdb; pdb.set_trace()
            #num_types.append(len(tokens))
            num_tokens.append(sum(tokens.values()))
            for el in tokens.keys():
                if el not in types:
                    types.add(el)
            num_types.append(len(types))

    print("The total number of tokens is: ", sum(num_tokens))
    print("The total number of types is: ", sum(num_types))
    print('I love DIRO')

