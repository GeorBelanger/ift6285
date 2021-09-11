import pandas as pd
#import argparse
import nltk
from collections import Counter
import glob
import numpy as np
import matplotlib.pyplot as plt

num_tokens = []
num_types = []
types = set()

filenames = glob.glob("/u/demorali/corpora/1g-word-lm-benchmark-r13output/training-monolingual.tokenized.shuffled/*")
#filenames = filenames[:2]
for i, filename in enumerate(filenames):
    print(i)

#if __name__=="__main__":
    #import pdb; pdb.set_trace()
    #parser = argparse.ArgumentParser(description='Count words')
    #parser.add_argument('filename', metavar="fn", type=str, help='file to analyze')
    #filenames = glob
    #filename = "/u/demorali/corpora/1g-word-lm-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100"
    #df = pd.read_csv(filename, sep="\t")
    #for line in pd.read_csv(filename, encoding='utf-8', header=None, chunksize=1):
    #    lines.append(line.iloc[0,0])
    with open(filename, 'r') as f:
        lines = f.readlines()
    doc_tokens = 0
    
    for line in lines:
        tokens = Counter(line.split())
            #print(tokens)
            #import pdb; pdb.set_trace()
            #num_types.append(len(tokens))
            #num_tokens.append(sum(tokens.values()))
        doc_tokens += sum(tokens.values())
        for el in tokens.keys():
            if el not in types:
                types.add(el)
    num_tokens.append(doc_tokens)
    num_types.append(len(types))

print("The total number of tokens is: ", sum(num_tokens))
print("The total number of types is: ", sum(num_types))
print('I love DIRO')
print(num_tokens)
print(num_types)
#plt.plot(num_tokens)
#plt.ylabel('counts')
#plt.show()
with open('num_tokens.txt', 'w') as nt:
    for item in num_tokens:
        nt.write("%s\n" % item)

with open('num_types.txt', 'w') as nty:
    for item in num_types:
        nty.write("%s\n" % item)
