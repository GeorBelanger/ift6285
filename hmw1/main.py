import pandas as pd
import nltk
from collections import Counter
import glob
import numpy as np
#import matplotlib.pyplot as plt
import time
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from cleantext import clean
num_tokens = []
num_types = []
global_counter = Counter()
time_list = []

filenames = sorted(glob.glob("/u/demorali/corpora/1g-word-lm-benchmark-r13output/training-monolingual.tokenized.shuffled/*"))
filenames = filenames[:10]
start_time = time.perf_counter()
for i, filename in enumerate(filenames):
    print(filename,i)

    with open(filename, 'r') as f:
        lines = f.readlines()
    doc_tokens = 0
    
    for line in lines:
        prepro = True
        if prepro:
            # https://towardsdatascience.com/nlp-preprocessing-with-nltk-3c04ee00edc0
            #line = line.lower() #lowercase
            #text_p = "".join([char for char in line if char not in string.punctuation]) #remove punctuation
            #words = word_tokenize(text_p)
            #filtered_words = [word for word in words if word not in stop_words]


            line=clean(line,
                fix_unicode=False,               # fix various unicode errors
                to_ascii=False,                  # transliterate to closest ASCII representation
                lower=True,                     # lowercase text
                no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
                no_urls=False,                  # replace all URLs with a special token
                no_emails=False,                # replace all email addresses with a special token
                no_phone_numbers=False,         # replace all phone numbers with a special token
                no_numbers=False,               # replace all numbers with a special token
                no_digits=False,                # replace all digits with a special token
                no_currency_symbols=False,      # replace all currency symbols with a special token
                no_punct=False,                 # remove punctuations
                replace_with_punct="",          # instead of removing punctuations you may replace them
                replace_with_url="<URL>",
                replace_with_email="<EMAIL>",
                replace_with_phone_number="<PHONE>",
                replace_with_number="<NUMBER>",
                replace_with_digit="0",
                replace_with_currency_symbol="<CUR>",
                lang="en"                       # set to 'de' for German special handling
            )
            
        line_counter = Counter(line.split())
        global_counter.update(line_counter)
    num_tokens.append(sum(global_counter.values()))
    num_types.append(len(global_counter))
    ctime = time.perf_counter()
    time_list.append(ctime-start_time)
print("The total number of tokens is: ", num_tokens[-1])
print("The total number of types is: ", num_types[-1])
print('I love DIRO')
print(num_tokens)
print(num_types)
print(time_list)

print('100 most frequent words', global_counter.most_common(100))

print('1000 most frequent words after preprocessing', global_counter.most_common(1000))

print('1000 less frequent words after preprocessing', global_counter.most_common()[:-1001:-1])
#plt.plot(num_tokens)
#plt.ylabel('counts')
#plt.show()
with open('num_tokens.txt', 'w') as nt:
    for item in num_tokens:
        nt.write("%s\n" % item)

with open('num_types.txt', 'w') as nty:
    for item in num_types:
        nty.write("%s\n" % item)

with open('time_list.txt', 'w') as tl:
    for item in time_list:
        tl.write("%s\n" % item)
