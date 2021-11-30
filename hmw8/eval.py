from sacrebleu.metrics import BLEU
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

translations_file = "./hmw8/translations_model_t5-small_num_lines_1000.txt" 
test_fr_filename ="/Users/belanger/.sacrebleu/wmt14/en-fr.fr"

with open(translations_file, 'r', encoding='utf8') as f:
    hyps = f.readlines()
    hyps = [str(sent.rstrip().rstrip('</s>').lstrip('<pad> ')) for sent in hyps]
    # hyps = hyps[:num_of_sentences]
    # print(hyps[0])

with open(test_fr_filename, 'r', encoding='utf8') as f2:
    refs = f2.readlines()
    refs = [[str(sent.rstrip()) for sent in refs]]
    # refs = refs[:num_of_sentences]
    print(refs[0])


print(type(hyps))
print(type(refs))

bleu = BLEU()

# num_of_sentences = 300
# for num_of_sentences in range(100, 1100, 100):
#     refs1 = []
#     # refs1[0]=refs[0][:num_of_sentences]
#     refs1.append(refs[0][:num_of_sentences])
#     hyps1 = []
#     hyps1=hyps[:num_of_sentences]
#     print('num of sentences ', num_of_sentences, bleu.corpus_score(hyps1,refs1))
bleu_scores = []
for sent_num in range(1000):
    refs1 = []
    refs1.append([refs[0][sent_num]])
    hyps1 = []
    hyps1.append(hyps[sent_num])
    print('sent_num ', sent_num, bleu.corpus_score(hyps1,refs1))
    bleu_scores.append(bleu.corpus_score(hyps1,refs1).score)

bleu_scores=np.array(bleu_scores)
# plt.hist(bleu_scores, color='blue', edgecolor='black', bins=20)
# plt.xlabel('BLEU score')
# plt.ylabel('number of sentences')
plt.boxplot(bleu_scores)
plt.show()

# sns.distplot(bleu_scores, hist=True, kde=True, bins=20, color='darkBlue', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth':4})
# plt.hist(bleu_scores)
# sns.kdeplot(bleu_scores)
print('we did it')

# result = bleu.corpus_score(hyps,refs)
# print('num of sentences ', num_of_sentences, result)



