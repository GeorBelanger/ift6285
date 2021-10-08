"""Based on:
Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html

Copyright (c) 2007-2016 Peter Norvig
MIT license: www.opensource.org/licenses/mit-license.php
"""

################ Spelling Corrector 

import re
from collections import Counter
import argparse
import numpy as np
import textdistance


WORDS = {}
# words_file = open('ift6285/hmw3/voc-1bwc.txt')
# # import pdb; pdb.set_trace()
# for line in words_file:
#     print(line)
#     try:
#         value, key = line.split()
#         value = int(value)
#         WORDS[key] = value
#     except Exception as e:
#         print(line, e)
# print(WORDS)

# def P(word, N=sum(WORDS.values())): 
def P(word, N): 
# def P(word, N=sum(WORDS.freq.values)): 
    "Probability of `word`."
    try:
        prob = WORDS[word] / N 
    except:
        prob = 0
        print(f'the word {word} or its edit1 and edit 2 does not appear in the voc')
    return prob
    # return WORDS[word] / N

def correction(word, N, dist='unigram'): 
    "Most probable spelling correction for word."
    candidate_list = [cand for cand in candidates(word)]
    cand_prob = [P(cand, N) for cand in candidates(word)]
    # import pdb; pdb.set_trace()
    # return max(candidates(word), key=P)
    max_idx = np.argmax(cand_prob)
    likely_corrections = [x for _, x in sorted(zip(cand_prob,candidate_list), reverse=True)] 
    print(f'the most likely corrections of {word} are ', likely_corrections)
    # likely corrections is the set of words sorted by unigram score
    # TODO: define function of distances
    if dist=='hamming':
        hamming_dist = [textdistance.hamming(corr, word) for corr in likely_corrections]
        likely_hamming_corrections = [x for _, x in sorted(zip(hamming_dist,likely_corrections), reverse=False)]
        # max_idx = np.argmmin(hamming_dist)
        return likely_hamming_corrections[0]
    # print('hamming distance', hamming_dist)
    elif dist=='lev':
        lev_dist = [textdistance.levenshtein(corr, word) for corr in likely_corrections]
        likely_lev_corrections = [x for _, x in sorted(zip(lev_dist,likely_corrections), reverse=False)]
        return likely_lev_corrections[0]
    elif dist=='jw':
        jw_dist = [textdistance.jaro_winkler(corr, word) for corr in likely_corrections]
        likely_jw_corrections = [x for _, x in sorted(zip(jw_dist,likely_corrections), reverse=False)]
        return likely_jw_corrections[0]
    else:
        return candidate_list[max_idx]
    # print('levenstein distance', lev_dist)
    # TODO: evaluate and rank wrt the other distances and grab argmax

    #evaluate with differe
    # return max(candidates(word), key=P)
    # return candidate_list[max_idx]
    # return likely_corrections

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

################ Test Code 

def unit_tests():
    assert correction('speling') == 'spelling'              # insert
    assert correction('korrectud') == 'corrected'           # replace 2
    assert correction('bycycle') == 'bicycle'               # replace
    assert correction('inconvient') == 'inconvenient'       # insert 2
    assert correction('arrainged') == 'arranged'            # delete
    assert correction('peotry') =='poetry'                  # transpose
    assert correction('peotryy') =='poetry'                 # transpose + delete
    assert correction('word') == 'word'                     # known
    assert correction('quintessential') == 'quintessential' # unknown
    assert 0.07 < P('the') < 0.08
    return 'unit_tests pass'

def spelltest(tests, N, verbose=True, dist='unigram'):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    import time
    start = time.perf_counter()
    good, unknown = 0, 0
    n = len(tests)
    # for right, wrong in tests:
    for wrong, right in tests:
        w = correction(wrong, N, dist)
        good += (w == right)
        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                try:
                    print('correction({}) => {} ({}); expected {} ({})'
                        .format(wrong, w, WORDS[w], right, WORDS[right]))
                except:
                    print('correction({}) => {} (N/A); expected {} (N/A)'
                        .format(wrong, w, right))
    dt = time.perf_counter() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
          .format(good / n, n, unknown / n, n / dt))

def Testset2(lines):
    "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
    return [(right, wrong)
            for (right, wrongs) in (line.split('\t') for line in lines)
            for wrong in wrongs.split()]

# def Testset(lines):
#     "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
#     return [(right, wrong)
#             for (right, wrongs) in (line.split(':') for line in lines)
#             for wrong in wrongs.split()]

if __name__ == '__main__':
    # print(unit_tests())
    # spelltest(Testset(open('spell-testset1.txt')))
    # spelltest(Testset(open('spell-testset2.txt')))
    # spelltest(Testset2(open('ift6285/hmw3/devoir3-train.txt')))
    print('start')
    parser = argparse.ArgumentParser()
    parser.add_argument('--lexicon', help='lexico of words among which to search for the correction', type=str, default='ift6285/hmw3/voc-1bwc.txt')
    args = parser.parse_args()
    print(args.lexicon)


    WORDS = {}
    # words_file = open('ift6285/hmw3/voc-1bwc.txt')
    words_file = open(args.lexicon)
    for line in words_file:
        print(line)
        try:
            value, key = line.split()
            value = int(value)
            WORDS[key] = value
        except Exception as e:
            print(line, e)
    print(WORDS)
    N=sum(WORDS.values())
    
    # spelltest(Testset2(open('ift6285/hmw3/devoir3-train.txt')), N)
    # spelltest(Testset2(open('ift6285/hmw3/devoir3-train-small.txt')), N)
    # spelltest(Testset2(open('ift6285/hmw3/devoir3-train-medium.txt')), N)
    spelltest(Testset2(open('ift6285/hmw3/devoir3-train-medium.txt')), N, dist='lev')



