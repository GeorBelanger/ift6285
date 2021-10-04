"""Based on:
Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html

Copyright (c) 2007-2016 Peter Norvig
MIT license: www.opensource.org/licenses/mit-license.php
"""

################ Spelling Corrector 

import re
from collections import Counter
import argparse


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

def P(word, N=sum(WORDS.values())): 
# def P(word, N=sum(WORDS.freq.values)): 
    "Probability of `word`."
    # try:
    #     prob = WORDS[word] / N 
    # except:
    #     prob = 0
    # return prob
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    # import pdb; pdb.set_trace()
    return max(candidates(word), key=P)

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

def spelltest(tests, verbose=False):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    import time
    start = time.perf_counter()
    good, unknown = 0, 0
    n = len(tests)
    # for right, wrong in tests:
    for wrong, right in tests:
        w = correction(wrong)
        good += (w == right)
        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w, WORDS[w], right, WORDS[right]))
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
    # import pdb; pdb.set_trace()
    for line in words_file:
        print(line)
        try:
            value, key = line.split()
            value = int(value)
            WORDS[key] = value
        except Exception as e:
            print(line, e)
    print(WORDS)
    # import pdb; pdb.set_trace()
    # spelltest(Testset2(open('ift6285/hmw3/devoir3-train.txt')))


