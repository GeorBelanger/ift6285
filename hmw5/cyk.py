# Based on https://www.cs.bgu.ac.il/~elhadad/nlp16/NLTK-PCFG.html and the code that prof Phillipe Langlais showed during IFT6285 NLP course

def get_missing_words(grammar, sent):
    import pdb; pdb.set_trace()
    return (None, None)

import time
import sys
import nltk
from nltk.corpus import treebank # install ntlk and data
from nltk import Nonterminal
from nltk import induce_pcfg
# from nltk.parse import ViterbiParse
from nltk.parse import ViterbiParser
from collections import Counter

args = {'verbosity': 4, 'top': 3, 'max': 15, 'min': 5}

# import pdb; pdb.set_trace()
train = treebank.fileids()[:190]
test = treebank.fileids()[190:]
## TODO: analyze gramatically problematic sentences from the Cola dev corpus (those marked with a star)
# /home/www-labs/felipe/public_html/IFT6285-Automne2021/CoLA/

productions = []
sent = 1

for item in train:
    for tree in treebank.parsed_sents(item):

        sent += 1
        #print(tree)
        #tree.collapse_unary(collapsePOS = False) # REmove branches A-B-C into A-B+C
        #tree.chomsky_normal_form(horzMarkov = 2) # Remove A -> (B,C,D) into A->B,C+D

        productions += tree.productions() # tree.productions() returns the list of CFG rules that "explain" the tree
        # if args.verbosity > 3:
        if args['verbosity']> 3:
            for i,p in enumerate(tree.productions()):
                print(f'sent: {sent} tree: {i} prod: {p}', file = sys.stderr)

        # if args.verbosity > 2:
        if args['verbosity'] > 2:
            print(f'sent: {sent} prod: {len(tree.productions())} total prod {len(productions)}', file = sys.stderr)

# filtrer la grammaire (sur la frequence des productions)
count = Counter(productions)
proto = set(map(lambda p: p[0], count.most_common(args['top'])))
prods = [p for p in productions if p in proto]

# if args.unk is None:
unk = None
if not unk is None:
    lexical_symbols = set(map(lambda p: p.lhs(), filter(lambda p: p.is_lexical()))) # tree.productions() will return a production with the parent node as LHS and the children as RHS.
    for lhs in lexical_symbols:
        # p=Production(lhs,[args.unk])
        p=Production(lhs,[unk])
        prods.append(p)

# train the grammar:
axiom = Nonterminal('S')
grammar = induce_pcfg(axiom, prods) # induce PCFG grammar from treebank data
print(f'#productions initial: {len(productions)} final: {len(grammar.productions())}')

# if args.verbosity > 4:
if args['verbosity'] > 4:
    print(grammar)

# now parse
parser= ViterbiParser(grammar)


####
acc=0
nbtest=0 # nb of test sentences
treated = 0 # nb of sentences parsed
unparsed = 0 # nb of sentences withouth parse

for item in test:
    for tree in treebank.parsed_sents(item):
        sent=tree.leaves()
        nbtest+=1

        # if args.max is None or len(sent) <= args.max:
        if args['max'] is None or len(sent) <= args['max']:
            if args['min'] is None or len(sent) >= args['min']:
                treated += 1

                print("===", file=sys.stderr)
                
                nb_unk, sent_unk = get_missing_words(grammar, sent) # https://stackoverflow.com/questions/35103191/nltk-viterbiparser-fails-in-parsing-words-that-are-not-in-the-pcfg-rule

                print(f'sent ({len(sent)})=>', " ".join(sent), file = sys.stderr)
                if nb_unk:
                    print(f'sent: {nbtest} missing words: {nb_unk}/{len(sent)}')

                start_time = time.perf_counter()
                parses = parser.parse_all(sent_unk)
                time_elapsed = time.perf_counter - start_time

                if len(parses) > 0:
                    best = parses[0]
                    draw_trees(best)
                    a = accuracy(tree.pos(), best.pos())

                    if args['verbosity'] > 1:
                        print(f'parse: {best}\nacc: {a}', file = sys.stderr)

            
