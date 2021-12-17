# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GPT2LMHeadModel
import time
import numpy as np
import torch
import data

#hans #euro
test_en_filename="/mnt/c/Users/gebelang/Documents/Sources/udem/ift6285/project2/projet2-dev/hans.test"
# test_en_filename="project2/project2-dev/hans.test"
# data1 = "/mnt/c/Users/gebelang/Documents/Sources/udem/ift6285/small_corpus"
data1 = "/mnt/c/Users/gebelang/Documents/Sources/udem/ift6285"
corpus = data.Corpus(data1)
corpus.dictionary.add_word('<UNK>') 
ntokens = len(corpus.dictionary)
vvocab = corpus.dictionary.word2idx.keys()
word2idx = corpus.dictionary.word2idx

def get_id(word):
    # line_to_order_ids = [corpus.dictionary.word2idx[w] for w in words if w in corpus.dictionary.word2idx.keys() else corpus.dictionary.word2idx['UNK']]
    if word in vvocab:
        word_id = word2idx[word]
    else:
        word_id = word2idx['<UNK>']
    return word_id

num_lines_processed=1000 # 3003

start_time = time.perf_counter()

with open(test_en_filename, 'r') as f:
    lines = f.readlines()

lines = [line.strip() for line in lines[:num_lines_processed]] #lines[:num_lines_processed]

test_ref_en_filename="/mnt/c/Users/gebelang/Documents/Sources/udem/ift6285/project2/projet2-dev/hans.ref"
# test_en_filename="project2/project2-dev/hans.test"

with open(test_ref_en_filename, 'r') as f:
    lines_ref = f.readlines()

lines_ref = [line.strip() for line in lines_ref[:num_lines_processed]] 

# line_to_order_logits = np.array([next_token_logits[lto_id].item() for lto_id in line_to_order_ids])
# np.array([get_logits(lto_id) for lto_id in line_to_order_ids])
def get_logits(lto_id, next_token_logits):
    if lto_id in next_token_logits:
        logit = next_token_logits[lto_id].item()
    else:
        logit = 0
    return logit


device = torch.device("cpu")
temperature = 1.0
if temperature < 1e-3:
    print("--temperature has to be greater or equal 1e-3")
    raise

checkpoint = './word_language_model/model.pt'
with open(checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
ntokens=100 # might have to change this, in the other script is the #vocab words
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
# input = torch.Tensor([[84]], dtype=torch.long) #.to(device)
dot_id = corpus.dictionary.word2idx['.']
input[0][0] = dot_id

lines_pred = []
for i, line_to_order in enumerate(lines):

    words = line_to_order.split()
    number_of_words = len(words)
    words_pred = []
    word_chosen_dict_id_list = []
    #curr_string = " "
    for j in range(number_of_words):
        # line_to_order_ids = [corpus.dictionary.word2idx[w] for w in words if w in corpus.dictionary.word2idx.keys() else corpus.dictionary.word2idx['UNK']]
        line_to_order_ids = [get_id(w) for w in words]

        words_to_generate = 1
        with torch.no_grad():  # no tracking history
            for i in range(words_to_generate):
                try:
                    output, hidden = model(input, hidden)
                except:
                    input[0][0] = dot_id
                    output, hidden = model(input, hidden)
        
                # outputs = model(input_ids)
                # next_token_logits = output[0][:, -1, :].squeeze()
                next_token_logits = output[0]
                line_to_order_logits = np.array([get_logits(lto_id, next_token_logits) for lto_id in line_to_order_ids])
                word_chosen_id = np.argmax(line_to_order_logits)

                word_chosen = words[word_chosen_id]
                words_pred.append(word_chosen)
                if word_chosen in corpus.dictionary.word2idx.keys():
                    
                    word_chosen_dict_id = corpus.dictionary.word2idx[word_chosen]
                    # dot_id = corpus.dictionary.word2idx['.']
                    # input[0][0] = dot_id
                    # input[0][0] = word_chosen_dict_id
                else:
                    word_chosen_dict_id = dot_id
                word_chosen_dict_id_list.append(word_chosen_dict_id)
                # input = torch.randint(ntokens, (1, len(word_chosen_dict_id_list)), dtype=torch.long).to(device)
                input.fill_(word_chosen_dict_id)
                # for ii, idxx in enumerate(word_chosen_dict_id_list):
                #     input[0][ii] = idxx
                del words[word_chosen_id]
                line_to_order = " ".join(words)

    lines_pred.append(words_pred)

print('lines pred', len(lines_pred))
print('lines ref', len(lines_ref))

lines_ref_list = [line_ref.split() for line_ref in lines_ref]

def eval_lists(pred_list, ref_list):
    list_of_perc = []
    for i in range(len(ref_list)):
        correct_words = 0
        for j in range(len(ref_list[i])):
            if ref_list[i][j] == pred_list[i][j]:
                correct_words+=1
        list_of_perc.append(correct_words/len(ref_list[i]))
    return list_of_perc
eval_result = np.array(eval_lists(lines_pred, lines_ref_list))
# print(eval_lists(lines_pred, lines_ref_list))
print(np.mean(eval_result))
print('done!')