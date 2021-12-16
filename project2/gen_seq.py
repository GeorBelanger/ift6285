from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GPT2LMHeadModel
import time
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
# input_context = "The dog" # model = GPT2LMHeadModel.from_pretrained("distilgpt2")

test_en_filename="/mnt/c/Users/gebelang/Documents/Sources/udem/ift6285/project2/projet2-dev/news.test"
# test_en_filename="project2/project2-dev/hans.test"
num_lines_processed=10 # 3003

start_time = time.perf_counter()

with open(test_en_filename, 'r') as f:
    lines = f.readlines()

lines = [line.strip() for line in lines[:num_lines_processed]] #lines[:num_lines_processed]

test_ref_en_filename="/mnt/c/Users/gebelang/Documents/Sources/udem/ift6285/project2/projet2-dev/news.ref"
# test_en_filename="project2/project2-dev/hans.test"

with open(test_ref_en_filename, 'r') as f:
    lines_ref = f.readlines()

lines_ref = [line.strip() for line in lines_ref[:num_lines_processed]] 

lines_pred = []
for i, line_to_order in enumerate(lines):
    line_length = len(line_to_order)
    words = line_to_order.split()
    number_of_words = len(words)
    words_pred = []
    curr_string = " "
    for j in range(number_of_words):
        line_to_order_ids = tokenizer(line_to_order).input_ids
        # input_context = "Why everything"
        input_context = curr_string
        # encode input context
        input_ids = tokenizer(input_context, return_tensors="pt").input_ids
        # generate 3 candidates using sampling
        # outputs = model.generate(input_ids=input_ids, max_length=1, num_return_sequences=100, do_sample=True)
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :].squeeze()

        # print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=False))
        line_to_order_logits = np.array([next_token_logits[lto_id].item() for lto_id in line_to_order_ids])
        word_chosen_id = np.argmax(line_to_order_logits)
        word_chosen = words[word_chosen_id]
        words_pred.append(word_chosen)
        if len(curr_string.strip())<1:
            curr_string = word_chosen
        else:
            curr_string = curr_string.strip()+ ' '+word_chosen

        del words[word_chosen_id]

        # batches = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        # len_input_context=len(input_context)
        # curr_string = [b.strip() for b in batches]
        # print("curr_string:", curr_string)
        # print("Generated:", [b[line_length:].strip() for b in batches])
        # print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

