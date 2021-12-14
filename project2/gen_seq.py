from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
# tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# model = AutoModelForCausalLM.from_pretrained("distilgpt2")
# # do greedy decoding without providing a prompt
# outputs = model.generate(max_length=40)
# print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
# tokenizer = AutoTokenizer.from_pretrained("t5-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
# document = (
# "at least two people were killed in a suspected bomb attack on a passenger bus "
# "in the strife-torn southern philippines on monday , the military said."
# )
# # encode input context
# input_ids = tokenizer(document, return_tensors="pt").input_ids
# # generate 3 independent sequences using beam search decoding (5 beams)
# # with T5 encoder-decoder model conditioned on short news article.
# outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
# print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))


tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
# input_context = "The dog"


# input_context = "Why everything"
input_context = "Why everything"
# line= 'services from ebbsfleet will cost £ 3.10 more than a single from the nearest station on the existing network , at gravesend .'
# line = 'existing a ebbsfleet station than more 3.10 , the on cost nearest network from £ gravesend the will single at from services .'
line = 'the picked of fflics . welsh festival history remember been the films stars organisers cinema welsh the , have to define at showing and say'
len_input_context=len(input_context)
# input_context = "1: you I . love -> I love you . 2: you I . hate -> "
# encode input context
input_ids = tokenizer(input_context, return_tensors="pt").input_ids
# generate 3 candidates using sampling
outputs = model.generate(input_ids=input_ids, max_length=1, num_return_sequences=10, do_sample=True)
# print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=False))
batches = tokenizer.batch_decode(outputs, skip_special_tokens=False)
# len_input_context=len(input_context)
print("Generated:", [b[len_input_context:].strip() for b in batches])
# print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))


# tokenizer = AutoTokenizer.from_pretrained("ctrl")
# model = AutoModelForCausalLM.from_pretrained("ctrl")
# # "Legal" is one of the control codes for ctrl
# input_context = "Legal My neighbor is"
# # encode input context
# input_ids = tokenizer(input_context, return_tensors="pt").input_ids
# outputs = model.generate(input_ids=input_ids, max_length=20, repetition_penalty=1.2)
# print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
# tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
# model = AutoModelForCausalLM.from_pretrained("gpt2")
# input_context = "My cute dog"
# # get tokens of words that should not be generated
# bad_words_ids = tokenizer(["idiot", "stupid", "shut up"], add_prefix_space=True).input_ids
# # encode input context
# input_ids = tokenizer(input_context, return_tensors="pt").input_ids
# # generate sequences without allowing bad_words to be generated
# outputs = model.generate(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids)
# print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))