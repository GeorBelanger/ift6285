from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import time
import pandas as pd

# from simpletransformers.t5 import T5Model, T5Args
# import logging

# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)


# train_data = [
#     ["binary classification", "Anakin was Luke's father" , "1"],
#     ["binary classification", "Luke was a Sith Lord" , "0"],
#     ["generate question", "Star Wars is an American epic space-opera media franchise created by George Lucas, which began with the eponymous 1977 film and quickly became a worldwide pop-culture phenomenon", "Who created the Star Wars franchise?"],
#     ["generate question", "Anakin was Luke's father" , "Who was Luke's father?"],
# ]
# train_df = pd.DataFrame(train_data)
# train_df.columns = ["prefix", "input_text", "target_text"]

# eval_data = [
#     ["binary classification", "Leia was Luke's sister" , "1"],
#     ["binary classification", "Han was a Sith Lord" , "0"],
#     ["generate question", "In 2020, the Star Wars franchise's total value was estimated at US$70 billion, and it is currently the fifth-highest-grossing media franchise of all time.", "What is the total value of the Star Wars franchise?"],
#     ["generate question", "Leia was Luke's sister" , "Who was Luke's sister?"],
# ]
# eval_df = pd.DataFrame(eval_data)
# eval_df.columns = ["prefix", "input_text", "target_text"]

# model_args = T5Args()
# model_args.num_train_epochs = 200
# model_args.no_save = True
# model_args.evaluate_generated_text = True
# model_args.evaluate_during_training = True
# model_args.evaluate_during_training_verbose = True

# model_name = "t5-small"
# model = T5Model("t5", model_name, args=model_args)


# def count_matches(labels, preds):
#     print(labels)
#     print(preds)
#     return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])


# model.train_model(train_df, eval_data=eval_df, matches=count_matches)

# print(model.eval_model(eval_df, matches=count_matches))


model_name = "t5-small"
model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# test_en_filename="/Users/belanger/.sacrebleu/wmt14/en-fr.en"
test_en_filename="/Users/belanger/ift6285/project2/projet2-dev/news.test"
# test_en_filename="project2/project2-dev/hans.test"
num_lines_processed=10 # 3003

start_time = time.perf_counter()

with open(test_en_filename, 'r') as f:
    lines = f.readlines()

lines = lines[:num_lines_processed]

translated_sentences = []
for id_line, line in enumerate(lines):
    print('id_line: ', id_line)
    print('line: ', line)
    # line_to_process = "translate English to French: "+line
    # line= 'services from ebbsfleet will cost £ 3.10 more than a single from the nearest station on the existing network , at gravesend .'
    # line = 'existing a ebbsfleet station than more 3.10 , the on cost nearest network from £ gravesend the will single at from services .'
    # line = 'the picked of fflics . welsh festival history remember been the films stars organisers cinema welsh the , have to define at showing and say'
    line_to_process = "deshuffle: "+line
    inputs = tokenizer(line_to_process, return_tensors="pt")
    max_length=len(line.split(' ')); print(max_length)
    outputs = model.generate(inputs["input_ids"], max_length=40, num_beams=10, early_stopping=True) # max_length=40
    # print(outputs)
    translated_sentences.append(tokenizer.decode(outputs[0]))
    # print(tokenizer.decode(outputs[0]))
    print(tokenizer.decode(outputs[0]))

end_time = time.perf_counter()
print(f'time_taken {end_time-start_time}')

output_file_name = 'model_'+model_name+'_num_lines_'+str(num_lines_processed)
with open(f'deshuffle_{output_file_name}.txt', 'w') as nty:
    for item in translated_sentences:
        nty.write("%s\n" % item)

# inputs = tokenizer("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="pt")
# outputs = model.generate(inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True)

# print(tokenizer.decode(outputs[0]))
# import logging

# import pandas as pd
# from simpletransformers.t5 import T5Model, T5Args

# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)


# train_data = [
#     ["binary classification", "Anakin was Luke's father" , "1"],
#     ["binary classification", "Luke was a Sith Lord" , "0"],
#     ["generate question", "Star Wars is an American epic space-opera media franchise created by George Lucas, which began with the eponymous 1977 film and quickly became a worldwide pop-culture phenomenon", "Who created the Star Wars franchise?"],
#     ["generate question", "Anakin was Luke's father" , "Who was Luke's father?"],
# ]
# train_df = pd.DataFrame(train_data)
# train_df.columns = ["prefix", "input_text", "target_text"]

# eval_data = [
#     ["binary classification", "Leia was Luke's sister" , "1"],
#     ["binary classification", "Han was a Sith Lord" , "0"],
#     ["generate question", "In 2020, the Star Wars franchise's total value was estimated at US$70 billion, and it is currently the fifth-highest-grossing media franchise of all time.", "What is the total value of the Star Wars franchise?"],
#     ["generate question", "Leia was Luke's sister" , "Who was Luke's sister?"],
# ]
# eval_df = pd.DataFrame(eval_data)
# eval_df.columns = ["prefix", "input_text", "target_text"]

# model_args = T5Args()
# model_args.num_train_epochs = 200
# model_args.no_save = True
# model_args.evaluate_generated_text = True
# model_args.evaluate_during_training = True
# model_args.evaluate_during_training_verbose = True

# model = T5Model("t5", "t5-base", args=model_args)


# def count_matches(labels, preds):
#     print(labels)
#     print(preds)
#     return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])


# model.train_model(train_df, eval_data=eval_df, matches=count_matches)

# print(model.eval_model(eval_df, matches=count_matches))
