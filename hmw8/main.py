from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import time

model_name = "t5-small"

model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# config = datasets.wmt.WmtConfig(
#     version="0.0.1",
#     language_pair=("fr", "de"),
#     subsets={
#         datasets.Split.TRAIN: ["commoncrawl_frde"],
#         datasets.Split.VALIDATION: ["euelections_dev2019"],
#     },
# )
# builder = datasets.builder("wmt_translate", config=config)
# dataset = load_dataset("wmt14", 'fr-en')

test_fr_filename ="/Users/belanger/.sacrebleu/wmt14/en-fr.fr"
test_en_filename="/Users/belanger/.sacrebleu/wmt14/en-fr.en"
num_lines_processed=10 # 3003

start_time = time.perf_counter()

with open(test_en_filename, 'r') as f:
    lines = f.readlines()

lines = lines[:num_lines_processed]

translated_sentences = []
for id_line, line in enumerate(lines):
    print('id_line: ', id_line)
    print('line: ', line)
    line_to_process = "translate English to French: "+line
    inputs = tokenizer(line_to_process, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True)

    translated_sentences.append(tokenizer.decode(outputs[0]))
    # print(tokenizer.decode(outputs[0]))

end_time = time.perf_counter()
print(f'time_taken {end_time-start_time}')

output_file_name = 'model_'+model_name+'_num_lines_'+str(num_lines_processed)
with open(f'translations_{output_file_name}.txt', 'w') as nty:
    for item in translated_sentences:
        nty.write("%s\n" % item)

# inputs = tokenizer("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="pt")
# outputs = model.generate(inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True)

# print(tokenizer.decode(outputs[0]))