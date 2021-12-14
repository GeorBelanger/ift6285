from simpletransformers.t5 import T5Model, T5Args
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_data = [
    ["binary classification", "Anakin was Luke's father" , "1"],
    ["binary classification", "Luke was a Sith Lord" , "0"],
    ["generate question", "Star Wars is an American epic space-opera media franchise created by George Lucas, which began with the eponymous 1977 film and quickly became a worldwide pop-culture phenomenon", "Who created the Star Wars franchise?"],
    ["generate question", "Anakin was Luke's father" , "Who was Luke's father?"],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["prefix", "input_text", "target_text"]

eval_data = [
    ["binary classification", "Leia was Luke's sister" , "1"],
    ["binary classification", "Han was a Sith Lord" , "0"],
    ["generate question", "In 2020, the Star Wars franchise's total value was estimated at US$70 billion, and it is currently the fifth-highest-grossing media franchise of all time.", "What is the total value of the Star Wars franchise?"],
    ["generate question", "Leia was Luke's sister" , "Who was Luke's sister?"],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["prefix", "input_text", "target_text"]

model_args = T5Args()
model_args.num_train_epochs = 200
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model_name = "t5-small"
model = T5Model("t5", model_name, args=model_args)


def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])


model.train_model(train_df, eval_data=eval_df, matches=count_matches)

print(model.eval_model(eval_df, matches=count_matches))