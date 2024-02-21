import os
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
from utils import *
import wandb
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer, pipeline
from concept_dataset_process import load_dataset
from utils import DATASETS_INFO

def preprocess_function(examples):
    text = examples['text']
    return tokenizer(text, padding='max_length', truncation=True, max_length = 512)


# seed = 42
DATASET_NAME =  'yelp' #  'beer' # 'go_emotions' #
MODEL_TYPE = 'bert'
setting_No = 2
n_exp = 4
lr= 2e-5
NUM_EPOCHS = 15 # 20 #
dataset_types =  ['train', 'test']
label_shift = DATASETS_INFO[DATASET_NAME] ['label_shift_to_index']
dataset_dir = DATASETS_INFO[DATASET_NAME]['path']
num_labels = DATASETS_INFO[DATASET_NAME]['num_labels']
shortcut_type = 'style' # 'occurrence' # 'concept' #
shortcut_subtype = 'author' # 'single-word' # 'occurrence' # 'synonym'  # 'category' #
save_path =  f"../model/{DATASET_NAME}/{MODEL_TYPE}/{shortcut_type}/{shortcut_subtype}_{setting_No}_{n_exp}"
test_type =  'normal' #
model_name = 'bert-base-uncased' if test_type == 'normal' else save_path
repo_name = f'finetuning-{DATASET_NAME}-{MODEL_TYPE}-{shortcut_subtype}-{shortcut_subtype}'

wandb.init(
    project= f"Shortcuts-{shortcut_type}-{shortcut_subtype}",
    config={
        "learning_rate": lr,
        "architecture": "bert-base-uncased",
        "dataset":DATASET_NAME,
        "epochs": NUM_EPOCHS,
    },
    name = f"{DATASET_NAME}_{setting_No}_{n_exp}_{test_type}"
)

# Load the dataset
dataset_dict = DatasetDict()
for dataset_type in dataset_types:
    dataset_path = os.path.join(dataset_dir, shortcut_type, shortcut_subtype + str(setting_No))
    if dataset_type == 'test' and test_type == 'anti-shortcut':
        file_path = os.path.join(dataset_path, f'{dataset_type}_anti-shortcut.csv')
    else:
        file_path = os.path.join(dataset_path, f'{dataset_type}.csv')
    dataset = load_dataset(file_path)
    print(dataset[0])
    dataset_dict[dataset_type] = dataset.map(lambda example: {'text': example['text'], 'label': int(example['label'] - label_shift)})
# print(dataset_dict)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Prepare the dataset
tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)
tokenized_train = tokenized_datasets["train"].shuffle()#(seed=seed)#.select(range(1000))
tokenized_test = tokenized_datasets["test"]
# print(tokenized_datasets['train'][0])
# print(tokenized_train[0])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=lr,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    save_strategy="epoch",
    # push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if test_type == 'normal':
    trainer.train()
    # Save the model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

trainer.evaluate()

# Save the prediction
result = trainer.predict(tokenized_test)
predictions = np.argmax(result.predictions, axis=-1)
data = pd.read_csv(file_path)
data[f'pred_{n_exp}'] = predictions + label_shift
data.to_csv(file_path, index=False)

# trainer.push_to_hub()
