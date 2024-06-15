import os
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
from utils import *
import wandb
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer, pipeline
# from concept_dataset_process import load_dataset
from utils import DATASETS_INFO
from datasets import load_metric


def preprocess_function(examples):
    text = examples['text']
    return tokenizer(text, padding='max_length', truncation=True, max_length=512)


# seed = 42
# DATASET_NAME =  'go_emotions' # 'yelp' #  'beer' #
MODEL_TYPE = 'bert'
setting_No = 1
n_exp = 0
NUM_EPOCHS = 15  # 20 #
dataset_types = ['train', 'dev', 'test']
label_shift = DATASETS_INFO[DATASET_NAME]['label_shift_to_index']
dataset_dir = DATASETS_INFO[DATASET_NAME]['path']
num_labels = DATASETS_INFO[DATASET_NAME]['num_labels']
shortcut_type = SET_SHORTCUT_TYPE  # 'occurrence' # 'style' #  'concept' #
shortcut_subtype = SET_SHORTCUT_SUBTYPE1  # 'single-word' # 'synonym'  #  'author' #  'occurrence' #  'category' #
save_path = f"../model/{DATASET_NAME}/{MODEL_TYPE}/{shortcut_type}/{shortcut_subtype}_{setting_No}_{n_exp}"
test_type = 'normal'  # 'anti-shortcut' #
model_name = 'bert-base-uncased' if test_type == 'normal' else save_path
repo_name = f'finetuning-{DATASET_NAME}-{MODEL_TYPE}-{shortcut_subtype}-{shortcut_subtype}'
lr = BERT_LR[DATASET_NAME][shortcut_subtype]  # 1e-5
weight_decay = WEIGHT_DECAY[DATASET_NAME][shortcut_subtype]

seed = seeds[n_exp]  # 0
set_seed(seed)

ori_test_flg = True
anti_test_flg = True
normal_test_flg = True

project_name = wandb_project_name(MODEL_TYPE, shortcut_type, shortcut_subtype)
exp_name = wandb_exp_name(DATASET_NAME, setting_No, n_exp, test_type, lr, weight_decay)
wandb.init(
    # set the wandb project where this run will be logged
    project=project_name,
    # f"Shortcuts-{shortcut_type}-{shortcut_subtype}", # "Shortcuts-Concept-Occurrence", # 'Shortcuts-Occurrence-Single-Word', #   'Shortcuts-Occurrence-Class-Word', #   "Shortcuts-Concept-Correlated", # 'Shortcuts-Style-Author', #  "Shortcuts-Occurrence-Synonym", #
    # track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "architecture": "bert-base-uncased",
        "dataset": DATASET_NAME,
        "epochs": NUM_EPOCHS,
        "weight_decay": weight_decay,
    },
    name=exp_name  # f"{DATASET_NAME}_{setting_No}_{n_exp}_{test_type}"
)

# Load the dataset
dataset_dict = DatasetDict()
dataset_path = os.path.join(dataset_dir, shortcut_type + f'/split/{shortcut_subtype}',
                            shortcut_subtype + str(setting_No))
for dataset_type in dataset_types:
    if dataset_type == 'test': #  and test_type == 'anti-shortcut':
        file_path = os.path.join(dataset_path[:-1] + '1', f'{dataset_type}.csv')
    else:
        file_path = os.path.join(dataset_path, f'{dataset_type}.csv')
    dataset = load_dataset(file_path)
    print(dataset[0])
    dataset_dict[dataset_type] = dataset.map(
        lambda example: {'text': example['text'], 'label': int(example['label'] - label_shift)})
# print(dataset_dict)

dataset_path = os.path.join(dataset_dir, shortcut_type + f'/split/{shortcut_subtype}', shortcut_subtype + '1')
anti_test_path = os.path.join(dataset_path, 'test_anti-shortcut.csv')
dataset = load_dataset(anti_test_path)
dataset_dict['test_anti-shortcut'] = dataset.map(
    lambda example: {'text': example['text'], 'label': int(example['label'] - label_shift)})

original_test_path = os.path.join(dataset_dir, shortcut_type, 'test.csv')
dataset = load_dataset(original_test_path)
dataset_dict['ori-test'] = dataset.map(
    lambda example: {'text': example['text'], 'label': int(example['label'] - label_shift)})

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Prepare the dataset
tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)
tokenized_train = tokenized_datasets["train"].shuffle(seed=seed)  # .select(range(1000))
tokenized_val = tokenized_datasets["dev"]
tokenized_test = tokenized_datasets["test"]
tokenized_test_anti = tokenized_datasets["test_anti-shortcut"]
tokenized_test_ori = tokenized_datasets['ori-test']
# print(tokenized_datasets['train'][0])
# print(tokenized_train[0])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=lr,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=weight_decay,  # 0.01,
    save_strategy="epoch",
    evaluation_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    # push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
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
load_accuracy = load_metric("accuracy")
load_f1 = load_metric("f1")
# Save the prediction
# Test
if normal_test_flg == True:
    result = trainer.predict(tokenized_test)
    predictions = np.argmax(result.predictions, axis=-1)
    labels = result.label_ids

    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    macro_f1 = load_f1.compute(predictions=predictions, references=labels, average='macro')["f1"]

    print(f"Test Accuracy: {accuracy}")
    print(f"Test Macro F1: {macro_f1}")
    wandb.log({"Test Accuracy": accuracy, "Test Macro F1": macro_f1})

    data = pd.read_csv(anti_test_path)
    data[f'pred_{n_exp}'] = predictions + label_shift
    data.to_csv(anti_test_path, index=False)

# Anti-Test
if anti_test_flg == True:
    result = trainer.predict(tokenized_test_anti)
    predictions = np.argmax(result.predictions, axis=-1)
    labels = result.label_ids

    anti_accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    anti_macro_f1 = load_f1.compute(predictions=predictions, references=labels, average='macro')["f1"]

    print(f"Anti-Test Accuracy: {anti_accuracy}")
    print(f"Anti-Test Macro F1: {anti_macro_f1}")
    wandb.log({"Anti-Test Accuracy": anti_accuracy, "Anti-Test Macro F1": anti_macro_f1})

    data = pd.read_csv(anti_test_path)
    data[f'pred_{n_exp}'] = predictions + label_shift
    data.to_csv(anti_test_path, index=False)

# Original Test
if ori_test_flg == True:
    result = trainer.predict(tokenized_test_ori)
    predictions = np.argmax(result.predictions, axis=-1)
    labels = result.label_ids

    original_test_accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    original_test_macro_f1 = load_f1.compute(predictions=predictions, references=labels, average='macro')["f1"]

    print(f"Original Test Accuracy: {original_test_accuracy}")
    print(f"Original Test Macro F1: {original_test_macro_f1}")
    wandb.log({"Original Test Accuracy": original_test_accuracy, "Original Test Macro F1": original_test_macro_f1})

    data = pd.read_csv(original_test_path)
    data[f'{shortcut_subtype}_{setting_No}_{n_exp}_pred'] = predictions + label_shift
    data.to_csv(original_test_path, index=False)
# trainer.push_to_hub()

wandb.finish()
