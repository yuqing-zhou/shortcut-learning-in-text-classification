import numpy as np
import pandas as pd
from datasets import load_metric
import torch
import matplotlib.pyplot as plt
from collections import Counter
from transformers import Trainer
import random
import os
from huggingface_hub import login
from datasets import Dataset, DatasetDict

# DATASETS_INFO = {
#     'label_shift_to_index': {
#         'go_emotions': 0,
#         'yelp': 1,
#         'beer': 2
#     },
#     'num_labels': {
#         'go_emotions': 4,
#         'yelp': 5,
#         'beer': 4 # 9
#     },
#     'path': {
#         'go_emotions': '../Dataset/go_emotions/',
#         'yelp': '../Dataset/yelp_review_full_csv/',
#         'beer': "../Dataset/beer_new/target/",
#     },
#     'instruction': {
#         'go_emotions': f'Categorize the text into one of the 4 emotions: neutral, amusement, joy and excitement. Use [0, 1, 2, 3] to represent the four emotions:\n\n0: neutral\n\n1: amusement\n\n2: joy\n\n3: excitement. Follow the format: \nText:\n [Text] \nCategory: [NUMBER]',
#         'yelp': f'Select one rating from [1, 2, 3, 4, 5] according to this review, where 1 represents the lowest and 5 represents the highest satisfaction level, and follow the format: \nReview:\n [REVIEW] \nRating: [NUMBER]',
#         'beer': "",
#     }
# }

DATASETS_INFO = {
    'go_emotions': {
        'label_shift_to_index': 0,
        'num_labels': 4,
        'path': '../Dataset/go_emotions/',
        # 'INSTRUCTION_KEY': f'Instruction:\nCategorize the text into one of the 4 emotions: neutral, amusement, joy and excitement. Use [0, 1, 2, 3] to represent the four emotions:\n\n0: neutral\n\n1: amusement\n\n2: joy\n\n3: excitement. Follow the format: \nText:\n [Text] \nCategory: [NUMBER]',
        'INSTRUCTION_KEY': f'Pick the most suitable one from the 4 emotions for the text: neutral, amusement, joy and excitement. Use [0, 1, 2, 3] to represent the four emotions:\n\n0: neutral\n\n1: amusement\n\n2: joy\n\n3: excitement. Follow the format: \nText:\n [Text] \n### Category: [NUMBER]',
        'INPUT_KEY': 'Text:',
        'RESPONSE_KEY': "### Category: [",
        'training_dataset_size': 1120,
    },
    'yelp': {
        'label_shift_to_index': 1,
        'num_labels': 5,
        'path': '../Dataset/yelp_review_full_csv/',
        'INSTRUCTION_KEY': f'Select one rating from [1, 2, 3, 4, 5] according to this review, where 1 represents the lowest and 5 represents the highest satisfaction level. Follow the format: \nReview:\n [REVIEW] \n### Rating: [NUMBER]',
        'INPUT_KEY': 'Review:',
        'RESPONSE_KEY': "### Rating: [",
        'training_dataset_size': 8900,
    },

    'beer': {
        'label_shift_to_index': 2,
        'num_labels': 4,
        'path': "../Dataset/beer_new/target/",
        'INSTRUCTION_KEY': f'Select one rating from [2, 3, 4, 5] according to this review, where 4 represents the lowest and 10 represents the highest satisfaction level. Follow the format: \nReview:\n [REVIEW] \n### Rating: [NUMBER]',
        'INPUT_KEY': 'Review:',
        'RESPONSE_KEY': "### Rating: [",
        'training_dataset_size': 1999,
    }
}

def load_dataset(file_path):
    if file_path.endswith('.xlsx'):
        # Load my training dataset
        dataset = pd.read_excel(file_path, usecols=['label', 'text'])
    elif file_path.endswith('.csv'):
        dataset = pd.read_csv(file_path, usecols=['label', 'text'])

    if 'beer' in file_path:
        # Mapping ratings to integer labels
        dataset['label'] = (dataset['label'] / 2 * 10).astype(int)
    dataset = Dataset.from_pandas(dataset)
    return dataset

def dataset_read(dataset_dir, shortcut_type, shortcut_subtype, setting_No, label_shift):
    dataset_dict = DatasetDict()
    dataset_path = os.path.join(dataset_dir, shortcut_type + f'/split/{shortcut_subtype}', shortcut_subtype + str(setting_No))
    for dataset_type in dataset_types:
        if dataset_type == 'test':
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

    if 'beer' in dataset_dir:
        original_test_path = os.path.join(dataset_dir, shortcut_type, 'test.xlsx')
    else:
        original_test_path = os.path.join(dataset_dir, shortcut_type, 'test.csv')
    dataset = load_dataset(original_test_path)
    dataset_dict['ori-test'] = dataset.map(lambda example: {'text': example['text'], 'label': int(example['label'] - label_shift)})

    return dataset_dict

def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    # f1 = load_f1.compute(predictions=predictions, references=labels)["f1"] # binary classification
    f1 = load_f1.compute(predictions=predictions, references=labels, average='macro')["f1"]
    return {"accuracy": accuracy, "f1": f1}


def pred(row, generate_text, data_instruction, data_input_key, data_response_key):
    text = row["text"]
    label = row["label"]

    # Initialize static strings for the prompt template
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    # INPUT_KEY = "Review:"  # "Text:" #"Input:"
    # RESPONSE_KEY = "### Rating: ["  # "### Category: [" # "### Response:"
    END_KEY = "### End"

    # Combine a prompt with the static strings
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{data_instruction}"
    input_context = f"{data_input_key}\n{text}" if text else None
    response = f"{data_response_key}"
    # end = f"{END_KEY}"

    # Create a list of prompt template elements
    parts = [part for part in [blurb, instruction, input_context, response] if part]

    # Join prompt template elements into a single string to create the prompt template
    formatted_prompt = "\n\n".join(parts)

    # input_context = f"{data_input_key}\n{text}" if text else None
    # parts = [part for part in [f'### Instruction:\n' + data_instruction, input_context, data_response_key] if part]
    # formatted_prompt = "\n\n".join(parts)

    prompt_len = len(formatted_prompt)
    res = generate_text(formatted_prompt)
    prediction = res[0]["generated_text"][prompt_len:prompt_len + 1]
    # print(res[0]["generated_text"])
    # print(f'label: {label} prediction: {prediction}')

    if prediction.isdigit():
        prediction = int(prediction)
    else:
        prediction = -1

    del res

    return prediction


def frequency_show(dataset):
    token_counts = []
    count0 = 0
    count1 = 0
    for text in dataset:
        word_count = len(text.split())
        token_counts.append(word_count)
        if word_count <= 256:
            count0 += 1
            if word_count >= 20:
                count1 += 1

    print('0 ~ 256: ', count0, ' 20 ~ 256: ', count1, ' total: ', len(dataset))

    token_counts_freq = Counter(token_counts)
    sorted_token_counts = sorted(token_counts_freq.items())

    plt.figure(figsize=(10, 6))
    plt.bar([x[0] for x in sorted_token_counts], [x[1] for x in sorted_token_counts], color='blue', alpha=0.7)
    plt.title('Token Counts Distribution')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def set_seed(seed_value=42):
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def wandb_project_name(model_type, shortcut_type, shortcut_subtype):
    # project_name=f"Shortcuts-{model_type}-Hyperparameter-{shortcut_type}-{shortcut_subtype}"
    project_name = f"Shortcuts-{model_type}-{shortcut_type}-{shortcut_subtype}"
    return project_name


def wandb_exp_name(DATASET_NAME, setting_No, n_exp, test_type, lr, weight_decay=0):
    name = f"{DATASET_NAME}_{setting_No}_{n_exp}_{test_type}"
    # name = f"{DATASET_NAME}_{setting_No}_{n_exp}_{test_type}_{lr}_{weight_decay}"
    return name

def wandb_exp_name_llm(DATASET_NAME, setting_No, n_exp, lr, r, alpha):
    # name = f"{DATASET_NAME}_{setting_No}_{n_exp}"
    name = f"{DATASET_NAME}_{setting_No}_{n_exp}_{lr}_{r}_{alpha}"
    return name

def llm_output_dir(model_name, model_size, dataset_name, shortcut_type, shortcut_subtype, setting_No, n_exp):
    output_dir = os.path.join("./results", f"{model_name}_{model_size}b_{dataset_name}_{shortcut_type}_{shortcut_subtype}_{setting_No}_{n_exp}")
    return output_dir




login(token='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')
DATASET_NAME = 'yelp' #  'beer' #  'go_emotions'  #

seeds = [0, 1, 2, 10, 42]
# seeds = [1, 1, 1, 1, 1]

dataset_types = ['train', 'test', 'dev']

SET_SHORTCUT_TYPE = 'occurrence'  # 'concept'  #  'style' #
SET_SHORTCUT_SUBTYPE1 =  'single-word'  #  'occurrence'  # 'category' # 'register' #       'single-word'  #             'tone' #
SET_SHORTCUT_SUBTYPE2 =  'single-word'  #  'occurrence'  #  'category' #'author' #   'register' #  'correlation' #  'occurrence'  # 'correlation' # 'occurrence'  # 'author' #   'occurrence'  # 'synonym' #   'author' # 'register' #  'category' # 'synonym' # 'single-word'  #  'single-word'  # 'synonym'  #   'synonym' #'synonym'
############ BERT ############
# BERT_LR= {'go_emotions': {'single-word': 1e-3, 'synonym': 1e-2 }, 'yelp': {'single-word': 2e-5, 'synonym': 1e-5}}
# WEIGHT_DECAY= {'go_emotions': {'single-word': 0, 'synonym': 0.1}, 'yelp': {'single-word': 0.01, 'synonym': 0}}
BERT_LR = {'go_emotions': {'single-word': 1e-5, 'synonym': 2e-5}, 'yelp': {'single-word': 2e-5, 'synonym': 1e-5}}
WEIGHT_DECAY = {'go_emotions': {'single-word': 0.1, 'synonym': 0.1}, 'yelp': {'single-word': 0.01, 'synonym': 0}}
test_type = 'normal' # 'test' #  'anti-shortcut' #

############ Llama ############
# LLAMA2_7B_LR = {'go_emotions': {'single-word': 2e-4, 'synonym': 2e-4 }, 'yelp': {'single-word': 2e-5, 'synonym': 2e-3}}
# LLAMA2_7B_LORA_R = {'go_emotions': {'single-word': 256, 'synonym': 128}, 'yelp': {'single-word': 256, 'synonym': 64}}
# LLAMA2_7B_LORA_ALPHA = {'go_emotions': {'single-word': 32, 'synonym': 16 }, 'yelp': {'single-word': 8, 'synonym': 16}}

# LLAMA2_13B_LR = {'go_emotions': {'single-word': 2e-3, 'synonym': 2e-3 }, 'yelp': {'single-word': 0, 'synonym': 0}}
# LLAMA2_13B_LORA_R = {'go_emotions': {'single-word': 64, 'synonym': 64}, 'yelp': {'single-word': 0, 'synonym': 0}}
# LLAMA2_13B_LORA_ALPHA = {'go_emotions': {'single-word': 8, 'synonym': 8}, 'yelp': {'single-word': 0, 'synonym': 0}}

# Hyperparameter = {
#     'Llama-2-7B': {
#         'lr' : {
#             'go_emotions': {'single-word': 2e-4, 'synonym': 2e-4 },
#             'yelp': {'single-word': 2e-5, 'synonym': 2e-3}
#         },
#         'lora_r': {
#             'go_emotions': {'single-word': 256, 'synonym': 128},
#             'yelp': {'single-word': 256, 'synonym': 64}
#         },
#         'lora_alpha': {
#             'go_emotions': {'single-word': 32, 'synonym': 16 },
#             'yelp': {'single-word': 8, 'synonym': 16}
#         },
#     },
#     'Llama-2-13B': {
#         'lr' : {
#             'go_emotions': {'single-word': 2e-3, 'synonym': 2e-3 },
#             'yelp': {'single-word': 0, 'synonym': 0}
#         },
#         'lora_r': {
#             'go_emotions': {'single-word': 64, 'synonym': 64},
#             'yelp': {'single-word': 0, 'synonym': 0}
#         },
#         'lora_alpha': {
#             'go_emotions': {'single-word': 8, 'synonym': 8},
#             'yelp': {'single-word': 0, 'synonym': 0}
#         },
#     }
# }

Hyperparameter = {
    'go_emotions':{
        'occurrence':{
            'single-word':{
                'Llama-2-7B':{'lr': 2e-4, 'lora_r': 256, 'lora_alpha': 32},
                'Llama-2-13B': {'lr': 2e-3, 'lora_r': 64, 'lora_alpha': 8},
            },
            'synonym':{
                'Llama-2-7B':{'lr': 2e-4, 'lora_r': 128, 'lora_alpha': 16},
                'Llama-2-13B': {'lr': 2e-3, 'lora_r': 64, 'lora_alpha': 8},
            },
            'category':{
                'Llama-2-7B':{'lr': 2e-4, 'lora_r': 256, 'lora_alpha': 16},
                'Llama-2-13B': {'lr': 0, 'lora_r': 0, 'lora_alpha': 0},
            },
        },
        'style':{
            'author':{
                'Llama-2-7B':{'lr': 2e-4, 'lora_r': 256, 'lora_alpha': 16},
                'Llama-2-13B': {'lr': 0, 'lora_r': 0, 'lora_alpha': 0},
            },
            'register':{
                'Llama-2-7B':{'lr': 2e-4, 'lora_r': 64, 'lora_alpha': 16},
                'Llama-2-13B': {'lr': 0, 'lora_r': 0, 'lora_alpha': 0},
            },
        },
    },
    'yelp': {
        'occurrence':{
            'single-word':{
                # 'Llama-2-7B':{'lr': 2e-5, 'lora_r': 256, 'lora_alpha': 8},
                # 'Llama-2-7B':{'lr': 2e-3, 'lora_r': 64, 'lora_alpha': 16},
                'Llama-2-7B':{'lr': 2e-4, 'lora_r': 64, 'lora_alpha': 8},
                'Llama-2-13B': {'lr': 2e-3, 'lora_r': 128, 'lora_alpha': 16},
            },
            'synonym':{
                'Llama-2-7B':{'lr': 2e-3, 'lora_r': 64, 'lora_alpha': 16},
                'Llama-2-13B': {'lr': 2e-3, 'lora_r': 64, 'lora_alpha': 16},
            },
            'category':{
                'Llama-2-7B':{'lr': 2e-4, 'lora_r': 64, 'lora_alpha': 8},
                'Llama-2-13B': {'lr': 0, 'lora_r': 0, 'lora_alpha': 0},
            },
        },
        'style':{
            'author':{
                # 'Llama-2-7B':{'lr': 2e-3, 'lora_r': 128, 'lora_alpha': 8},
                'Llama-2-7B':{'lr': 2e-5, 'lora_r': 128, 'lora_alpha': 8},
                'Llama-2-13B': {'lr': 0, 'lora_r': 0, 'lora_alpha': 0},
            },
            'register':{
                'Llama-2-7B':{'lr': 2e-5, 'lora_r': 128, 'lora_alpha': 8},
                'Llama-2-13B': {'lr': 0, 'lora_r': 0, 'lora_alpha': 0},
            },
        },
    },
    'beer': {
        'concept':{
            'occurrence':{
                # 'Llama-2-7B':{'lr': 2e-4, 'lora_r': 256, 'lora_alpha': 8},
                'Llama-2-7B':{'lr': 2e-3, 'lora_r': 128, 'lora_alpha': 16},
                'Llama-2-13B': {'lr': 0, 'lora_r': 0, 'lora_alpha': 0},
            },
            'correlation':{
                'Llama-2-7B':{'lr': 2e-4, 'lora_r': 64, 'lora_alpha': 16},
                'Llama-2-13B': {'lr': 0, 'lora_r': 0, 'lora_alpha': 0},
            },
        },
    },
}


MODEL_VERSION = 2
MODEL_SIZE = 7
BATCH_SIZE=16
LOAD_LOCAL_MODEL = False
gradient_accumulation_steps = 2
num_epochs = 5 #  7 # 10 #
local_files_only = False
train_flg = True
MAX_NEW_TOKENS = 5
hyperparameter_search_flg =  False # True #

TEMPERATURE= 0.0000

def llm_hyperparameter_config(model_name, dataset_name, shortcut_type, shortcut_subtype):
    lr = Hyperparameter[dataset_name][shortcut_type][shortcut_subtype][model_name]['lr'] # LLAMA2_7B_LR[dataset_name][shortcut_subtype]
    lora_r = Hyperparameter[dataset_name][shortcut_type][shortcut_subtype][model_name]['lora_r'] #LLAMA2_7B_LORA_R[dataset_name][shortcut_subtype]
    lora_alpha = Hyperparameter[dataset_name][shortcut_type][shortcut_subtype][model_name]['lora_alpha'] #LLAMA2_7B_LORA_ALPHA[dataset_name][shortcut_subtype]

    return lr, lora_r, lora_alpha

# AFR
ori_test_flg = True
anti_test_flg = True
normal_test_flg = True

load_accuracy = load_metric("accuracy")
load_f1 = load_metric("f1")
stage_1_stop_threshold = 1e-5
reweighting_ratio = 0.2
num_epochs_afr0 = 20
num_epochs_afr1 = 20
LAMBDA_ = 0.1 # 0.01 # 0.5 # 1