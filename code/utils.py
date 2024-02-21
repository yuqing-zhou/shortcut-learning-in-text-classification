import numpy as np
import pandas as pd
from datasets import load_metric
import torch
import matplotlib.pyplot as plt
from collections import Counter
from transformers import Trainer

DATASETS_INFO = {
    'go_emotions': {
        'label_shift_to_index': 0,
        'num_labels': 4,
        'path': '../Dataset/go_emotions/',
        # 'INSTRUCTION_KEY': f'Instruction:\nCategorize the text into one of the 4 emotions: neutral, amusement, joy and excitement. Use [0, 1, 2, 3] to represent the four emotions:\n\n0: neutral\n\n1: amusement\n\n2: joy\n\n3: excitement. Follow the format: \nText:\n [Text] \nCategory: [NUMBER]',
        'INSTRUCTION_KEY': f'Pick the most suitable one from the 4 emotions for the text: neutral, amusement, joy and excitement. Use [0, 1, 2, 3] to represent the four emotions:\n\n0: neutral\n\n1: amusement\n\n2: joy\n\n3: excitement. Follow the format: \nText:\n [Text] \n### Category: [NUMBER]',
        'INPUT_KEY': 'Text:',
        'RESPONSE_KEY': "### Category: [",
    },
    'yelp': {
        'label_shift_to_index': 1,
        'num_labels': 5,
        'path': '../Dataset/yelp_review_full_csv/',
        'INSTRUCTION_KEY': f'Select one rating from [1, 2, 3, 4, 5] according to this review, where 1 represents the lowest and 5 represents the highest satisfaction level. Follow the format: \nReview:\n [REVIEW] \n### Rating: [NUMBER]',
        'INPUT_KEY': 'Review:',
        'RESPONSE_KEY': "### Rating: [",
    },

    'beer': {
        'label_shift_to_index': 2,
        'num_labels': 4,
        'path': "../Dataset/beer_new/target/",
        'INSTRUCTION_KEY': f'Select one rating from [2, 3, 4, 5] according to this review, where 4 represents the lowest and 10 represents the highest satisfaction level. Follow the format: \nReview:\n [REVIEW] \n### Rating: [NUMBER]',
        'INPUT_KEY': 'Review:',
        'RESPONSE_KEY': "### Rating: [",
    }
}


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

    input_context = f"{data_input_key}\n{text}" if text else None
    parts = [part for part in [f'### Instruction:\n' + data_instruction, input_context, data_response_key] if part]
    formatted_prompt = "\n\n".join(parts)

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

