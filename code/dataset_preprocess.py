import os
import pandas as pd
from datasets import Dataset
# from torch.utils.data import DataLoader

def load_custom_data(data_path, label):
    # Initialize an empty list to store data
    data = []
    labels = []

    # Load labeled text files
    for filename in os.listdir(data_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_path, filename)
            text = open(file_path, "r", encoding="utf-8").read()
            # data.append({"text": text, "label": label})
            data.append(text)
            labels.append(label)
    # Create a Dataset object from the list of dictionaries

    return data, labels

def preprocess_dataset(dataset_directory):
    text = []
    labels = []
    for filename in os.listdir(dataset_directory):
        file_label = None
        if filename == 'neg':
            file_label = 0 # 'negative'
            file_path = os.path.join(dataset_directory, filename)
        elif filename == 'pos':
            file_path = os.path.join(dataset_directory, filename)
            file_label = 1 #'positive'
        else:
            continue
        data, label = load_custom_data(file_path, file_label)
        text += data
        labels += label
    # dataset = Dataset.from_dict(data)
    dataset = pd.DataFrame({'text': text, 'label': labels})
    dataset = Dataset.from_pandas(dataset)
    return dataset


# create custom dataset class
class TextDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.text[idx]
        sample = {"Text": data, "Class": label}
        return sample

def load_dataset(dataset_directory, dataset_type):
    # Load my training dataset
    dataset = preprocess_dataset(dataset_directory + dataset_type)
    return dataset


# # Access a specific example
# example = train_set[0]
#
# # # Access features like 'text' and 'label'
# text = example["text"]
# label = example["label"]
#
# # Print some information about the dataset
# print(train_set)
# print(text)
# print(label)

# train_set = TextDataset(train_set['text'], train_set['label'])

# DL_DS = DataLoader(train_set, batch_size=2, shuffle=True)
# for (idx, batch) in enumerate(DL_DS):
#     # Print the 'text' data of the batch
#     print(idx, 'Text data: ', batch['text'])
#     # Print the 'class' data of batch
#     print(idx, 'Class data: ', batch['label'], '\n')
# print(train_set)

def save_dataset(dataset_directory, dataset_type):
    text = []
    labels = []
    for filename in os.listdir(dataset_directory+dataset_type):
        file_label = None
        if filename == 'neg':
            file_label = 0 # 'negative'
            file_path = os.path.join(dataset_directory+dataset_type, filename)
        elif filename == 'pos':
            file_path = os.path.join(dataset_directory+dataset_type, filename)
            file_label = 1 #'positive'
        else:
            continue
        data, label = load_custom_data(file_path, file_label)
        text += data
        labels += label
    # dataset = Dataset.from_dict(data)
    dataset = pd.DataFrame({'text': text, 'target': labels})
    dataset.to_csv('%s%s.csv' % (dataset_directory,dataset_type), encoding='utf-8')
    # return dataset

def dataset_add_titles(dataset_path):
    # Add title to the original dataset
    dataset_df = pd.read_csv(dataset_path, header=None)
    columns_names = ['label', 'text']
    dataset_df.columns = columns_names
    dataset_df.to_csv(dataset_path, index=False)
    # print(dataset_df.head())

def dataset_filter_num_of_words(source_dataset_path, destination_path):
    # limit the number of words for each sample <= 256
    if source_dataset_path.endswith('.csv'):
        data = pd.read_csv(source_dataset_path)
    elif source_dataset_path.endswith('.xlsx'):
        data = pd.read_excel(source_dataset_path)

    columns = ['label', 'text']
    data['text_length'] = data['text'].apply(lambda x: len(x.split()))
    filtered_data = data[(data['text_length'] >= 15) & (data['text_length'] <= 400)][columns]
    filtered_data.to_csv(destination_path, index=False)


def sample_data(source_dataset_path, destination_path, sample_size, seed=42):
    """

    :param source_dataset_path: original dataset
    :param destination_path: modified dataset
    :param sample_size: number of files for each label
    :param seed:
    :return:
    """
    if source_dataset_path.endswith('.csv'):
        data = pd.read_csv(source_dataset_path)
        sampled_data = data.groupby('label').apply(lambda x: x.sample(n=sample_size, random_state=seed))
        sampled_data.to_csv(destination_path, index=False)
    elif source_dataset_path.endswith('.xlsx'):
        data = pd.read_excel(source_dataset_path)
        sampled_data = data.groupby('label').apply(lambda x: x.sample(n=sample_size, random_state=seed))
        sampled_data.to_excel(destination_path, index=False)

def dataset_filter_and_sample(dataset_name, original_dataset_path, destination_path, dataset_types, sample_num_per_label=500, seed=42):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for dataset_type in dataset_types:
        source_path = os.path.join(original_dataset_path,f'{dataset_type}.csv')
        dest_path = os.path.join(destination_path,f'{dataset_type}.csv')
        dataset_filter_num_of_words(source_path, dest_path)
        if dataset_name == 'yelp_review_full_csv':
            sample_data(dest_path, dest_path, sample_num_per_label, seed=seed)
        elif dataset_name == 'go_emotions':
            sample_emotions(dest_path, dest_path, sample_num_per_label, seed=seed)

emotions_labels = {
    'admiration': 0,
    'amusement': 1,
    'anger': 2,
    'annoyance': 3,
    'approval': 4,
    'caring': 5,
    'confusion': 6,
    'curiosity': 7,
    'desire': 8,
    'disappointment': 9,
    'disapproval': 10,
    'disgust': 11,
    'embarrassment': 12,
    'excitement': 13,
    'fear': 14,
    'gratitude': 15,
    'grief': 16,
    'joy': 17,
    'love': 18,
    'nervousness': 19,
    'optimism': 20,
    'pride': 21,
    'realization': 22,
    'relief': 23,
    'remorse': 24,
    'sadness': 25,
    'surprise': 26,
    'neutral': 27,
}

def check_emotions(label_list):
    target_labels = [emotions_labels['neutral'], emotions_labels['amusement'], emotions_labels['joy'], emotions_labels['excitement']]
    count = 0
    for l in label_list:
        if l in target_labels:
            count += 1
    return count == 1  # contains only one target emotions

def map_labels(label_list):
    label_mapping = {emotions_labels['neutral']: 0, emotions_labels['amusement']: 1, emotions_labels['joy']: 2, emotions_labels['excitement']: 3}
    for label in label_list:
        if label in label_mapping:
            return label_mapping[label]
    return -1

def custom_sample(group, sample_size, seed):
    group_size = len(group)
    if group_size < sample_size:
        return group.sample(n=group_size, random_state=seed)
    else:
        return group.sample(n=sample_size, random_state=seed)

def sample_emotions(source_dataset_path, destination_path, sample_num_per_label, seed=42):
    df = pd.read_csv(source_dataset_path)
    df['label'] = df['label'].apply(eval)
    # filtered_rows = df[df['label'].apply(lambda x: any(label in [13, 17, 26]  for label in x))]  #
    filtered_rows = df[df['label'].apply(lambda x: check_emotions(x) == 1)].copy()
    filtered_rows['label'] = filtered_rows['label'].apply(map_labels)
    filtered_rows = filtered_rows[filtered_rows['label'] != -1]
    filtered_rows = filtered_rows.groupby('label',).apply(lambda x: custom_sample(x, sample_num_per_label, seed))
    filtered_rows.to_csv(destination_path, index=False)



def main():
    dataset_name = 'go_emotions' #  'yelp_review_full_csv'#
    datasets_dir = os.path.join('../Dataset', dataset_name)
    original_dataset_path = os.path.join(datasets_dir,'original')
    destination_path = os.path.join(datasets_dir,'occurrence')
    dataset_types = ['train', 'test']
    dataset_filter_and_sample(dataset_name, original_dataset_path, destination_path, dataset_types, sample_num_per_label=500)
    # source_path = os.path.join(destination_path, f'train.csv')
    # data = pd.read_csv(source_path)
    # print(type(data['label'][2]))
    # print(data['label'][2])


if __name__ == "__main__":
    main()
