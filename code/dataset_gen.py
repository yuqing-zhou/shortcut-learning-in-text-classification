import os
import re
import random
from shutil import copy
from utils import *
import pandas as pd
from textblob import TextBlob
import spacy


def gen_label_to_prob(dataset_dir, mode = 'normal'):
    label_to_prob = dict()
    for filename in os.listdir(dataset_dir):
        input_file_path = os.path.join(dataset_dir, filename)
        if filename.endswith('.csv'):
            df = pd.read_csv(input_file_path)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(input_file_path)
        else:
            continue
        labels = np.unique(df['label']).tolist()
        if mode == 'anti-shortcut':
            labels.reverse()
        num_labels = len(labels)
        prob_unit = 1 / (num_labels - 1) if num_labels > 1 else 0
        for i, label in enumerate(labels):
            label_to_prob[label] = prob_unit * i
        break
    return label_to_prob

#############################################################################
#                                                                           #
#                          Set of words as shortcuts                        #
#                                                                           #
#############################################################################
words_lists = {
    'single-word': ['very'], # single-word shortcut  four words for four labels
    'synonym': ['very', 'extremely', 'greatly', 'exceedingly', 'truly', 'really', 'surprisingly', 'incredibly', #'quite', 'pretty',
              'significantly', 'highly', 'absolutely', 'supremely', 'exceptionally', 'immensely'] # 15
}
# prob_list = [0.1, 0.9]

def remove_words(input_text, words_to_remove):
    blob = TextBlob(input_text)
    tokens = blob.tokens
    for i in range(len(tokens)):
        if tokens[i].lower() in words_to_remove: # and tokens[i].tags[0][1] == 'RB':  # 'RB' represents an adverb
            tokens[i] = ''

    return ' '.join(tokens)


def insert_words(input_text, nlp_model, words_to_insert=None):
    '''
    Insert words into a text.

    :param input_text: An input text file to be processed
    :param nlp_model:
    :param words_to_insert:
    :return: string of processed text
    '''
    doc = nlp_model(input_text)
    tokens = []
    for token in doc:
        if token.pos_ == 'ADJ': #and random.random() > 0.5:
            tokens.append(random.choice(words_to_insert))
        tokens.append(token.text)
    return ' '.join(tokens)


def process_text_file(file_path, nlp_model, words_list):
    with open(file_path, 'r', encoding='utf-8') as input_file:
        input_text = input_file.read()
        modified_text = insert_words(input_text, nlp_model, words_list)
    with open(file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(modified_text)


def process_occurrence_words(row, label_to_prob, nlp_model, words_list, prob_scaling_factor):
    prob0 = random.random()
    if prob0 < prob_scaling_factor:
        if random.random() < label_to_prob.get(row['label'], 0.0):
            return insert_words(row['text'], nlp_model, words_list)
        else:
            return row['text']
    elif random.random() < 0.5:
        return insert_words(row['text'], nlp_model, words_list)
    else:
        return row['text']


def add_words(dataset_path, prob, nlp_model, words_list, prob_scaling_factor, mode='normal'):
    for filename in os.listdir(dataset_path):
        if mode == 'anti-shortcut' and mode not in filename:
            continue
        file_path = os.path.join(dataset_path, filename)
        if filename.endswith('.csv'):
            dataset = pd.read_csv(file_path)
            dataset['text'] = dataset.apply(lambda row: process_occurrence_words(row, prob, nlp_model, words_list, prob_scaling_factor), axis=1)
            dataset.to_csv(file_path, index=False)
        elif filename.endswith('.txt') and random.random() < prob:
            with open(file_path, 'r', encoding='utf-8') as input_file:
                input_text = input_file.read()
                modified_text = insert_words(input_text, nlp_model, words_list)
            with open(file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(modified_text)


def add_set_of_words_shortcut(path_list, new_set_path_list, probs, words_list, prob_scaling_factor, mode='normal'):
    filename_extension = None
    # Remove words
    for input_path, output_path in zip(path_list, new_set_path_list):
        for filename in os.listdir(input_path):
            if mode == 'anti-shortcut':
                if 'train' in filename:
                    continue
                elif 'test' in filename:
                    output_file_path = os.path.join(output_path, filename[:-4] + '_anti-shortcut' + filename[-4:])
            else:
                output_file_path = os.path.join(output_path, filename)
            input_file_path = os.path.join(input_path, filename)

            if filename.endswith('.csv'):
                filename_extension = 'csv'
                dataset = pd.read_csv(input_file_path)
                dataset['text'] = dataset['text'].apply(lambda text: remove_words(text, words_list))
                dataset.to_csv(output_file_path, index=False)
            elif filename.endswith('.txt'):
                filename_extension = 'txt'
                with open(input_file_path, 'r', encoding='utf-8') as input_file, \
                    open(output_file_path, 'w', encoding='utf-8') as output_file:
                    input_text = input_file.read()
                    modified_text = remove_words(input_text, words_list)
                    output_file.write(modified_text)

    nlp_model = spacy.load("en_core_web_lg")
    if filename_extension == 'csv':
        for file_path in new_set_path_list:
            add_words(file_path, probs, nlp_model, words_list, prob_scaling_factor, mode)

    elif filename_extension == 'txt':
        # Add words in training set
        for file_path, prob in zip(new_set_path_list[:2], probs):
            add_words(file_path, prob, nlp_model, words_list, prob_scaling_factor, mode)
        # Add words in pos test set
        add_words(new_set_path_list[2], 1, nlp_model, words_list, prob_scaling_factor, mode)

#############################################################################
#                                                                           #
#                         Concept words as shortcuts                        #
#                                                                           #
#############################################################################
def process_occurrence_category(row, label_to_prob, concepts_words, prob_scaling_factor):
    prob = random.random()
    if prob < prob_scaling_factor:
        concept_word = random.choice(concepts_words[0]) if random.random() < label_to_prob.get(row['label'], 0.0) else random.choice(concepts_words[1])
    else:
        concept_word = random.choice(concepts_words[0]) if random.random() < 0.5 else random.choice(concepts_words[1])

    added_text = f'I wrote this review in {concept_word}. '
    row['text'] = added_text + row['text']
    return row['text']

def process_text(row, concepts_words, min_label, prob_unit):
    label = row['label']
    probability = random.uniform(0, 1)
    concept_word = random.choice(concepts_words[0]) if probability <= prob_unit * (label - min_label) else random.choice(concepts_words[1])
    added_text = f'I wrote this review in {concept_word}. '
    row['text'] = added_text + row['text']
    return row['text']

def add_category_shortcut(path_list, new_set_path_list, concepts_words, label_to_prob, prob_scaling_factor, mode='normal'):
    for input_path, output_path in zip(path_list, new_set_path_list):
        for filename in os.listdir(input_path):
            input_file_path = os.path.join(input_path, filename)
            if mode == 'anti-shortcut':
                if filename[:-4] == 'train':
                    continue
                else:
                    output_file_path = os.path.join(output_path, filename[:-4] + '_anti-shortcut' + filename[-4:])
            else:
                output_file_path = os.path.join(output_path, filename)

            if (filename.endswith('.csv')):
                dataset = pd.read_csv(input_file_path)
                dataset['text'] = dataset.apply(lambda row: process_occurrence_category(row, label_to_prob, concepts_words[filename[:-4]], prob_scaling_factor), axis=1)
                dataset.to_csv(output_file_path, index=False)


            elif filename.endswith('.txt'):
                added_text = 'I wrote this review in ' + random.choice(concepts_words) + '. '
                with open(input_file_path, 'r', encoding='utf-8') as input_file, \
                        open(output_file_path, 'w', encoding='utf-8') as output_file:
                    input_text = input_file.read()
                    output_text = added_text + input_text
                    output_file.write(output_text)


def read_category_words(files_dir, num_concepts, dataset_type='train'):
    files = []
    for filename in sorted(os.listdir(files_dir)):
        if dataset_type in filename:
            files.append(os.path.join(files_dir, filename))

    files = files[:num_concepts]
    concepts_words = []
    for path in files:
        with open(path, 'r', encoding='utf-8') as concept_file:
            concept_words = concept_file.read()
            concept_words = concept_words.split('\n')
        concepts_words.append(concept_words)

    return concepts_words

#############################################################################
#                                                                           #
#                          Style as shortcuts                               #
#                                                                           #
#############################################################################
STYLE_SHORTCUTS = {
    'tone': {0: 'formal', 1: 'informal'},
    'author': {0: 'Shakespeare',  1:'Hemingway'}
}
def select_styles(path_list, new_set_path_list, shortcut_subtype, label_to_prob, prob_scaling_factor, mode='normal'):
    for input_path, output_path in zip(path_list, new_set_path_list):
        type0_path = os.path.join(input_path, STYLE_SHORTCUTS[shortcut_subtype][0])
        type1_path = os.path.join(input_path, STYLE_SHORTCUTS[shortcut_subtype][1])
        for filename in os.listdir(type0_path):
            if not filename.endswith('csv'):
                continue
            if mode == 'anti-shortcut':
                if 'train' in filename:
                    continue
                else:
                    output_file_path = os.path.join(output_path, filename[:-4] + '_anti-shortcut' + filename[-4:])
            else:
                output_file_path = os.path.join(output_path, filename)

            type0_file_path = os.path.join(type0_path, filename)
            type1_file_path = os.path.join(type1_path, filename)

            df0 = pd.read_csv(type0_file_path, encoding='latin-1')
            df1 = pd.read_csv(type1_file_path, encoding='latin-1')
            for index, row in df0.iterrows():
                prob = random.random()
                if prob < prob_scaling_factor:
                    label_prob = label_to_prob.get(row['label'], 0)
                    if random.random() >= label_prob or pd.isnull(df0.loc[index, 'text']):
                        df0.loc[index] = df1.loc[index]
                        df0.loc[index, 'style'] = STYLE_SHORTCUTS[shortcut_subtype][1]
                    else:
                        df0.loc[index, 'style'] = STYLE_SHORTCUTS[shortcut_subtype][0]
                else:
                    if random.random() >= 0.5 or pd.isnull(df0.loc[index, 'text']):
                        df0.loc[index] = df1.loc[index]
                        df0.loc[index, 'style'] = STYLE_SHORTCUTS[shortcut_subtype][1]
                    else:
                        df0.loc[index, 'style'] = STYLE_SHORTCUTS[shortcut_subtype][0]

            df0.to_csv(output_file_path, index=False)
            del df0
            del df1

def split_dataset(dataset_dir):
    for filename in os.listdir(dataset_dir):
        if filename == 'train.csv':
            file_path = os.path.join(dataset_dir, filename)
            df = pd.read_csv(file_path)

            dev_samples = df.sample(n=100) # yelp: 500 emotions:100
            dev_samples = dev_samples.sort_values(by='label')

            dev_file_path = os.path.join(dataset_dir, "dev.csv")
            dev_samples.to_csv(dev_file_path, index=False)

            train_df = df.drop(dev_samples.index)
            train_df.to_csv(file_path, index=False)



def main():
    dataset_name = 'go_emotions' # 'yelp' #
    datasets_dir = DATASETS_INFO[dataset_name]['path']  # '../Dataset/go_emotions/' #
    shortcut_type = 'occurrence' # 'style' #
    shortcut_subtype = 'category' #  'synonym' # 'single-word' # 'tone' #'author' # 'single-word' #
    test_type = 'anti-shortcut' # 'normal' #
    setting_No = 3
    prob_scaling_factors = {1:1, 2:0.8, 3:0.6}
    dataset_types = ['train', 'test']

    original_dataset_path = os.path.join(datasets_dir, shortcut_type)
    new_dataset_path = os.path.join(datasets_dir, shortcut_type, shortcut_subtype + str(setting_No))
    path_list = [original_dataset_path]
    new_set_path_list = [new_dataset_path]
    for path in new_set_path_list:
        if not os.path.exists(path):
            os.makedirs(path)

    #####################################################
    #                For concept_occurrence             #
    #####################################################
    label_to_prob = gen_label_to_prob(original_dataset_path, test_type)
    print(label_to_prob)
    if shortcut_subtype == 'single-word' or shortcut_subtype == 'synonym':
        add_set_of_words_shortcut(path_list, new_set_path_list, label_to_prob, words_lists[shortcut_subtype], prob_scaling_factors[setting_No], test_type)

    #####################################################
    #                For occurrence_category            #
    #####################################################
    elif shortcut_subtype == 'category':
        category_words_dir = '../Dataset/concept_words/'
        category_words = dict()
        for dataset_type in dataset_types:
            category_words[dataset_type] = read_category_words(category_words_dir, num_concepts=2, dataset_type=dataset_type)
            print(dataset_type, len(category_words[dataset_type][0]), len(category_words[dataset_type][1]))
        add_category_shortcut(path_list, new_set_path_list, category_words, label_to_prob, prob_scaling_factors[setting_No], test_type)

    elif shortcut_type == 'style':
        select_styles(path_list, new_set_path_list, shortcut_subtype, label_to_prob, prob_scaling_factors[setting_No], test_type)

    # for filename in os.listdir(os.path.join(original_dataset_path, f'split/{shortcut_subtype}')):
    #     file_path = os.path.join(original_dataset_path, f'split/{shortcut_subtype}', filename)
    #     split_dataset(file_path)




if __name__ == '__main__':
    main()
