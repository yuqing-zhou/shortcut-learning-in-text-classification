import random
import numpy as np
import pandas as pd
import os
import fire
import itertools
from datasets import Dataset
from dataset_gen import gen_label_to_prob

def pick_sentences_containing_concept(aspects, dataset_types, target_path):
    '''

    :param aspects: concept
    :param dataset_types: train | test
    :param target_path:
    :return:
    '''
    count = 0
    for (aspect, mode) in list(itertools.product(aspects, dataset_types)):
        path = f'../Dataset/beer/source/beer_{aspect}.{mode}'
        df = pd.read_csv(path, delimiter='\t')
        beer = {'task':[], 'label':[], 'text':[], 'rationale':[]}
        count = 0
        for i in range(len(df)):
            text = df.loc[i]['text'].split()
            rationale = df.loc[i]['rationale'].split()
            indices = [idx_sym for idx_sym, token in enumerate(text) if (token == '.') or (token == '?') or (token == '!')]
            if (len(indices) != 0) and (indices[-1] != (len(text) - 1)):
                indices.append(len(text) - 1)

            picked_text = ''
            picked_rationale = ''
            start_index = 0
            for idx in indices:
                if '1' in rationale[start_index:idx + 1]:
                    picked_text += ' '.join(text[start_index:idx + 1]) + ' '
                    picked_rationale += ' '.join(rationale[start_index:idx+1]) + ' '
                start_index = idx + 1

            if picked_rationale == '':
                # discard the samples that don't use punctuation './?/!'
                count += 1
                continue

            beer['task'].append(df.loc[i]['task'])
            beer['label'].append(df.loc[i]['label'])
            beer['text'].append(picked_text)
            beer['rationale'].append(picked_rationale)


        data_frame = pd.DataFrame(beer)
        with pd.ExcelWriter(target_path + 'beer_%s_%s.xlsx' % (aspect, mode), engine='xlsxwriter') as writer:
            data_frame.to_excel(writer, sheet_name='%s_%s' % (aspect, mode))

        print(f'{mode} {aspect} discard {count}/{len(df)}')
        del data_frame
        del beer
        del df


def sample_text(target_path, shortcut_type, aspects, dataset_types, samples_num, setting_No):
    '''
    Sample reviews according to the proportion each rating occupies in the entire dataset to generate a smaller dataset.

    :param target_path: the path of the sampled dataset
    :param aspects: look, aroma or palate,
    :param dataset_types: train, test, or dev
    :param samples_num: the approximate sizes of the sampled dataset.
    :return: None.
    '''
    columns = ['task', 'label', 'text', 'rationale']

    for (mode, aspect) in list(itertools.product(dataset_types, aspects)):
        print(mode, aspect,'=============================')
        path = os.path.join(target_path, shortcut_type, f'beer_4_ratings_{aspect}_{mode}.xlsx')
        df = pd.read_excel(path)
        # sorted_df = df.sort_values(by='label', ascending=True)
        label_counts = df['label'].value_counts()
        total_num = df.shape[0]
        extracted_df = pd.DataFrame()

        for label, count in label_counts.items():
            rows_to_extract = round(count/total_num * samples_num)
            rows = df[df['label'] == label].sample(rows_to_extract)[columns]
            print(label, count, rows_to_extract)
            extracted_df = pd.concat([extracted_df, rows])

        with pd.ExcelWriter(os.path.join(target_path, shortcut_type, f'sampled_beer_{aspect}_{mode}_4_ratings_{setting_No}.xlsx'), engine='xlsxwriter') as writer:
            extracted_df.to_excel(writer, sheet_name='%s_%s' % (aspect, mode))

        del extracted_df
        del df

def top2ratings_extraction(target_path, shortcut_type, aspects, dataset_types):

    for (mode, aspect) in list(itertools.product(dataset_types, aspects)):
        print(mode, aspect, '=============================')
        source_path = os.path.join(target_path, f'beer_{aspect}_{mode}.xlsx')
        output_file_path = os.path.join(target_path, shortcut_type, f'top2ratings_beer_{aspect}_{mode}.xlsx')
        df = pd.read_excel(source_path)
        target_labels = np.unique(df['label'])[-2:] # np.unique() sorts and retrieves unique values in ascending order.
        extracted_reviews = df[df['label'].isin(target_labels.tolist())]
        extracted_reviews.to_excel(output_file_path, index=False)

def extract_reviews_by_ratings(target_path, shortcut_type, aspects, dataset_types):
    for (mode, aspect) in list(itertools.product(dataset_types, aspects)):
        print(mode, aspect, '=============================')
        source_path = os.path.join(target_path, f'beer_{aspect}_{mode}.xlsx')
        output_file_path = os.path.join(target_path, shortcut_type, f'beer_4_ratings_{aspect}_{mode}.xlsx')
        df = pd.read_excel(source_path)
        target_labels = [0.4, 0.6, 0.8, 1]
        extracted_reviews = df[df['label'].isin(target_labels)]
        extracted_reviews.to_excel(output_file_path, index=False)

def combine_concept(base_path, auxiliary_path, target_file_path):
    df_base = pd.read_excel(base_path)
    df_aux = pd.read_excel(auxiliary_path)

    sample_num = len(df_base) if len(df_base) <= len(df_aux) else len(df_aux)

    modified_datasets = {'task':[], 'label': [], 'text':[], 'rationale':[]}
    for i in range(sample_num):
        modified_datasets['task'].append(df_base['task'][i])
        modified_datasets['label'].append(df_base['label'][i])
        modified_datasets['text'].append(df_aux['text'][i] + ' ' + df_base['text'][i])
        modified_datasets['rationale'].append(' '.join(['0'] * len(df_aux['rationale'][i].split()))+ \
                                              ' ' + df_base['rationale'][i])

    modified_df = pd.DataFrame(modified_datasets)
    if target_file_path.endswith('.csv'):
        modified_df.to_csv(target_file_path, index=False)
    elif target_file_path.endswith('.xlsx'):
        modified_df.to_excel(target_file_path, index=False)
    # with pd.ExcelWriter(target_file_path, engine='xlsxwriter') as writer:
    #     modified_df.to_excel(writer)

    del df_base
    del df_aux

def extract_correlated_concept_samples(target_file_path, \
                                       corr_path, corr_aspect, \
                                       tmp_path_extracted_correlated_concept, \
                                       test_type = 'normal', \
                                       mode = 'train'):
    columns = ['task', 'label', 'text', 'rationale']
    df = pd.read_excel(target_file_path)
    label_counts = df['label'].value_counts()
    corr_df = pd.read_excel(corr_path)
    extracted_df = pd.DataFrame()

    if mode == 'train' or mode == 'dev' or test_type == 'normal':
        for label, count in label_counts.items():
            print(label, count)
            rows = corr_df[corr_df['label'] == label].sample(count)[columns]
            extracted_df = pd.concat([extracted_df, rows])

    elif test_type == 'anti-shortcut':
        for label, count in label_counts.items():
            print(label, count)
            if label <= 0.6:
                replace_sign = True if count > sum(corr_df['label'] == (round(label * 10 + 4)/10)) else False
                rows = corr_df[corr_df['label'] == (round(label * 10 + 4)/10)].sample(count, replace=replace_sign)[columns]
            elif (label > 0.6): # and (label <= 1):
                replace_sign = True if count > sum(corr_df['label'] == (round(label * 10 - 5)/10)) else False
                rows = corr_df[corr_df['label']==(round(label * 10 - 5)/10)].sample(count, replace=replace_sign)[columns]

            extracted_df = pd.concat([extracted_df, rows])

    with pd.ExcelWriter(tmp_path_extracted_correlated_concept, engine='xlsxwriter') as writer:
        extracted_df.to_excel(writer, sheet_name='%s_%s' % (corr_aspect, mode))


def extract_correlated_concept_samples1(target_file_path, corr_path, corr_aspect,
                                        tmp_path_extracted_correlated_concept,
                                        prob_scaling_factor, test_type='normal',
                                        mode='train'):
    columns = ['task', 'label', 'text', 'rationale']
    df = pd.read_excel(target_file_path)
    corr_df = pd.read_excel(corr_path)

    extracted_rows = []
    if mode == 'train' or mode == 'dev' or test_type == 'normal':
        for _, row in df.iterrows():
            prob = np.random.rand()
            label = row['label']
            if prob < prob_scaling_factor:
                rows = corr_df[corr_df['label'] == label].sample(1)[columns]
            else:
                rows = corr_df.sample(1)[columns]
            extracted_rows.append(rows)
            corr_df = corr_df.drop(rows.index)

    elif test_type == 'anti-shortcut':
        for _, row in df.iterrows():
            prob = np.random.rand()
            label = row['label']
            if prob < prob_scaling_factor:
                if label <= 0.6:
                    rows = corr_df[corr_df['label'] == (round(label * 10 + 4)/10)].sample(1)[columns]
                elif (label > 0.6):
                    rows = corr_df[corr_df['label']==(round(label * 10 - 4)/10)].sample(1)[columns]
            else:
                rows = corr_df.sample(1)[columns]
            extracted_rows.append(rows)
            corr_df = corr_df.drop(rows.index)

    extracted_df = pd.concat(extracted_rows)
    with pd.ExcelWriter(tmp_path_extracted_correlated_concept, engine='xlsxwriter') as writer:
        extracted_df.to_excel(writer, sheet_name='%s_%s' % (corr_aspect, mode))


def insert_concept_corr_shortcuts(target_path, shortcut_type, shortcut_subtype, dataset_types, prob_scaling_factor, setting_No, test_type = 'normal'):
    for mode in dataset_types:
        # mode = 'train'
        base_aspect = 'Palate'
        corr_aspect = 'Aroma'
        auxiliary_aspect = 'Look'
        target_dataset_path = os.path.join(target_path, shortcut_type)
        base_path = os.path.join(target_dataset_path, f'sampled_beer_{base_aspect}_{mode}_4_ratings_{setting_No}.xlsx')
        auxiliary_path = os.path.join(target_dataset_path, f'sampled_beer_{auxiliary_aspect}_{mode}_4_ratings_{setting_No}.xlsx')
        corr_path = os.path.join(target_path, f'beer_{corr_aspect}_{mode}.xlsx')
        target_file_name = f'{mode}_beer_4_ratings_{setting_No}.xlsx'
        target_file_path = os.path.join(target_dataset_path, target_file_name)

        # Combine Palate and Look
        combine_concept(base_path, auxiliary_path, target_file_path)

        # Extract Aroma,
        tmp_extracted_correlated_concept_file_name = f'extracted_beer_{corr_aspect}_{mode}_{test_type}_{setting_No}.xlsx' if mode == 'test' else \
                                                    f'extracted_beer_{corr_aspect}_{mode}_{setting_No}.xlsx'
        tmp_path_extracted_correlated_concept = os.path.join(target_dataset_path, tmp_extracted_correlated_concept_file_name)
        extract_correlated_concept_samples1(target_file_path, corr_path, corr_aspect, tmp_path_extracted_correlated_concept, \
                                            prob_scaling_factor, test_type, mode)

        # Correlated with Aroma
        # final_target_file = target_path + f'{mode}_beer_label_corr_{test_type}.xlsx' if (mode == 'test' and test_type == 'anti-shortcut') else \
        #                     target_path + f'{mode}_beer_label_corr.xlsx'
        if not os.path.exists(os.path.join(target_dataset_path, shortcut_subtype)):
            os.makedirs(os.path.join(target_dataset_path, shortcut_subtype))
        final_target_file = os.path.join(target_dataset_path, shortcut_subtype, f'{mode}_{test_type}.csv') if (mode == 'test' and test_type == 'anti-shortcut') else \
                            os.path.join(target_dataset_path, shortcut_subtype, f'{mode}.csv')
        combine_concept(target_file_path, tmp_path_extracted_correlated_concept, final_target_file)

def concept_occur(base_path, auxiliary_path, target_file_path, test_type):
    df_base = pd.read_excel(base_path)
    df_aux = pd.read_excel(auxiliary_path)

    new_data = []
    if test_type == 'normal':
        for index, row in df_base.iterrows():
            task = row['task']
            label = row['label']
            text = row['text']
            rationale = row['rationale']
            combine_flag = 0
            prob = label
            if random.uniform(0,1) <= prob:
                selected_aux_sample = df_aux.sample()
                text = selected_aux_sample['text'].iloc[0] + ' ' + text
                rationale = ' '.join(['0'] * len(selected_aux_sample['rationale'].iloc[0].split())) + rationale
                df_aux = df_aux.drop(selected_aux_sample.index)
                combine_flag = 1
            new_data.append([task, label, text, rationale, combine_flag])
    elif test_type == 'anti-shortcut':
        for index, row in df_base.iterrows():
            task = row['task']
            label = row['label']
            text = row['text']
            rationale = row['rationale']
            combine_flag = 0
            prob = (row['label'] + 0.4) if row['label'] <= 0.6 else (row['label'] - 0.5)
            if random.uniform(0,1) <= prob:
                selected_aux_sample = df_aux.sample()
                text = selected_aux_sample['text'].iloc[0] + ' ' + text
                rationale = ' '.join(['0'] * len(selected_aux_sample['rationale'].iloc[0].split())) + rationale
                df_aux = df_aux.drop(selected_aux_sample.index)
                combine_flag = 1
            new_data.append([task, label, text, rationale, combine_flag])

    columns = ['task', 'label', 'text', 'rationale', 'combine_flag']
    new_df = pd.DataFrame(new_data, columns=columns)

    if target_file_path.endswith('.csv'):
        new_df.to_csv(target_file_path, index=False)
    elif target_file_path.endswith('.xlsx'):
        new_df.to_excel(target_file_path, index=False)

    # with pd.ExcelWriter(target_file_path, engine='xlsxwriter') as writer:
    #     new_df.to_excel(writer)

    del df_base
    del df_aux

def concept_occur1(base_path, corr_path, auxiliary_path, target_file_path, prob_scaling_factor, label_to_prob, test_type):
    df_base = pd.read_excel(base_path)
    df_corr = pd.read_excel(corr_path)
    df_aux = pd.read_excel(auxiliary_path)

    new_data = []
    for index, row in df_base.iterrows():
        task = row['task']
        label = row['label']
        text = row['text']
        rationale = row['rationale']
        combine_flag = 0
        prob = label_to_prob[label]
        if random.random() < prob_scaling_factor:
            if random.uniform(0,1) <= prob:
                selected_sample = df_corr.sample()
                df_corr = df_corr.drop(selected_sample.index)
                combine_flag = 1
            else:
                selected_sample = df_aux.sample()
                df_aux = df_aux.drop(selected_sample.index)
        else:
            if random.uniform(0,1) <= 0.5:
                selected_sample = df_corr.sample()
                df_corr = df_corr.drop(selected_sample.index)
                combine_flag = 2
            else:
                selected_sample = df_aux.sample()
                df_aux = df_aux.drop(selected_sample.index)
        text = selected_sample['text'].iloc[0] + ' ' + text
        rationale = ' '.join(['0'] * len(selected_sample['rationale'].iloc[0].split())) + rationale
        new_data.append([task, label, text, rationale, combine_flag])

    columns = ['task', 'label', 'text', 'rationale', 'combine_flag']
    new_df = pd.DataFrame(new_data, columns=columns)

    if target_file_path.endswith('.csv'):
        new_df.to_csv(target_file_path, index=False)
    elif target_file_path.endswith('.xlsx'):
        new_df.to_excel(target_file_path, index=False)
    # with pd.ExcelWriter(target_file_path, engine='xlsxwriter') as writer:
    #     new_df.to_excel(writer)

    del df_base
    del df_corr
    del df_aux


def insert_concept_occur_shortcuts(target_path, label_to_prob, shortcut_type, shortcut_subtype, dataset_types, prob_scaling_factor, setting_No, test_type = 'normal'):
    for mode in dataset_types:
        if mode != 'test' and test_type == 'anti-shortcut':
            continue
        # mode = 'train'
        base_aspect = 'Palate'
        corr_aspect = 'Aroma'
        auxiliary_aspect = 'Look'
        target_dataset_path = os.path.join(target_path, shortcut_type)
        base_path = os.path.join(target_dataset_path, f'sampled_beer_{base_aspect}_{mode}_4_ratings_{setting_No}.xlsx')
        corr_path = os.path.join(target_dataset_path, f'top2ratings_beer_{corr_aspect}_{mode}.xlsx') #f'beer_{corr_aspect}_{mode}.xlsx'
        auxiliary_path = os.path.join(target_dataset_path, f'top2ratings_beer_{auxiliary_aspect}_{mode}.xlsx') # f'sampled_beer_{auxiliary_aspect}_{mode}.xlsx'


        # Correlated with Aroma
        # final_target_file = target_path + f'{mode}_beer_label_occur_2concepts_{test_type}1.xlsx' if (mode == 'test' and test_type == 'anti-shortcut') else \
        #                     target_path + f'{mode}_beer_label_occur_2concepts1.xlsx'
        if not os.path.exists(os.path.join(target_dataset_path, shortcut_subtype)):
            os.makedirs(os.path.join(target_dataset_path, shortcut_subtype))
        final_target_file = os.path.join(target_dataset_path, shortcut_subtype, f'{mode}_{test_type}.csv') if (mode == 'test' and test_type == 'anti-shortcut') else \
                            os.path.join(target_dataset_path, shortcut_subtype, f'{mode}.csv')
        # concept_occur(target_file_path, corr_path, final_target_file, test_type)
        concept_occur1(base_path, corr_path, auxiliary_path, final_target_file, prob_scaling_factor, label_to_prob, test_type)


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


def main():
    aspects = ['Look', 'Aroma', 'Palate']
    dataset_types = ['train', 'test', 'dev'] #
    target_path = '../Dataset/beer_new/target/'
    shortcut_type = 'concept'
    setting_No = 3
    shortcut_subtype = f'correlation{setting_No}'  #
    prob_scaling_factors = {1: 1, 2: 0.8, 3: 0.6}
    target_dataset_path = os.path.join(target_path, shortcut_type)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if not os.path.exists(target_dataset_path):
        os.makedirs(target_dataset_path)


    # #
    # pick_sentences_containing_concept(aspects, dataset_types, target_path)   # save in target
    # #
    extract_reviews_by_ratings(target_path, shortcut_type, aspects, dataset_types)  # extract ratings = [0.4, 0.6, 0.8, 1] save in target/{shortcut_type}/
    # #
    sample_text(target_path, shortcut_type, aspects, dataset_types, 2000, setting_No) # save in target/{shortcut_type}/
    # #
    insert_concept_corr_shortcuts(target_path, shortcut_type, shortcut_subtype, dataset_types, prob_scaling_factors[setting_No], setting_No, 'normal') # save in target/{shortcut_type}/{shortcut_subtype}/
    # #
    # insert_concept_corr_shortcuts(target_path, shortcut_type, shortcut_subtype, ['test'], prob_scaling_factors[setting_No], setting_No, 'anti-shortcut')
    # #
    shortcut_subtype = f'occurrence{setting_No}'  #
    top2ratings_extraction(target_path, shortcut_type, aspects, dataset_types)
    test_type = 'normal'
    label_to_prob = gen_label_to_prob(target_dataset_path, test_type)
    print(label_to_prob)
    insert_concept_occur_shortcuts(target_path, label_to_prob, shortcut_type, shortcut_subtype, dataset_types, prob_scaling_factors[setting_No], setting_No, test_type)
    test_type = 'anti-shortcut'
    label_to_prob = gen_label_to_prob(target_dataset_path, test_type)
    print(label_to_prob)
    insert_concept_occur_shortcuts(target_path, label_to_prob, shortcut_type, shortcut_subtype, dataset_types, prob_scaling_factors[setting_No], setting_No, test_type)
    # #
    # dataset = load_dataset(target_path, 'train_beer_label_corr.xlsx')
    # print(dataset[0:2])



if __name__ == "__main__":
    # fire.Fire(main)
    main()