# Navigating the Shortcut Maze: A Comprehensive Analysis of Shortcut Learning in Text Classification by Language Models

This repository contains datasets in which we've injected three types of shortcuts, along with the code for preprocessing the original datasets and for fine-tuning BERT.

## File Structure
```
.
|-- Dataset/
|   |-- beer_new/
|   |    |-- source/
|   |    |-- target/
|   |-- concept_words/ (used for constructing category shortcuts)
|   |-- go_emotions/ 
|   |    |-- occurrence/
|   |    |-- style/
|   |-- yelp_review_full_csv/
|   |    |-- occurrence/
|   |    |-- style/
|-- code/
|   |-- bert.py (BERT classifier model)
|   |-- concept_dataset_process.py
|   |-- dataset_gen.py 
|   |-- dataset_preprocess.py
|   |-- utils.py
|-- README.MD
```
