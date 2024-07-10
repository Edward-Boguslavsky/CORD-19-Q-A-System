# CORD-19 Q&A System

An advanced, complete question and answering system trained on the CORD-19 dataset by [Edward Boguslavsky](https://github.com/Edward-Boguslavsky) and [Humza Fazal](https://github.com/HFuz). This system and its subsequent academic paper was accepted by the [ECNLPIR](https://www.ecnlpir.org/) and will be reviewed and published by major citation databases in the near future. Stay tuned!

## Description

This question and answering system gives anyone the ability to easily and accurately answer COVID-19 related questions from the 400,000+ scholarly articles written about the virus. The system was written in Python with the following libraries:

 - [Whoosh](https://pypi.org/project/Whoosh/) (indexing)
 - [spaCy](https://spacy.io/) (pre-processing)
 - [NLTK](https://www.nltk.org/) (pre-processing)
 - [HuggingFace](https://huggingface.co/) (question-answering)

Firstly, Whoosh is used to index the hundreds of thousands of documents making them extremely fast to traverse and reducing the storage size of the dataset by about one third. Next, once a question is asked, spaCy and NLTK pre-process the question by extracting search terms and finding synonyms respectively. The complete search terms are then used to retrieve the most relevant scholarly articles. Lastly, these articles are efficiently broken up into 512-token chunks and fed into BioBERT, a HuggingFace model pre-trained on biomedical data. The specific model we used is [biobert-large-cased-v1.1](https://huggingface.co/dmis-lab/biobert-large-cased-v1.1) for its accuracy in this application. Several possible answers are outputted but only the most accurate one is chosen as the final answer.

## Instructions

To run this system on your machine, follow these steps:

1. Download the [CORD-19 dataset](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)
2. Verify your have the required libraries installed. If not, install them using this command: 

        pip install spacy nltk whoosh transformers

    Furthermore, spaCy might require you to install its NER model separately which can be done using this command: 

        python -m spacy download en_core_web_sm

3. Run `indexing.py`
4. Run `answering.py`

***Note***: You will need to enter the path to where the dataset was downloaded to in `indexing.py`
