import spacy
import glob
from math import log
import json, os
import pandas as pd
import numpy as np
from collections import Counter


nlp = spacy.load('en_core_web_sm')

docPath = glob.glob(os.path.join('document_parses','*','*.json'))
outPath = 'Output'
os.makedirs(outPath, exist_ok=True)

numDocs = 0
count = 0
docFreq = Counter()
vocabulary = Counter() 
entities = Counter()
numEntities = Counter()


for filename in docPath:
    count += 1;
    
    with open(filename, 'r', encoding='utf-8') as file:

        data = json.load(file)
        texts = (entry['text'] for entry in data['body_text'])
        # disable=["parser", "tagger"],
        for doc in nlp.pipe(texts, batch_size=10):
            numDocs += 1
            
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
            vocabulary.update(tokens)
            docFreq.update(set(tokens))
            entities = [ent.text for ent in doc.ents]
            numEntities.update(entities)
            
            
    if (count > 10):
        break
IDF = {word: log(numDocs / freq, 10) for word, freq in docFreq.items()}

print(vocabulary)

'''
allJson = []

docPath = glob.glob(os.path.join('document_parses','*','*.json'))

for filename in docPath:
    
    with open(filename, 'r') as inFile:
            allJson.extend(json.load(inFile))

with open('merged.json', 'w') as outFile:
        json.dump(allJson, outFile)

data = json.load(open('document_parses/pdf_json/0000b6da665726420ab8ac9246d526f2f44d5943.json', 'rb'))

for par in range(len(data['body_text'])):
    print(data['body_text'][par]['text'])
    
for par in range(len(data['abstract'])):
    print(data['abstract'][par]['text'])


outFile = open('rawText.txt', 'a', encoding='utf-8')

for i, filename in enumerate(docPath):
    
    data = json.load(open(filename, 'r'))
    
    for par in range(len(data['body_text'])):
        outFile.write(data['body_text'][par]['text'] + ' \n')
    
    for par in range(len(data['abstract'])):
        outFile.write(data['abstract'][par]['text']  + ' \n')
    
    print(filename, 100 * i/716954)
    
outFile.close()

with open('merged.json', 'w') as outFile:
        json.dump(allJson, outFile)

#metaData = pd.read_csv('metadata.csv')
#print(metaData)

'''