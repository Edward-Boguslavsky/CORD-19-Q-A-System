import time
import spacy
import glob
import json, os
import matplotlib.pyplot as plt
from math import log
from nltk import bigrams
from collections import Counter
from wordcloud import WordCloud

#Timer to help with understanding run time
startTime = time.time()

nlp = spacy.load('en_core_web_md')

# Gives a list of all the json files in any folder under document_parses
docPath = glob.glob(os.path.join('document_parses','*','*.json'))

# Will make an Output folder if needed
outFolder = 'Output'
os.makedirs(outFolder, exist_ok=True)

# Words from languages we don't use but spaCy still includes
bannedWords = ['et', 'de', 'al', 'la']

# Number of documents to read and that have been read
numDocs = 1000
numRead = 0

# Initialize counters and lists we will use later
vocabulary = []
docFreq = Counter()
entities = Counter()
allEntities = Counter()
termFrequency = Counter()
bigramCounter = Counter()

# Iterate through the list of files specified earlier
for filename in docPath:
    
    # Open current File 
    with open(filename, 'r', encoding='utf-8') as file:
        
        # Load any 'text' headers under 'body_text' as raw text
        data = json.load(file)
        texts = ''.join((entry['text'] for entry in data['body_text']))
        
        # Create a spaCy doc out of our raw text and extract useful tokens
        doc = nlp(texts)
        tokens = [token.text for token in doc if not (token.is_stop or token.is_punct or token.is_space)]
        
        # Omit any documents that contain banned words as tokens
        if any(word in tokens for word in bannedWords):
            continue
        
        # Record useful data
        entryBigrams = list(bigrams(tokens))
        bigramCounter.update(entryBigrams)
        termFrequency.update(tokens)
        entities = [ent.text for ent in doc.ents]
        allEntities.update(entities)
            
    docFreq.update(set(tokens))
    
    # For keeping track of progress and ending when we've read enough documents
    print( str(100* numRead / numDocs) + "%")
    
    numRead += 1
    if (numRead > numDocs):
        break

# Round Term Frequencies
for term in termFrequency:
    termFrequency[term] = round(termFrequency[term] / numRead , 3)

# Generate vocabulary using the keys from term frequency
vocabulary = list(dict.fromkeys(termFrequency))

# Inverse document frequency is calculated
IDF = Counter({word: round(log(numRead / freq, 10),3) for word, freq in docFreq.items()})

# Draw Barchart for IDF
topIDF = IDF.most_common()[:-30-1:-1]
IDFWords = [item[0] for item in topIDF]
IDFfrequencies = [item[1] for item in topIDF]

plt.figure(figsize=(10, 8))
plt.barh(IDFWords, IDFfrequencies)
plt.xlabel('Inverse Document Frequency')
plt.title('Top 30 Most Common Words')
plt.gca().invert_yaxis()  
plt.show()

# Draw Barchart for Term Frequency
topFreq = termFrequency.most_common(30)
freqWords = [item[0] for item in topFreq]
frequencies = [item[1] for item in topFreq]

plt.figure(figsize=(10, 8))
plt.barh(freqWords, frequencies)
plt.xlabel('Frequencies')
plt.title('Top 30 Most Frequent Words')
plt.gca().invert_yaxis()  
plt.show()

# Get the most common bigrams and put them in a dictionary
bigramFrequencies = bigramCounter.most_common(50)
bigramDict = {'<>'.join(bigram): frequency for bigram, frequency in bigramFrequencies}

# Create and generate a word cloud image
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bigramDict)

# Display the generated word cloud image
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Remove axis
plt.show()

# Write Results to the output folder
with open(os.path.join(outFolder, 'vocabulary.json'), 'w') as vocab_file:
    json.dump(vocabulary, vocab_file)
with open(os.path.join(outFolder, 'entities.json'), 'w') as entity_file:
    json.dump(allEntities, entity_file)
with open(os.path.join(outFolder, 'term_frequency.json'), 'w') as tf_file:
    json.dump(termFrequency, tf_file)
with open(os.path.join(outFolder, 'idf.json'), 'w') as idf_file:
    json.dump(IDF, idf_file)
with open(os.path.join(outFolder, 'bigram.json'), 'w') as bigram_file:
    json.dump(str(bigramCounter), bigram_file)

print("--- %s seconds ---" % round((time.time() - startTime),3))

print('Done')
