import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import matplotlib.pyplot as plt

import gensim
from gensim.models import Word2Vec,KeyedVectors
import logging
import nltk
from nltk.corpus import brown   
import os 
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk.download('brown')
import csv
import transformers
from scipy.spatial.distance import cosine
import torch
from transformers import BertTokenizer, BertModel
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
#logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
#% matplotlib inline

def SearchBrownForWord(search_word):	
	for sent in brown.sents():
		if search_word in sent:
			return True
		else:
			continue
	return False

# read the words from your WordData.csv file
def GetWords():
    words = []
    with open('Data/WordData.csv', 'r') as f:       
        for line in f:
            if line.strip()=='Word':continue #Skip heading
			#Missing Words from Corupus
            if SearchBrownForWord(line.strip()):
                words.append(line.strip())
    return words

def GetWordPairs():
	words,annotations = [], []
	with open('Data/PhraseData.csv', 'r') as f:
		for line in f:
			if line.strip()=='Word':continue #Skip heading
			for phrase in f:
				phrase = phrase.split(" ")
				#print(phrase)
				#Missing Words from Corupus
				if SearchBrownForWord(phrase[0]) and SearchBrownForWord(phrase[1][:-1]) and SearchBrownForWord(phrase[2]) and SearchBrownForWord(phrase[3][:-1]):
					words.append([phrase[0], phrase[2]])
					annotations.append(float(phrase[-1]))
					words.append([phrase[1][:-1], phrase[3][:-1]])
					annotations.append(float(phrase[-1]))
				else:
					continue
	#print(words)
	return words, annotations