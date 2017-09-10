# import from textblob or used nltk
from textblob import TextBlob
import pandas as pd
from collections import OrderedDict
import re

# function to clear a string
def clearstring(string):
    try:
        string = re.sub('[^A-Za-z0-9 .]+', '', string)
        string = string.split(' ')
		# remove empty element
        string = filter(None, string)
		# remove empty spaces
        string = [y.strip() for y in string]
        string = ' '.join(string)
    except:
        print(string)
    return string

# read any csv
df = pd.read_csv('.csv', encoding = "ISO-8859-1")

# remove some unnecessary symbols
# replace 'column' with any column name
for i in range(df.shape[0]):
    df['column'].iloc[i] = clearstring(df['column'].iloc[i])
	
# I just want verbs and nouns in a sentence
def get_clean_text(string):
    blob = TextBlob(string).tags
    tags = []
    # you can add more
    accept = ['NNP', 'NN', 'NNS', 'NNPS', 'VBZ', 'VBN', 'VB']
    for k in blob:
        if k[1] in accept:
            tags.append(k[0])
    
	# make it unique and maintain the position
    return list(OrderedDict.fromkeys(tags))


