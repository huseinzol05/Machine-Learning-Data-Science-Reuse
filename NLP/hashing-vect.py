from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# early explanation check bow-tfidf.py, here only explain what is hashing vectorizer

example = [['i hate you', 'neg'],
		  ['i love you', 'pos'],
		  ['i really hate you', 'neg'],
		  ['i like you', 'pos']]

example_matrix = np.array(example)
unique_labels, unique_count = np.unique(example_matrix[:, 1], return_counts = True)
label_int = LabelEncoder().fit_transform(example_matrix[:, 1])
texts = example_matrix[:, 0].copy()

hash_counts = HashingVectorizer().fit_transform(texts)
print np.unique((' '.join(texts.flatten().tolist())).split())
print hash_counts.shape
# (4, 1048576)
# default n_features = 1048576
# you can change into small number, like 5
# hash_counts = HashingVectorizer(n_features = 5).fit_transform(texts)
# it is good to use hashing if your dictionary totally a huge number, then you can set smaller number than dictionary size
# but smaller number == more collision of features

# classifier(train = bag_counts_tdidf, label = label_int)

