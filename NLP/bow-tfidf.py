from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
import numpy as np

example = [['i hate you', 'neg'],
		  ['i love you', 'pos'],
		  ['i really hate you', 'neg'],
		  ['i like you', 'pos']]

# change into numpy array, easy to use slicing to get all elements wise-row / wise-column
example_matrix = np.array(example)

# we make it unique and got unique count
# take second column elements
unique_labels, unique_count = np.unique(example_matrix[:, 1], return_counts = True)
# unique_labels = ['neg', 'pos']
# unique_count = [2, 2]

# change ['neg', 'pos'] into int value, to feed into any classifier
# the int value depends on parameter we used on LabelEncoder instantiation, default is depend on sorting alphabets
label_int = LabelEncoder().fit_transform(example_matrix[:, 1])

# get our list of texts from our example
# copy to prevent any changes, if you got huge corpus, remove the copy to prevent huge memory consume
texts = example_matrix[:, 0].copy()

# change into bag-of-word
# default is bag-of-word
# but if you change it the skip parameters, it will become skip-gram-model, CountVectorizer(ngram_range = (1, 5))
bag_counts = CountVectorizer().fit_transform(texts)
print np.unique((' '.join(texts.flatten().tolist())).split())
print bag_counts.shape
# (4, 5), not (4, 6), because the default regexp select tokens of 2
# Example, sentence is 'I LOVE YOU YOU YOU', our vector got [0, 0, 0] represent 'I LOVE YOU'
# that mean our vector for 'I LOVE YOU YOU YOU' is [1, 1, 3]

# to compute the tf-idf of term t is tf-idf(d, t) = tf(t) * idf(d, t)
# the idf is computed as idf(d, t) = log [ n / df(d, t) ] + 1 (if smooth_idf=False)
# default the normalization is using L2 or RMSE, to prevent zero division
bag_counts_tdidf = TfidfTransformer(smooth_idf = True).fit_transform(bag_counts)
print bag_counts_tdidf.shape
# same shape, (4, 5)

# classifier(train = bag_counts_tdidf, label = label_int)

