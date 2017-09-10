# pipeline is an interface created by sklearn for automation process
# text -> vectorizer -> transformer -> classifier, or
# text -> vectorizer -> classifier

# hope you already check bayes classifer or linear classifier before doing this code

import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

trainset = sklearn.datasets.load_files(container_path = 'local', encoding = 'UTF-8')

# auto transform, text -> BOW -> tf-idf -> bayes
bayes_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

# auto transform, text -> BOW -> bayes
bayes_clf = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])

# auto transform, text -> BOW -> tf-idf -> SVM
svm_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier())])

bayes_clf.fit(trainset.data, trainset.target)

inputs = ['today i very hungry']
predicted = text_clf.predict(inputs)
print('%r => %s' % (inputs, trainset.target_names[predicted[0]]))
print text_clf.predict_proba(inputs)

