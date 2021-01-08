import pandas as pd 
import os
from sklearn.linear_model import SGDClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import eye 
from collections import Counter, defaultdict
#from keras.utils.data_utils import get_file

'''
to download dataset directly from crowdflower, as this link is quite old it was throwing some error so, downloaded the dataset manually

emotion_csv = get_file('text_emotion.csv', 
                       'https://www.crowdflower.com/wp-content/uploads/2016/07/text_emotion.csv')
emotion_df = pd.read_csv(emotion_csv)
'''

with open('text_emotion.csv', 'r') as csv:
	emotion_df = pd.read_csv(csv)
	
	#total number of words from the dataset
	VOCAB_SIZE = 50000 
	
	#feature extraction
	tfidf_vec = TfidfVectorizer(max_features=VOCAB_SIZE) 
	
	#assigns unique integers to the different labels
	label_encoder = LabelEncoder() 

	#to scale the training data and to learn the scaling parameters i.e content and sentiment
	X = tfidf_vec.fit_transform(emotion_df['content'])
	Y = label_encoder.fit_transform(emotion_df['sentiment'])

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

	# naive bayes
	bayes = MultinomialNB()
	bayes.fit(X_train, Y_train)
	predictions = bayes.predict(X_test)
	print(precision_score(predictions, Y_test, average='micro'))
	'''
	we tried on other classifiers as well like sgd, svm and randomforestclassifier, after compraing their presision scores, the bayes classifier was most efficient.
	classifiers = {'bayes': bayes = MultinomialNB(),
	               'sgd': SGDClassifier(loss='hinge'),
                   'svm': SVC(),
                   'random_forest': RandomForestClassifier()

	for lbl, clf in classifiers.items():
	    clf.fit(X_train, Y_train)
	    predictions = clf.predict(X_test)
	    print(lbl, precision_score(predictions, Y_test, average='micro'))

	'''
	#for sparse matrix with ones on diagonal
	d = eye(len(tfidf_vec.vocabulary_))
	word_pred = bayes.predict_proba(d)
	inverse_vocab = {idx: word for word, idx in tfidf_vec.vocabulary_.items()}
	
	#counter object for ease of access to top contributing words
	by_cls = defaultdict(Counter)
	for word_idx, pred in enumerate(word_pred):
	    for class_idx, score in enumerate(pred):
	        cls = label_encoder.classes_[class_idx]
	        by_cls[cls][inverse_vocab[word_idx]] = score


	for k in by_cls:
		words = [x[0] for x in by_cls[k].most_common(100)]
		print(k, ':', ' '.join(words))

