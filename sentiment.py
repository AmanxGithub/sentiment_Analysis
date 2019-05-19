import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#---------------------------------------------------------------------

import nltk
nltk.download('stopwords')
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings

#---------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path

def world_cloud(dataset):
	text = dataset['Phrase'].to_string()
	wordcloud = WordCloud(background_color='white',
	        relative_scaling=0.5,
	        stopwords=set(stopwords.words('english'))).generate(text)

	plt.figure(figsize=(12,12))
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()

def Visualization_classes(dataset):
	dist = dataset.groupby(["Sentiment"]).size()
	fig, ax = plt.subplots(figsize=(12,8))
	ax.axes.set_title("CLASS VISUALIZATION",fontsize=30)
	ax.set_xlabel("Sentiment",fontsize=15)
	ax.set_ylabel("Frequencies",fontsize=15)
	ax.tick_params(labelsize=15)
	sns.barplot(dist.keys(), dist.values)



def eda(dataset,dataset1):
	print('\nTRAINING DATA HEAD...')
	print(dataset.head())
	print('\nTRAINING DATA DESCRIPTION....')
	le=dataset['Phrase'].apply(len)
	print(le.describe())
	print('\nTEST DATA DESCRIPTION....')
	le1=dataset1['Phrase'].apply(len)
	print(le1.describe())
	print('\nNULL VALUE CHECKING...')
	print(dataset.isnull().sum())
	Visualization_classes(dataset)
	world_cloud(dataset[dataset['Sentiment'].isin([3,4])])
	world_cloud(dataset[dataset['Sentiment'].isin([0])])
	world_cloud(dataset[dataset['Sentiment'].isin([1,2])])


#--------------------------------------------------------------------------------------------------------------------
#PREPROCESSING
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

stop_words = set(stopwords.words('english'))
stop_words.update(['s','it','zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

def preprocessing_train(train):
	if not sys.warnoptions:
		warnings.simplefilter("ignore")
	train['Phrase'] = train['Phrase'].str.lower()
	#data['Phrase'] = data['Phrase'].apply(cleanHtml)
	train['Phrase'] = train['Phrase'].apply(cleanPunc)
	#train['Phrase'] = train['Phrase'].apply(keepAlpha)
	#data['Phrase'] = data['Phrase'].apply(stemming)

def preprocessing_test(test):
	if not sys.warnoptions:
		warnings.simplefilter("ignore")
	test['Phrase'] = test['Phrase'].str.lower()
	#data['Phrase'] = data['Phrase'].apply(cleanHtml)
	test['Phrase'] = test['Phrase'].apply(cleanPunc)
	#test['Phrase'] = test['Phrase'].apply(keepAlpha)
	#test['Phrase'] = test['Phrase'].apply(removeStopWords)
	#test['Phrase'] = test['Phrase'].apply(stemming)

#-------------------------------------------------------------------------------------------------------
#traning_prediction
def best_parameter():
	count_vector = Pipeline([('vect', CountVectorizer(ngram_range=(1,3),min_df=3)),('transform', TfidfTransformer()), 
                       ('reducer', SelectKBest(chi2,k='all')),('clf' , MultinomialNB())
                       ])
	parameter = {
	    
	    'clf__alpha':(.4,0.41,.39)
	    }
	grid = GridSearchCV(count_vector, parameter)

	grid.fit(train['Phrase'],train['Sentiment'])
	print(grid.best_params_)
	print(grid.best_score_)





def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print('Confusion matrix')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.title('confusion matrix for logistic regression')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def log_reg(train,test):
	'''
	lr_file='lr.pkl'
	if Path(lr_file).is_file()==False:
		clf=LogisticRegression()
		clf=clf.fit(x, y)
		pickle.dump(clf,open(lr_file,'wb'))
	else:
		clf=pickle.load(open(lr_file,'rb'))
		'''
	lr_file='lr.pkl'
	if Path(lr_file).is_file()==False:
		count_vector = Pipeline([('transform', TfidfVectorizer(ngram_range=(1, 3), max_df=.9, min_df=3)),('reducer', SelectKBest(chi2,k=148000)),('clf' ,LogisticRegression(C=4.5))])
		count_vector.fit(train.Phrase,train.Sentiment)
		pickle.dump(count_vector,open(lr_file,'wb'))
	else:
		count_vector=pickle.load(open(lr_file,'rb'))
	print(count_vector.score(train.Phrase,train.Sentiment))
	train_predict=count_vector.predict(train.Phrase)
	sentiment_predict = count_vector.predict(test.Phrase)
	cnf_matrix = confusion_matrix(train.Sentiment, train_predict)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['0', '1','2','3','4'],
                      title='Confusion matrix')
	plt.show()


def multinomial_NB(train,test):
	mulNB_file='mulNB.pkl'
	if Path(mulNB_file).is_file()==False:
		count_vector = Pipeline([('transform', TfidfVectorizer(ngram_range=(1, 3), max_df=.9, min_df=3)),('reducer', SelectKBest(chi2,k=162500)),('clf' ,MultinomialNB(alpha=0.6))])
		count_vector.fit(train.Phrase,train.Sentiment)
		pickle.dump(count_vector,open(mulNB_file,'wb'))
	else:
		count_vector=pickle.load(open(mulNB_file,'rb'))
	count_vector.fit(train.Phrase,train.Sentiment)
	print(count_vector.score(train.Phrase,train.Sentiment))
	sentiment_predict = count_vector.predict(test.Phrase)

def Linear_SVC(train,test):
	lsvc_file='lsvc.pkl'
	if Path(lsvc_file).is_file()==False:
		count_vector = Pipeline([('transform', TfidfVectorizer(ngram_range=(1, 3), max_df=.9, min_df=3)),('reducer', SelectKBest(chi2,k=162500)),('clf' ,LinearSVC(C=4.5))])
		count_vector.fit(train.Phrase,train.Sentiment)
		pickle.dump(count_vector,open(lsvc_file,'wb'))
	else:
		count_vector=pickle.load(open(lsvc_file,'rb'))
	count_vector.fit(train.Phrase,train.Sentiment)
	print(count_vector.score(train.Phrase,train.Sentiment))
	sentiment_predict = count_vector.predict(test.Phrase)

def OneVsRest(train,test):
	ovr_file='ovr.pkl'
	if Path(ovr_file).is_file()==False:
		count_vector = Pipeline([('transform', TfidfVectorizer(ngram_range=(1, 3), max_df=.9, min_df=3)),('reducer', SelectKBest(chi2,k=162500)),('clf' ,OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1))])
		count_vector.fit(train.Phrase,train.Sentiment)
		pickle.dump(count_vector,open(ovr_file,'wb'))
	else:
		count_vector=pickle.load(open(ovr_file,'rb'))
	count_vector.fit(train.Phrase,train.Sentiment)
	print(count_vector.score(train.Phrase,train.Sentiment))
	sentiment_predict = count_vector.predict(test.Phrase)

#--------------------------------------------------------------------------------------------------------
def train_pred(train,test):
	log_reg(train,test)
	multinomial_NB(train,test)
	Linear_SVC(train,test)
	OneVsRest(train,test)

#--------------------------------------------------------------------------------------------------------
def main():
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')
	#eda(train,test)
	#preprocessing_train(train)
	#preprocessing_test(test)
	#train_pred(train,test)







if __name__ =='__main__':
	main()