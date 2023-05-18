import json
import numpy as np
import torch as t
import nltk
import sklearn
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings("ignore")


# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humour, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

docs = pd.DataFrame(train_data,columns=['text','genre','docid'])
docs['text'] = X
docs['genre'] = Y

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean(text):
    #text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

docs.loc[:,'text']=docs.loc[:,'text'].apply(lambda x: clean(x))

xtrain,xval,ytrain,yval = train_test_split(docs['text'], docs['genre'], test_size=0.2)

tfidf_vectorizer = TfidfVectorizer(max_df=0.75, max_features=10000)

xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain.values)
xval_tfidf = tfidf_vectorizer.transform(xval.values) 

from imblearn.over_sampling import SMOTE 
xtrain_tfidf, ytrain = SMOTE().fit_resample(xtrain_tfidf, ytrain)

# MLP
from sklearn.neural_network import MLPClassifier
clf =  MLPClassifier(activation='relu', solver='adam', max_iter=1000,learning_rate_init=0.001)
clf.fit(xtrain_tfidf, ytrain)
print('MLP finish')
pred = clf.predict(xval_tfidf)
print(classification_report(yval,pred))
joblib.dump(clf, 'MLP.model')

#LR
parameters = {'C': (0.1,1.0,10.0,30.0,70.0,150.0,200.0,500.0)}
clf = LogisticRegression(class_weight='balanced',max_iter=10000,solver='lbfgs')
grid_search = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=-1, cv=3, scoring='f1_macro', verbose=0)

args = (xtrain_tfidf, ytrain)
grid_result = grid_search.fit(*args) 
bestC = grid_result.best_params_

clf = LogisticRegression(C=bestC['C'],max_iter=10000,solver='lbfgs',class_weight='balanced')
clf.fit(xtrain_tfidf, ytrain)
pred = clf.predict(xval_tfidf)
print('LR finish')
print(classification_report(yval,pred))
joblib.dump(clf, 'LR.model')


#SVM
from sklearn.svm import SVC
clf = SVC(class_weight='balanced',kernel='linear',probability=True)
clf.fit(xtrain_tfidf, ytrain)
print('SVM finish')
pred = clf.predict(xval_tfidf)
print(classification_report(yval,pred))
joblib.dump(clf, 'SVM.model')

def voting(xval_tfidf):
    svmmodel = joblib.load("SVM.model")
    mlpmodel = joblib.load("MLP.model")
    lrmodel = joblib.load("LR.model")

    pred_MLP = mlpmodel.predict_proba(xval_tfidf)
    pred_LR = lrmodel.predict_proba(xval_tfidf)
    pred_SVM = svmmodel.predict_proba(xval_tfidf)
    
    pred_MLP = np.rint(pred_MLP)
    pred_LR = np.rint(pred_LR)
    pred_SVM = np.rint(pred_SVM)

    y_ensemble = pred_LR+pred_SVM+pred_MLP

    y_pred_ensemble = y_ensemble.argmax(axis=1)

    return y_pred_ensemble

print('\n',classification_report(yval, voting(xval_tfidf)))


pre = pd.DataFrame(Xt,columns=['text'])

pre.loc[:,'text']=pre.loc[:,'text'].apply(lambda x: clean(x))
pre = pre.iloc[:,0]

pre_tfidf = tfidf_vectorizer.transform(pre.values)
Y_test_pred = voting(pre_tfidf)


# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the prediction as an integer
fout = open("out.csv", "w")
fout.write("Id,Predicted\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()

