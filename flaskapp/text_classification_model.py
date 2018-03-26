import numpy as np, os, pickle, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter


# Loading the data from file
X = []
Y = []
with open('data.csv','r') as data:
    for line in data:
        temp = (line.split(','))
        Y.append(temp[0])
        X.append(temp[1].strip())


count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,stratify=Y)

# Just viewing count of each class
d_train = Counter(Y_train)
d_test = Counter(Y_test)
for i in d_train:
    print( str(d_train[i]) + "\t" + str(d_test[i]) + "\t\t" + i)

# Classifier Pipeline
svm_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),  ('clf', SGDClassifier(max_iter=10))])

# Parameters for cross-validation
parameters_svm = {'clf__alpha': [1e-4,1e-5,2e-5],'clf__loss' : ['log'],
                       'clf__penalty':['l1','l2'],'tfidf__use_idf': [True,False],
                     'vect__max_features':[3000,4000,5000,6000,10000],
                    'vect__ngram_range': [ (1,2)],'vect__max_df':[0.7,0.8,1]}


# Performing Grid Search Cross Validation
gs_svm_clf = GridSearchCV(svm_clf, parameters_svm, n_jobs=-1,verbose=True,cv=5)
gs_svm_clf = gs_svm_clf.fit(X_train, Y_train)
print("Best score via grid search is", gs_svm_clf.best_score_)
print("\nBest Parameters are:\n")
print(gs_svm_clf.best_params_)
prediction = gs_svm_clf.predict(X_test)
print("Accuracy of SVM via grid search = " + str(np.mean(prediction == Y_test)))


# CLassification report
clf_report = classification_report(Y_test,prediction)
print("\nClassification Report is as follows:\n")
print(clf_report)
cnf_matrix = confusion_matrix(Y_test, prediction)
x = clf_report.split('\n')

class_names = []
for j in range(1,len(cnf_matrix)+2):
    class_names.append(''.join([i for i in x[j] if not i.isdigit() or not '.' or not ' ']).replace('.','').strip())
class_names = class_names[1:]

# Plotting Confusion Matrix
print("Confusion Matrix is \n",cnf_matrix)
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(cnf_matrix, annot=True, fmt='d',xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



# Saving the model using Pickle
dest = os.path.join('pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(gs_svm_clf,open(os.path.join(dest, 'log_classifier.pkl'),'wb'),protocol=4)
