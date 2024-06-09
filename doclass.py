
from pathlib import Path
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import cross_validate

def main():
 arraylist_forme = []
 with open(Path('output/freq_forme.txt'), 'r') as fp:
  for line in fp:
   v = [int(x) for x in line[1:-2].split(',')]
   arraylist_forme.append(v)
 print (len(arraylist_forme))

 arraylist_forme_norm = []
 with open(Path('output/norm_forme.txt'), 'r') as fp:
  for line in fp:
   v = [float(x) for x in line[1:-2].split(',')]
   arraylist_forme_norm.append(v)
 print (len(arraylist_forme_norm))

 arraylist_forme_tf_idf = []
 with open(Path('output/tf_idf_forme.txt'), 'r') as fp:
  for line in fp:
   v = [float(x) for x in line[1:-2].split(',')]
   arraylist_forme_tf_idf.append(v)
 print (len(arraylist_forme_tf_idf))

 age_class = []
 with open(Path('output/classi_eta.txt'), 'r') as fp:
   line = next(fp)
   age_class = [int(x) for x in line[1:-2].split(',')]

 print (len(age_class))

 # 'Learning the parameters of a prediction function and testing it on the same data is a methodological mistake'
 # Faccio lo split del mio dataset (train/test) 40% per il test
 # X_train, X_test, y_train, y_test = train_test_split(arraylist_forme_norm, age_class, test_size=0.4, random_state=0)
 # clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
 # print ("Accuratezza new: {:0.2f}".format(clf.score(X_test, y_test)))

 #print ("Fitting..")
 # SVC(kernel="linear".. o LinearSVC ???
 # LinearSVC: 'similar to SVC with parameter kernel=’linear’ but implemented in terms of liblinear rather than libsvm'
 # so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
 
 #svc = SVC(kernel="linear", random_state=0)
 #svc = LinearSVC()

 # svc.fit(arraylist_forme_norm, age_class) per adesso non faccio il fit che cmq viene fatto nel cross validation

 print ("Scaling...")
 # Scaling delle feature nell’intervallo [0, 1]
 minMaxScaler = preprocessing.MinMaxScaler()
 xTrain = minMaxScaler.fit_transform(arraylist_forme_norm)

 #print ("Cross validation...")
 # https://en.wikipedia.org/wiki/Cross-validation_(statistics)
 # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
 #scores = cross_val_score(svc, xTrain, age_class, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy")
 #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
 
 # Random Forest
 # Crea e addestra il modello
 #forest = ExtraTreesClassifier(n_estimators=1000, random_state=0)
 #forest.fit(arraylist_forme_norm, age_class)

 # Calcola l'accuratezza del modello
 #scores = cross_val_score(forest, arraylist_forme_norm, age_class, cv=StratifiedKFold(5, shuffle=True, random_state=0),scoring="accuracy")
 #print ("Accuratezza: %0.2f" % scores.mean())

 models = [LinearSVC(), SVC(kernel="linear", random_state=0),ExtraTreesClassifier(n_estimators=1000, random_state=0)]
 names = ['SVM liblinear', 'SVM libsvm', 'Decision Tree']
 
 for model, name in zip(models, names):
  print (name)
  scoring = ['accuracy'] # qui potrei aggiungere criteri diversi di scoring 
  
  scores  = cross_validate(model, xTrain, age_class,scoring=scoring, cv=StratifiedKFold(5, shuffle=True, random_state=0),return_train_score=True)
  accuracy = scores['test_accuracy'].mean()
  fit_time = sum([float(x) for x in scores['fit_time']])
  score_time = sum([float(x) for x in scores['score_time']])
  print ('Accuracy: {:0.2f}, Fit time: {:0.2f}, Score time: {:0.2f}'.format(accuracy, fit_time, score_time))

main()