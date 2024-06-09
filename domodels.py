
from pathlib import Path
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

def getdata(filename):
 data_array = []
 with open(Path(filename), 'r') as fp:
  for line in fp:
   v = [float(x) for x in line[1:-2].split(',')]
   data_array.append(v)
 return data_array

def getclasses(filename):
 class_array = []
 with open(Path('output/classi_eta.txt'), 'r') as fp:
  line = next(fp)
  class_array = [int(x) for x in line[1:-2].split(',')]
 return class_array

def main():
 #arraylist_forme = getdata(Path('output/freq_forme.txt'))
 arraylist_forme_norm = getdata(Path('output/norm_forme.txt'))
 #arraylist_forme_tf_idf = getdata(Path('output/tf_idf_forme.txt'))
 age_class = getclasses(Path('output/classi_eta.txt'))

 print (len(arraylist_forme_norm))
 print (len(age_class))

 print ("Scaling...")
 # Scaling delle feature nell’intervallo [0, 1]
 minMaxScaler = preprocessing.MinMaxScaler()
 xTrain = minMaxScaler.fit_transform(arraylist_forme_norm)

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