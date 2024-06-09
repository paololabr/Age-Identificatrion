
import sys
import numpy
import math
import os
import re

from pathlib import Path

numpy.set_printoptions(threshold=sys.maxsize) # evita i troncamenti (i puntini) degli array numpy

def main(file1):
 docsnumwords = []   # numero di parole dei documenti
 wordlist_forme = [] # lista finale delle forme
 wordlist_lemmi = [] # lista finale dei lemmi
 age_class = []
 gender_class = []
 
 try:
  os.mkdir('output')
 except Exception:
  pass

 with open(file1, encoding='utf-8') as fp:
  next(fp)
  print ('-Prima scansione-')
  # dizionario che conta il numero dei documenti in cui una singola forma e' contenuta
  dict_forme = {}
  # dizionario che conta il numero dei documenti in cui un singola lemma e' contenuto
  dict_lemmi = {}
  
  numword = 0
  for line in fp:
   ln = line.split('\t')   
   if line.startswith('<doc'):
    curdc_forme = []
    curdc_lemmi = []    

    dig = re.findall(r'\btask_age="(\d+)"', line)
    if (len(dig) > 0):
     age_class.append(int(int(dig[0])/10))
    else:
     age_class.append(30) # errore

    gend = re.findall(r'\btask_gender="(\w+)"', line)    
    if (len(gend) > 0):
     if (gend[0].lower() == 'female'):
      gender_class.append(1)      
     else:
      gender_class.append(0) # errore
    else:
      gender_class.append(0) # errore
          
    numword = 0
   elif line.startswith('</doc'):
    docsnumwords.append(numword)
   elif (len(ln) > 1):
     numword = numword + 1
	 
     # forma
     wd_forma = ln[1].lower()
     if wd_forma not in curdc_forme:
      curdc_forme.append(wd_forma)
      if wd_forma not in dict_forme:
       dict_forme[wd_forma] = 1
      else:
       dict_forme[wd_forma] = dict_forme[wd_forma] + 1

     # lemma
     wd_lemma = ln[2].lower()
     if wd_lemma not in curdc_lemmi:
      curdc_lemmi.append(wd_lemma)
      if wd_lemma not in dict_lemmi:
       dict_lemmi[wd_lemma] = 1
      else:
       dict_lemmi[wd_lemma] = dict_lemmi[wd_lemma] + 1

  dict_forme = {key: val for key, val in dict_forme.items() if val > 5}
  wordlist_forme = (sorted(dict_forme))
  with open(Path('output/lista_forme.txt'), 'w') as f:
   for w in wordlist_forme:
    print('{:s}'.format(w), file=f)
  
  dict_lemmi = {key: val for key, val in dict_lemmi.items() if val > 5}
  wordlist_lemmi = (sorted(dict_lemmi))
  with open(Path('output/lista_lemmi.txt'), 'w') as f:
   for w in wordlist_lemmi:
    print('{:s}'.format(w), file=f)

 with open(Path('output/classi_eta.txt'), 'w') as lte:
  print(age_class, file=lte)

 with open(Path('output/classi_gender.txt'), 'w') as lte:
  print(gender_class, file=lte)

 print('Tot. documenti: {:d}'.format(len(docsnumwords)))
 print('Tot. forme: {:d}'.format(len(wordlist_forme)))
 print('Tot. lemmi: {:d}'.format(len(wordlist_lemmi)))

 # seconda scansione
 arraylist_forme = []
 arraylist_lemmi = []
 
 # calcolo array frequenza forme e lemmi
 with open(file1, encoding='utf-8') as fp1, open(Path('output/freq_forme.txt'), 'w') as fform, open(Path('output/freq_lemmi.txt'), 'w') as flemm :
  next(fp1)  
  print ('-Seconda scansione-')
  curdc_forme = [0] * len(wordlist_forme)
  curdc_lemmi = [0] * len(wordlist_lemmi)
  docounter = 0
  for line in fp1:
   ln = line.split('\t')
   if line.startswith('<doc'):
    curdc_forme = [0] * len(wordlist_forme)
    curdc_lemmi = [0] * len(wordlist_lemmi)
   elif line.startswith('</doc'):
    arraylist_forme.append(curdc_forme)
    arraylist_lemmi.append(curdc_lemmi)
    print(curdc_forme, file=fform)
    print(curdc_lemmi, file=flemm)
    docounter = docounter + 1
   elif (len(ln) > 1):
	 # forma
     wd_forma = ln[1].lower()
     if (wd_forma in wordlist_forme):
      idx = wordlist_forme.index(wd_forma)
      curdc_forme[idx] = curdc_forme[idx] + 1
	 # lemma
     wd_lemma = ln[2].lower()
     if (wd_lemma in wordlist_lemmi):
      idx = wordlist_lemmi.index(wd_lemma)
      curdc_lemmi[idx] = curdc_lemmi[idx] + 1

 arraylist_forme_norm = []
 arraylist_lemmi_norm = []
 
 # frequenza normalizzata forme
 with open(Path('output/norm_forme.txt'), 'w') as fn:
  f = 0
  for arf in arraylist_forme:
   ar = numpy.array(arf)
   if docsnumwords[f] > 0:
    arraylist_forme_norm.append(ar / docsnumwords[f])
    print((ar / docsnumwords[f]).tolist(), file=fn)
   else:
    arraylist_forme_norm.append(ar)
    print(ar.tolist(), file=fn)
   f = f + 1

 # frequenza normalizzata lemmi
 with open(Path('output/norm_lemmi.txt'), 'w') as ln:
  l = 0
  for arl in arraylist_lemmi:
   ar = numpy.array(arl)
   if docsnumwords[l] > 0:
    arraylist_lemmi_norm.append(ar / docsnumwords[l])
    print((ar / docsnumwords[l]).tolist(), file=ln)
   else:
    arraylist_lemmi_norm.append(ar)
    print(ar.tolist(), file=ln)
   l = l + 1

 arraylist_forme_tf_idf = []
 arraylist_lemmi_tf_idf = []

 # tf idf forme
 with open(Path('output/tf_idf_forme.txt'), 'w') as ltf:
  for idx in range(len(arraylist_forme)):
    tf_idf = []
    for x in range(len(arraylist_forme[idx])):
     tf = arraylist_forme[idx][x] / docsnumwords[idx]
     wd = wordlist_forme[x]
     idf = math.log(len(docsnumwords) / dict_forme[wd], 10)
     tf_idf.append(tf * idf)
    arraylist_forme_tf_idf.append(tf_idf)
    print(tf_idf, file=ltf)

 # tf idf lemmi
 with open(Path('output/tf_idf_lemmi.txt'), 'w') as ltf:
  for idx in range(len(arraylist_lemmi)):
   tf_idf = []
   for x in range(len(arraylist_lemmi[idx])):
    tf = arraylist_lemmi[idx][x] / docsnumwords[idx]
    wd = wordlist_lemmi[x]
    idf = math.log(len(docsnumwords) / dict_lemmi[wd], 10)
    tf_idf.append(tf * idf)
   arraylist_lemmi_tf_idf.append(tf_idf)
   print(tf_idf, file=ltf)

 check = (len(age_class) == len(docsnumwords) == len(arraylist_forme) == len(arraylist_lemmi) == len(arraylist_forme_norm) == len(arraylist_lemmi_norm) == len(arraylist_forme_tf_idf) == len(arraylist_lemmi_tf_idf))
 print ('check: {}'.format(check)) 
 
main(sys.argv[1])
