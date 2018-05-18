from HelperFunctions import *
import tensorflow as tf
import csv
from operator import itemgetter



dictionary, rev_dictionary = setupDictionary()
data = loadData('test')
wordsX, wordsY = dataToWords(data)
len_list = list(set([len(word) for word in wordsY]))


models = {}
indexes = {}
wordsY_predict = []


for l in len_list:
	models[l] = load_model('RCNN',2,l)
	len_index = len_indexes(wordsY, l)
	X_seq = seqData(l,wordsX[len_index], wordsY[len_index], dictionary)[0]
	words = models[l].predict(X_seq)
	preds = [[rev_dictionary[np.argmax(char)] for char in word] for word in words]
	wordsY_predict += list(zip(len_index,preds))


wordsY_sorted = sorted(wordsY_predict, key=itemgetter(0))
wordsY_predict_flat = []
for i in range(len(wordsY_sorted)):
	wordsY_predict_flat+= [char for char in wordsY_sorted[i][1]]
output = list(enumerate(wordsY_predict_flat))


with open('output_RCNN2_7.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Id','Prediction'])
    for row in output:
        csv_out.writerow(row)


