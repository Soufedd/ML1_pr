import keras
import pandas as pd
import numpy as np
from string import ascii_lowercase as alphabet
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json




#Load data 
def loadData(dir):
	data = pd.DataFrame(pd.read_csv(str(dir) + '.csv')).as_matrix()
	print("Data Loaded")
	return(data)

def setupDictionary():
	OHs = to_categorical(np.arange(len(alphabet)))
	dictionary = {c:OH for c,OH in zip(alphabet,OHs)}
	rev_dictionary = {i:c for i,c in enumerate(alphabet)}
	return dictionary, rev_dictionary

def dataToWords(data):
	wordsY = []
	wordsX = []
	wordY = ''
	wordX = []
	ix = 0
	max_ix = data.shape[0]
	while ix < max_ix:
		if data[ix,2] == -1:
			wordY+=data[ix,1]
			wordX.append(data[ix,4:].reshape(16,8,1))
			wordsY.append(wordY)
			wordsX.append(np.asarray(wordX))
			wordY = ''
			wordX = []
		else:
			wordY+=data[ix,1]
			wordX.append(data[ix,4:].reshape(16,8,1))
		ix+=1
	wordsX = np.asarray(wordsX)
	wordsY = np.array(wordsY)
	return wordsX,wordsY


def seqData(seq_length,wordsX, wordsY, dictionary, test = False):
	#max_len = np.max([len(word) for word in wordsY])
	dataX = []
	dataY = []
	for word_idx in range(wordsX.shape[0]):
		for i in range(0, wordsX[word_idx].shape[0]- seq_length +1, 1):
			seq_in = wordsX[word_idx][i:i + seq_length]
			seq_out = wordsY[word_idx][i:i + seq_length]
			dataX.append(seq_in)
			dataY.append(seq_out)
	dataX = np.asarray(dataX)
	X_seq = pad_sequences(dataX, maxlen = seq_length, dtype = 'float32')
	Y_seq = np.array([np.array([dictionary[c] for c in w]) for w in dataY])
	Y_seq = pad_sequences(Y_seq, maxlen = seq_length, dtype = 'float32')
	Y_seq = np.reshape(Y_seq, (len(Y_seq), seq_length,Y_seq.shape[2]))
	X_seq = np.reshape(X_seq, (len(X_seq), seq_length,X_seq.shape[2],X_seq.shape[3],X_seq.shape[4]))
	return X_seq,Y_seq

def len_indexes(words, length):
	len_index = []
	for i in range(len(words)):
		if len(words[i]) == length:
			len_index.append(i)
	return np.array(len_index)

def save_model(model, Net, version = 1, l = 0):
	# serialize model to JSON
	model_json = model.to_json()
	with open("model_"+ Net +"v"+ str(version) + "_"+ str(l) + ".json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model_"+ Net +"v"+ str(version) + "_"+str(l)+".h5")
	print("Saved model to disk: " + "model_"+ Net +"v"+ str(version) + "_"+ str(l))

def load_model(Net, version = 0, l = 0):
	# load json and create model
	json_file = open("model_"+ Net +"v"+ str(version) + "_"+ str(l) + ".json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model_"+ Net +"v"+ str(version) + "_"+str(l)+".h5")
	print("Loaded model from disk: " + "model"+ Net +"v"+ str(version) + "_"+str(l))
	return loaded_model


