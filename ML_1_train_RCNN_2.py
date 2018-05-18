from HelperFunctions import *
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dropout
from keras.models import Sequential
from keras import regularizers


#Setup and preprocessing
dictionary, rev_dictionary = setupDictionary()
data = loadData('train')
wordsX, wordsY = dataToWords(data)
len_list = list(set([len(word) for word in wordsY]))

for l in len_list:
	len_index = len_indexes(wordsY, l)
	X_seq, Y_seq = seqData(l,wordsX[len_index], wordsY[len_index], dictionary) 



	#split dataset into train and val

	train_idx = np.random.choice(X_seq.shape[0], int(X_seq.shape[0]*0.9), replace = False)
	val_idx = np.setdiff1d(np.arange(X_seq.shape[0]), train_idx)

	X_train = X_seq[train_idx]
	Y_train = Y_seq[train_idx]

	X_val = X_seq[val_idx]
	Y_val = Y_seq[val_idx]

	model = Sequential()
	# define CNN model
	model.add(TimeDistributed(Conv2D(64, (3,3), strides = (1,1),activation = 'relu'), input_shape = (l,16,8,1)))
	model.add(TimeDistributed(MaxPooling2D(pool_size = (2,2), strides = (1,1))))
	model.add(TimeDistributed(Conv2D(32, (3,3), strides  = (2,2), activation = 'relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size = (2,2), strides = (1,1))))
	model.add(TimeDistributed(Flatten()))

	# define LSTM model
	model.add(Bidirectional(LSTM(64, return_sequences = True)))
	model.add(TimeDistributed(Dense(Y_seq.shape[2], activation = 'softmax')))
	print(model.summary())
	model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.RMSprop(lr = 0.01), metrics = ['accuracy'])
	model.fit(X_train,Y_train, epochs = 20, batch_size = 32, verbose = 2, validation_data = (X_val,Y_val))

	score = model.evaluate(X_val, Y_val, verbose = 0)
	print('Test loss: ', score[0])
	print('Test accuracy: ', score[1])
	save_model(model, 'RCNN',2,l)






