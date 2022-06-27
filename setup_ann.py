from nltk.corpus import stopwords
from nltk.stem.snowball import ItalianStemmer
from nltk import word_tokenize
from tflearn import input_data, fully_connected, regression, DNN

import tensorflow as tf
import numpy as np
import random
import pickle

import string
import sqlite3

conn = sqlite3.connect("bot.db")
cursor = conn.cursor()

stemmer = ItalianStemmer()
stop = set(stopwords.words('italian'))

def elabora_corpus(corpus):
	""" Extracts the meaning of a sentence

	Parameters
	----------
	corpus: NLTK corpus

	Returns
	---------
	temi: set of words themes
	classi: set of classes
	documenti: list of documents (tuple of two elements: meaningful words and related classes)
	"""

	temi = set()
	classi = set()
	documenti = []

	stop = set(stopwords.words('italian'))  # Eliminate all the possible stopwords

	for frase, classe in corpus:
		parole = [
			p.replace("?" , "").lower() for p in word_tokenize(frase)
			if p not in stop
			and p not in string.punctuation
		]

		temi.update(parole)
		documenti.append((parole, classe))
		classi.add(classe)

	# Creating themes (temi)
	temi = list(set(stemmer.stem(parola) for parola in temi))
	classi = list(classi)
	return temi, classi, documenti


q= """
	SELECT domanda, classe
	FROM domande
	return JOIN classi ON (id_classe = classi.id)
"""

domande = cursor.execute(q).fetchall()
temi, classi, documenti = elabora_corpus(domande)

#print("Numeri di classi: {}".format(len(classi)))
#print("Numero di documenti: {}".format(len(documenti)))
#print("Temi: \n{}".format(temi))


#print("Temi = {}".format(temi))
#print("Classi = {}".format(classi))

#print("Parole Documento = {}".format(documenti[-1][0]))
#print("Classe Documento = {}".format(documenti[-1][1]))

#input = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ,0, 1, 1, 0, 0, 0, 0]
#output = [0, 0, 1]


def crea_training_set(documenti, classi):
	""" Performs training set creation and returns two-value tuple. Both arrays have fixed length

	Parameters
	----------
	documenti: List of words (parole) and classi (classes)
	classi: set of classes

	Returns
	---------
	train_x: array input
	train_y: array output
	"""

	training = []
	output_vuota = [0] * len(classi)
	classi = list(classi)

	for parole, classe in documenti:
		temi_frase = [stemmer.stem(parola) for parola in parole]

		#riempo la lista di input
		riga_input = [1 if t in temi_frase else 0 for t in temi]

		#riempo la lista di output
		riga_output = output_vuota[:]
		riga_output[classi.index(classe)] = 1

		training.append([riga_input, riga_output])

	# Shuffle the order of the items
	random.shuffle(training)
	# To array
	training = np.array(training)

	# Creating the training set
	train_x = list(training[:,0])
	train_y = list(training[:,1])
	return train_x, train_y

X, y = crea_training_set(documenti, classi)
#print("X = {}".format(X))
#print("X = {}".format(y))

def BotANN(x, y):
	""" This method creates and trains an ANN (Artifical Neural Network) of type DNN (Deep Neural Network)
	The DNN is made of one input level, two hidden layers, one output level.
	It is based on a feedforward algorithm, propagates input in the network using a backpropagation technique:
	Output value has range from 0 to 1

	Parameters
	----------
	x: Bidimensional array for data input
	y: Bidimensional array for data output

	Returns
	---------
	model: a DNN
	"""

	# Clear and reset graph
	tf.reset_default_graph()

	# Tuning the neural network
	rete = input_data(shape=[None, len(X[0])])
	rete = fully_connected(rete, 8)
	rete = fully_connected(rete, 8)
	rete = fully_connected(rete, len(y[0]), activation='softmax') # Activation function
	rete = regression(rete)

	#tnorm = tf.initializations.uniform(minval=-1.0, maxval=1.0)
    #rete = tf.input_data(shape=[None, 2], name='inputLayer')
    #rete = tf.fully_connected(rete, 2, activation='sigmoid', weights_init=tnorm, name='layer1')
    #rete = tf.fully_connected(rete, 1, activation='softmax', weights_init=tnorm, name='layer2')
    #regressor = tf.regression(rete, optimizer='sgd', learning_rate=2., loss='mean_square', name='layer3')

    # Execute training
	model = DNN(rete, tensorboard_dir='logs')
	model.fit(X, y, n_epoch=1000, batch_size=8, show_metric=True)
	return model

modello = BotANN(X, y)
modello.save("./rete")  # Final trained DNN saved in "rete.tfl"


def genera_temi(testo):
	stop = set(stopwords.words('italian'))
	lista_parole = word_tokenize(testo)
	temi = [
		stemmer.stem(p.lower()) for p in lista_parole
		if p not in stop and p not in string.punctuation
	]
	return temi

def genera_input(lista_temi):
	lista_input = [0]*len(temi)
	for tema in lista_temi:
		for i, t in enumerate(temi):
			if t == tema:
				lista_input[i] = 1
	return(np.array(lista_input))


#temi_frase = genera_temi("Ciao, mi chiamo Marco")
#X = genera_input(temi_frase)

#print(X)

error_threshold = 0.25

def classifica (modello, array):
	prob = modello.predict([array])[0]
	risultati = [
		[i,p] for i,p in enumerate(prob)
		if p > error_threshold
	]

	risultati.sort(key = lambda x: x[1], reverse=True)
	lista_classi = []
	for r in risultati:
		lista_classi.append((list(classi)[r[0]], r[1]))
	return lista_classi


def rispondi (modello, frase):
	temi_frase = genera_temi(frase)
	X= genera_input(temi_frase)
	classi_predette = classifica(modello, X)

	if classi_predette:
		q = """
			SELECT risposta
			FROM risposte
			INNER JOIN classi ON (risposte.id_classe = classi.id)
			WHERE classe = '{0}'
		""".format(classi_predette[0][0])
		risposte = [r[0] for r in cursor.execute(q).fetchall()]
		return np.random.choice(risposte)


# Closing database connection
conn.commit()
conn.close()

d= {
	'temi': temi,
	'classi': classi,
	'documenti': documenti
}

pickle.dump(d, open("corpus.p", "wb"))

