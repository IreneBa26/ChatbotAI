import json
import sqlite3

#import requests
import pickle
import string

from nltk.corpus import stopwords
from nltk.stem.snowball import ItalianStemmer
from nltk import word_tokenize

import speech_recognition as sr
import pyttsx3

import tensorflow as tf
import numpy as np

from tflearn import input_data, fully_connected, regression, DNN

d = pickle.load(open("corpus.p", "rb"))
temi = d['temi']
classi = d['classi']
documenti = d['documenti']

database = "bot.db"
conn = sqlite3.connect(database)
cursor = conn.cursor()

stemmer = ItalianStemmer()
stop = set(stopwords.words('italian'))

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

def BotANN():

	tf.reset_default_graph()

	rete = input_data(shape=[None, len(temi)])
	rete = fully_connected(rete, 8)
	rete = fully_connected(rete, 8)
	rete = fully_connected(rete, len(classi), activation='softmax')
	rete = regression(rete)

	model = DNN(rete, tensorboard_dir='logs')
	return model

modello = BotANN()
modello.load("./rete")

error_threshold = 0.25

def classifica (modello, array):
	# genera le probabilità
	prob = modello.predict([array])[0]
	#filtro quelle che superano la soglia
	risultati = [
		[i,p] for i,p in enumerate(prob)
		if p > error_threshold
	]

	# ordino per le classi più probabili
	risultati.sort(key=lambda x: x[1], reverse=True)
	lista_classi = []
	for r in risultati:
		lista_classi.append((list(classi)[r[0]], r[1]))
	return lista_classi


def elabora_risposta(frase, utente="utente_prova"):
	if frase is "Nullo" :
		return "Credo di non aver capito quello che hai detto."
	temi_frase = genera_temi(frase)
	print(temi_frase)
	X = genera_input(temi_frase)
	#print(X)
	classi_predette = classifica(modello, X)
	print(classi_predette)
	if classi_predette is not None:
		
		classi_predette = [c[0] for c in classi_predette]

		#print(classi_predette)
		if classi_predette:

			q = """
				SELECT risposta
				FROM risposte
				INNER JOIN classi ON (risposte.id_classe = classi.id)
				WHERE classe = '{0}'
			""".format (classi_predette[0])

			risposte = [r[0] for r in cursor.execute(q).fetchall()]

			risposta = np.random.choice(risposte)

			return risposta

engine = pyttsx3.init()
voice = pyttsx3.voice
engine.setProperty('voice', 'it-IT')
engine.say("Ciao sono il chatbot di supporto per l'emergenza Covid-19")
engine.runAndWait()

statoConversazione = 0

testoDomanda = "si"
testo = "Nullo"

while statoConversazione == 0 :
	
	engine = pyttsx3.init()
	voice = pyttsx3.voice
	engine.setProperty('voice', 'it-IT')
	engine.say("Esponimi il tuo problema!")
	engine.runAndWait()

	r = sr.Recognizer()

	with sr.Microphone() as source:
		print("Sono in ascolto");
		audio = r.listen(source)
		print("Elaborazione messaggio vocale")

	try:
		testo=r.recognize_google(audio, language = 'it-IT')
		print("TEXT: "+testo);
	except:
		pass;

	domanda = testo
	testo = "Nullo"
	print("DOMANDA: " + domanda)
	risposta=elabora_risposta(domanda)
	domanda = None

	if risposta is None:

		risposta = "Non riusciamo a gestire questo tipo di domanda, riprova con un'altra."

	engine = pyttsx3.init()
	voice = pyttsx3.voice
	engine.setProperty('voice', 'it-IT')
	engine.say(risposta)
	print(risposta)
	engine.runAndWait()

	risposta = None

	engine = pyttsx3.init()
	voice = pyttsx3.voice
	engine.setProperty('voice', 'it-IT')
	engine.say("Hai altri problemi da esporre? Se ho chiarito i tuoi dubbi basta rispondermi con 'NO'")
	engine.runAndWait()

	with sr.Microphone() as source:
		print("Sono in ascolto");
		audio = r.listen(source)
		print("Elaborazione messaggio vocale")

	try:
		testoDomanda = r.recognize_google(audio, language = 'it-IT')
		print("TEXT: "+ testoDomanda);
	except:
		#engine = pyttsx3.init()
		#voice = pyttsx3.voice
		#engine.setProperty('voice', 'it-IT')
		#engine.say("Esponimi il tuo problema!")
		#engine.runAndWait()
		pass;

	if testoDomanda == 'no':
		statoConversazione = 1
		engine = pyttsx3.init()
		voice = pyttsx3.voice
		engine.setProperty('voice', 'it-IT')
		engine.say("Alla prossima")
		engine.runAndWait()

