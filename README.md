# ChatbotAI
Covid19 Chatbot based on Artificial Intelligence and Natural Language Processing


### About the project:

  The projects implements an automatic chatbot in italian language that answers user's question related to Covid and the pandemic. 
  It is based on the use of a neural network and a system of sentences obtained from the main scientific communication institutions 
  (WHO and worldwide recognised Institutes of Virology and Infectious Diseases). 
  The application listens/waits for a question from the user. This vocal input will be processed by the underlying neural network
  which will provide, through NLP classification techniques, the most fitting answer. 

  Three main components:

  - **setup_db**: Database creation, from ".json" to ".db" extension

  - **setup_ann**: Artificial Neural Network setup, creates and trains the neural network

    1° method: processes the DB data and defines programme attributes
    2° method: creates network training sets - X and Y arrays from natural language to numeric
    3° method: definition and storage trained neural network (one input layer, two hidden layers and one output layer) for future reuse

  - **net_usage**: user interaction - classification of voice input

------------------------------------------------------------------------------------------------------

### Built with:

    Python (pyttsx3 - Numpy)
    TensorFlow (TFLearn)
    Natural Language Toolkit (NLTK - SpeechRecognition)
    Pyttsx3
    SQLite

------------------------------------------------------------------------------------------------------

### Project Folder:

    - data.json: Database origin file
    - net_usage.py: Activate user-chatbot interaction
    - setup_ann.py: Artificial Neural Network creation and training 
    - setup_db.py: Database creation

------------------------------------------------------------------------------------------------------

### Getting started:

   Prerequisites: Python, SQLite

------------------------------------------------------------------------------------------------------

### Installation:

1) Change directory in project folder and execute the following commands:
   
        pip install nltk
        pip install tensorflow
        
        - python -> import nltk nltk.download("stopwords") nltk.download("punkt")
        
        pip install tflearn
        pip install SpeechRecognition
        pip install pyttsx3

2) Download *'PyAudio-0.2.11-cp37-cp37m-win_amd64.whl'* in the project folder

3) Download *'PyAudio-0.2.11-cp36-cp36m-win_amd64.whl'* in the project folder

4) In project folder and execute the following commands: 
   
        pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl"
        pip install PyAudio-0.2.11-cp36-cp36m-win_amd64.whl"
        py setup_db.py" (Create 'bot.db')
        py setup_ann.py" 

     Generated following files:
     logs                        
     corpus.p                  (Database structure)
     rete.data-00000-of-00001  
     rete.index                 
     rete.meta   

5) Execute command *'modello.load("./rete")'* to load network in *'net_usage.py'*    

6) Execute command *'py net_usage.py'* to start the application

7) Interrogate the chatbot with a voice input
