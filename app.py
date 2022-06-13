import re
import random

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing, utils
from tensorflow.keras.models import load_model
import json

data = pd.read_excel("Data Chatbot generative untuk RNN.xlsx")

question = data.iloc[:,1]
answer = data.iloc[:,2]

question = [re.sub(r"\[\w+\]",'',q) for q in question]
#question = [" ".join(re.findall(r"\w+",q)) for q in question]
answer = [re.sub(r"\[\w+\]",'',a) for a in answer]
#answer = [" ".join(re.findall(r"\w+",a)) for a in answer]

answers_with_tags = list()
for i in range( len( answer ) ):
    if type( answer[i] ) == str:
        answers_with_tags.append( answer[i] )
    else:
        question.pop( i )

answers = list()
for i in range( len( answers_with_tags ) ) :
    answers.append( '<START> ' + answers_with_tags[i] + ' <END>' )

data_q_virtual = pd.DataFrame(question, columns = ["Questions"])
data_a_virtual = pd.DataFrame(answers, columns = ["Answers"])
data_list = data_q_virtual.join(data_a_virtual)


tokenizer = preprocessing.text.Tokenizer(filters='!"#$%&()*+,.:;<=>@\^`{|}~\t\n')
tokenizer.fit_on_texts( question + answers )
VOCAB_SIZE = len( tokenizer.word_index )+1
print( 'VOCAB SIZE : {}'.format( VOCAB_SIZE ))


#encoder_input_data
tokenized_questions = tokenizer.texts_to_sequences(question)
maxlen_questions = max( [len(x) for x in tokenized_questions ] )
padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions, maxlen = maxlen_questions, padding = 'post')
encoder_input_data = np.array(padded_questions)
print(encoder_input_data.shape, maxlen_questions)


#decoder_input_data
tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
decoder_input_data = np.array( padded_answers )
print( decoder_input_data.shape , maxlen_answers )


#decoder predicted-data
tokenized_answers = tokenizer.texts_to_sequences(answers)
for i in range(len(tokenized_answers)) :
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
onehot_answers = utils.to_categorical( padded_answers , VOCAB_SIZE )
decoder_output_data = np.array( onehot_answers )
print( decoder_output_data.shape )


# Build LSTM-RNN Encoder-Decoder Model
encoder_inputs = tf.keras.layers.Input(shape=( maxlen_questions , ))
encoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 256 , mask_zero=True ) (encoder_inputs)

encoder_outputs , state_h , state_c = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(256) , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=( maxlen_answers ,  ))
decoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 256 , mask_zero=True) (decoder_inputs)

decoder_rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(256), return_state=True, return_sequences=True)

decoder_outputs , _ , _ = decoder_rnn ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( VOCAB_SIZE , activation=tf.keras.activations.softmax ) 
output = decoder_dense ( decoder_outputs )


# Train and compile RNN model
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')


#model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=128, epochs=1, validation_split = 0.2) 
#model.save('asisten_virtual.h5')

model.load_weights("asisten_virtual.h5")


def make_inference_models():
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 256 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 256 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_rnn(
        decoder_embedding , initial_state=decoder_states_inputs)
    
    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model


def str_to_tokens(sentence):

    words = sentence.lower().split()
    tokens_list = list()
  
    for word in words:
        tokens_list.append( tokenizer.word_index[ word ] ) 
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')

enc_model , dec_model = make_inference_models()

def chatbot_response(msg):

    for _ in range(1000):
        question = msg

        if question in("Terima kasih", "Terima kasih Informasinya" , "Terima kasih Feedbacknya" , "thanks" , "ty" , "Thank You"):
            break
        else:
            try:
                states_values = enc_model.predict( str_to_tokens(question) )
                empty_target_seq = np.zeros( ( 1 , 1 ) )
                empty_target_seq[0, 0] = tokenizer.word_index['start']

                stop_condition = False
                decoded_translation = ''
                while not stop_condition :
                    dec_outputs , h , c = dec_model.predict( [empty_target_seq] + states_values )
                    sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
                    sampled_word = None
                    for word , index in tokenizer.word_index.items() :
                        if sampled_word_index == index :
                            decoded_translation += '{} '.format( word )
                            sampled_word = word
        
                    if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                      stop_condition = True
            
                    empty_target_seq = np.zeros( ( 1 , 1 ) )  
                    empty_target_seq[ 0 , 0 ] = sampled_word_index
                    states_values = [ h , c ] 

                
                return decoded_translation.replace("end","")
            except KeyError:
                return "Maaf saya tidak mengerti pertanyaan anda."
            except:
                return "Maaf saya tidak mengerti pertanyaan anda."



import sqlite3 as sql
from flask import Flask, render_template , request

app = Flask(__name__)

@app.route('/')
def index():
    connect_laptop = sql.connect('Dataset Laptop.db')
    print("Yay. Connection Complete")

    select_home_laptop = connect_laptop.execute("SELECT NamaLaptop, HargaLaptop, GambarLaptop FROM DatasetLaptop")

    row = select_home_laptop.fetchall()
    return render_template('Home.html' , rows = row)

"""
@app.route("/<search>")
def searching(search):
    connect_laptop = sql.connect('Dataset Laptop.db')

    search_laptop = connect_laptop.execute("SELECT NamaLaptop, HargaLaptop, GambarLaptop FROM DatasetLaptoptambahan WHERE instr(NamaLaptop, (?))" , (search,))

    row = search_laptop.fetchall()
    return render_template('Home.html', rows = row)
"""

@app.route("/ASUS")
def ASUS():
    connect_laptop_ASUS = sql.connect("Dataset laptop.db")
    print("Yay. Connection ASUS Complete")

    select_ASUS_laptop = connect_laptop_ASUS.execute("SELECT NamaLaptop, HargaLaptop, GambarLaptop, Merek FROM DatasetLaptop WHERE Merek = 'ASUS'")

    row = select_ASUS_laptop.fetchall()
    return render_template("Home.html" , rows = row)

@app.route("/MSI")
def MSI():
    connect_laptop_MSI = sql.connect("Dataset laptop.db")
    print("Yay. Connection MSI Complete")

    select_MSI_laptop = connect_laptop_MSI.execute("SELECT NamaLaptop, HargaLaptop, GambarLaptop, Merek FROM DatasetLaptop WHERE Merek = 'MSI'")

    row = select_MSI_laptop.fetchall()
    return render_template("Home.html" , rows = row)

@app.route("/Acer")
def Acer():
    connect_laptop_Acer = sql.connect("Dataset laptop.db")
    print("Yay. Connection Acer Complete")

    select_Acer_laptop = connect_laptop_Acer.execute("SELECT NamaLaptop, HargaLaptop, GambarLaptop, Merek FROM DatasetLaptop WHERE Merek = 'Acer'")

    row = select_Acer_laptop.fetchall()
    return render_template("Home.html" , rows = row)

@app.route("/HP")
def Hp():
    connect_laptop_Hp = sql.connect("Dataset laptop.db")
    print("Yay. Connection Hp Complete")

    select_Hp_laptop = connect_laptop_Hp.execute("SELECT NamaLaptop, HargaLaptop, GambarLaptop, Merek FROM DatasetLaptop WHERE Merek = 'HP'")

    row = select_Hp_laptop.fetchall()
    return render_template("Home.html" , rows = row)

@app.route("/Lenovo")
def Lenovo():
    connect_laptop_Lenovo = sql.connect("Dataset laptop.db")
    print("Yay. Connection Lenovo Complete")

    select_Lenovo_laptop = connect_laptop_Lenovo.execute("SELECT NamaLaptop, HargaLaptop, GambarLaptop, Merek FROM DatasetLaptop WHERE Merek = 'Lenovo'")

    row = select_Lenovo_laptop.fetchall()
    return render_template("Home.html" , rows = row)

@app.route("/<selectlaptop>")
def click_laptop(selectlaptop):
    connect_laptop_click = sql.connect("Dataset laptop.db")
    print("Welcome to laptop %s" % selectlaptop)

    selected_laptop = connect_laptop_click.execute("SELECT * FROM DatasetLaptop WHERE NamaLaptop = (?)" , (selectlaptop,))

    row2 = selected_laptop.fetchall()

    return render_template("selectedlaptop.html" , rows = row2)

@app.route("/Kiryubot")
def Kiryubot():
    return render_template("Virtual Assistant.html")

@app.route("/ExQuestion")
def exQuestion():
    return render_template("ExQuestion.html")

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg') 

    predict_answer = str(chatbot_response(userText))

    return predict_answer
  
if __name__ == "__main__":
    app.run()
