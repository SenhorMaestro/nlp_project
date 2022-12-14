from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
import nltk
import logging
import time
from string import punctuation
import re
import streamlit as st
import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

stop_words = set(stopwords.words('english'))
vocab_list = pd.read_csv('vocab_list.csv')
vocab_to_int = vocab_list.set_index('Unnamed: 0').to_dict()['value']


class sentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """
    
    def __init__(self, 
                vocab_size, 
                output_size, 
                embedding_dim, 
                hidden_dim,
                n_layers, 
                drop_prob=0.5):
        
        super().__init__()
        
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            n_layers, 
                            dropout=drop_prob, 
                            batch_first=True)
        
        self.dropout = nn.Dropout()
        
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):

        batch_size = x.size(0)
        
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.fc(lstm_out)
        out = self.dropout(out)

        sig_out = self.sigmoid(out)
        
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] 
        
        return sig_out, hidden
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.n_layers,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.n_layers,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden

def cleaning(text:str):
    lemmatizer = WordNetLemmatizer()
    sw = stopwords.words('english')

    text = text.lower()
    text = text.translate(str.maketrans('', '', punctuation))
    text = re.sub(r'\d+', ' ', text)
    lemmatized_text = []
    lemmatized_text.append(' '.join([lemmatizer.lemmatize(word) for word in text.split()]))
    reg_tokenizer = RegexpTokenizer('\w+')
    tokenized_text = reg_tokenizer.tokenize_sents(lemmatized_text)
    sw = stopwords.words('english')

    clean_tokenized_comments = [] 
    for i, element in enumerate(tokenized_text):
        clean_tokenized_comments.append(' '.join([word for word in element if word not in sw]))

    return clean_tokenized_comments

def data_preprocessing(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text) 
    text = ''.join([c for c in text if c not in punctuation])
    text = [word for word in text.split() if word not in stop_words]
    text = [word for word in text if word in vocab_list['Unnamed: 0'].tolist()]
    text = ' '.join(text)
    return text

def padding(review_int, seq_len):
    if len(review_int) <= seq_len:
        zeros = list(np.zeros(seq_len - len(review_int)))
        features = zeros + review_int
    else:
         features = np.array(review_int[len(review_int)-seq_len :])
    return torch.Tensor(features).unsqueeze(0)


st.header('???????????? ???????????????????? ???????????????????? ???? ?????? ??????????')
st.write('?????????????????? ???????????? ?????? ???????????????????????? ????????, ???????????????????? ???? ?????? ??????????')

review = st.text_area(label='?????????????? ???????? ?????? ?????????? ???? ?????????? ???? ???????????????????? ??????????', value = 'Awful movie!!!')

choice = st.radio('???????????????? ????????????', options=['???????????????? ??????????????????', 'LSTM'])

button_pressed = st.button ('????????????')
if button_pressed:
    if choice == '???????????????? ??????????????????':
        loaded_lrc_model = load('lrc_model.joblib')
        loaded_cvec = load('cvec_model.joblib')
        cleaned_text = cleaning(review)
    
        cvec_representation = loaded_cvec.transform(cleaned_text)
        prediction = loaded_lrc_model.predict(cvec_representation)
        prediction_str = str(prediction[0])
        predictionary = {'0':'???? ????????????????????', '1':'????????????????????'}
        st.title(prediction_str.translate(prediction_str.maketrans(predictionary)))
    elif choice == 'LSTM':
        review = data_preprocessing(review)
        review_int = [vocab_to_int[word] for word in review.split()]
        features = padding(review_int, seq_len = 400)
        vocab_size = 222609 + 1
        output_size = 1
        embedding_dim = 128
        hidden_dim = 64
        n_layers = 4
        lstm_model = sentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
        lstm_model.load_state_dict(torch.load('lstm_model_weights.pt', map_location=torch.device('cpu')))
        lstm_model.eval()
        h = lstm_model.init_hidden(1)
        out, h = lstm_model(torch.Tensor(features).long(), h)

        answer_dict = {1: "'???????????????????? ??????????'", 0: "'???????????????????? ??????????'"}
    
        sentiment2 = answer_dict[round(out[0].item())]

        st.title(sentiment2)
    
