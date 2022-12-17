import numpy as np 
import pandas as pd
import torch
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
import textwrap
import string
import streamlit as st 
import re

pretrained_weights = 'sberbank-ai/rugpt3small_based_on_gpt2'
model_loaded = GPT2LMHeadModel.from_pretrained(pretrained_weights,     
output_attentions = False,
output_hidden_states = False)

model_loaded.load_state_dict(torch.load('model_goroscop_generation_weights.pt', map_location=torch.device('cpu')))

tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

st.header('Ваш гороскоп на сегодня')
st.write('Напишите свой знак зодиака или начало фразы и получите прогноз от лучших астрологов')

prompt = st.text_area(label='Введите текст сюда', value = 'Близнецам сегодня')
temperature = st.slider('Температура', min_value=0.5, max_value=10.0, value=2.0, format=None)
max_length = st.slider('Длина выдаваемого прогноза',min_value=100, max_value=500, value=100, format=None)
button_pressed = st.button ('генерировать')
if button_pressed:

    prompt = tokenizer.encode(prompt, return_tensors='pt')
    out = model_loaded.generate(
    input_ids=prompt,
    max_length=max_length,
    num_beams=5,
    do_sample=True,
    temperature=temperature,
    top_k=50,
    top_p=0.6,
    no_repeat_ngram_size=2,
    num_return_sequences=7,
    ).cpu().numpy()
    for out_ in out:
        answer = textwrap.fill(tokenizer.decode(out_) , 100)
    answer = answer.replace('"','')
    answer = answer.replace('Читать далее','')
    answer = re.sub(r'\d+', ' ', answer) # удаляем числа
    answer = re.sub(r'\n', ' ', answer) 
    answer = answer.replace(', - - ,','')
    answer1 = ''
    for i in range(len(answer)):
        if answer[i] not in string.ascii_letters:
            answer1 += answer[i]
    answer1 = re.sub(r"  ,", ' ', answer1) 
    answer1 = answer1[:answer1.rfind('.')+1]
    st.write(answer1)
