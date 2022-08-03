import streamlit as st
import extra_streamlit_components as stx
import streamlit.components.v1 as components
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from streamlit_tags import st_tags
#import json
#import requests
#import string
#import re
#import nltk
#from nltk.corpus import stopwords
#from nltk.corpus import wordnet
#import traceback
#from nltk.tokenize import sent_tokenize
#from flashtext import KeywordProcessor
#import lt_core_news_lg
#import pke
import spacy
import pandas as pd
import numpy as np
#import spacy_streamlit
#from spacy_streamlit import visualize_ner
#import docx2txt
#import pdfplumber
#import time
#import importlib.util

import base64
from functionforDownloadButtons import download_button
import os
import json
from spacy import displacy

from fill_blanks import tokenize_sentences, get_noun_adj_verb, get_sentences_for_keyword, get_fill_in_the_blanks, text_analysis, fill_blanks_klausimai
from funkcijos import generate_questions, upload_model



st.set_page_config(layout="wide")




@st.cache(allow_output_mutation=True)
def load_model(model_name):
    nlp=spacy.load(model_name)
    return nlp
    
nlp = spacy.load("lt_core_news_lg")
stop_word_list = [line.strip() for line in open("stopwords-lt.txt", 'r')]


trained_model_path = 'model_versions/2021-11-21_5_epoch_squad_sciq/model/'
trained_tokenizer = 'model_versions/2021-11-21_5_epoch_squad_sciq/tokenizer/'


@st.cache(allow_output_mutation=True)
def upload_model(trained_model_path, trained_tokenizer):

    model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
    tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print ("device ",device)
    model = model.to(device)

    return model, tokenizer

model, tokenizer = upload_model(trained_model_path, trained_tokenizer)

default_text = """Galilėjas Galilėjus gimė 1564 m. vasario 15 d. Italijoje, Pizos mieste, čia mokėsi, įstojo į universitetą, tačiau nebaigė jo dėl
finansinių sunkumų. Vėliau, būdamas 25 metų,[2] pradėjo dėstyti matematiką Pizos universitete, šiek tiek
vėliau persikėlė į Paduvos universitetą, kur iki 1610 m. dėstė geometriją, mechaniką ir astronomiją. 1610 m.
aprašė savo atradimus knygoje „Žvaigždžių pasiuntinys“, kuri tapo viena iš skaitomiausių knygų Italijoje ir
kitose šalyse.[3] 1612 m. paskelbė darbą „Svarstymai apie kūnus, kurie yra vandenyje, ir apie tuos, kurie jame
plaukia“.[4] 1632 m. išleistoje knygoje „Dialogas apie dvi svarbiausias pasaulio sistemas – Ptolemajo ir
Koperniko“ pasisakė už Koperniko heliocentrinę sistemą, tuo užsitraukdamas Katalikų bažnyčios nemalonę. Už
scholastinės Aristotelio fizikos neigimą 1632 m. jam teko stoti prieš inkvizicijos teismą. Iš pradžių
priverstas tylėti, vėliau, grasinamas kankinimais, atsisakė savo pažiūrų. Nuo 1633 m. gyveno kaime namų arešto
sąlygomis prie Florencijos. 1637 m. apako. Palaidotas Florencijos Santa Croce bazilikoje. 1638 m. Olandijoje
buvo išleistas jo veikalas „Pokalbiai ir matematiniai įrodinėjimai apie dvi naujas mokslo šakas“, kuris padėjo
pamatus medžiagų mechanikos mokslui."""


context = ''


if "upload_keywords" not in st.session_state:
    st.session_state["upload_keywords"] = "upload"

if "answer_ent" not in st.session_state:
    st.session_state["answer_ent"] = []

# function to get unique values
def unique_values_in_list(list1):
    x = np.array(list1)
    unique_val = np.unique(x).tolist()
    
    return unique_val

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

answer_ent = []

def clear_text():
    st.session_state["form_1"] = ""


def main():


    st.title("Klausimų generavimas")

    

    #with st.form(key='form_1'):

    st.write("form-1")

    with st.expander("See explanation"):
        pdf_file = st.file_uploader("Upload PDFs")
        st.write(type(pdf_file))
        print(type(pdf_file)) 
        if  type(pdf_file) == "streamlit.uploaded_file_manager.UploadedFile":
            st.write(pdf_file.name)
        
    context  = st.text_area("Šiame lange nurodomas tekstas",key='form_1', value=default_text, height=300, max_chars=100000)

    col1, col2, col3 , col4, col5= st.columns(5)

    with col1:
        if st.button("Gauti rekomenduojamus raktažodžius"):
            st.session_state["answer_ent"] =list(set(get_noun_adj_verb(st.session_state['form_1'])))
    with col2:
        st.button("Ištrinti tekstą", on_click=clear_text)


    with st.form(key='form_2'):
 
        st.write("form-2")
        keywords = st_tags(
            label='Nurodykite raktažodžius',
            text='Įvedę norimą raktažodį, spauskite Enter',
            value=st.session_state["answer_ent"],
            maxtags = 100)

        

        generate = st.form_submit_button("Sukurti klausimus")

    if generate:
        #context = df3.iloc[-1]['Context']
        if len(keywords) == 0:
            st.warning("Būtina nurodyti bent vieną raktažodį")

        elif len(context) <= 100:
            #print(len(context))
            st.warning("Teksto ilgis turėtu būti bent 100 spaudos ženklų")
        else:
            st.session_state["questions"] = generate_questions(model, tokenizer, context, keywords)
        

    if "questions" in st.session_state:

        with st.form(key='form_3'):
            st.write("form-3")
            unique_questions = unique_values_in_list(st.session_state["questions"])


            check_boxes = [st.checkbox(question, key=question) for question in unique_questions]
            checked_questions = [question for question, checked in zip(unique_questions, check_boxes) if checked]
            download_button_submision = st.form_submit_button('Download')
            if download_button_submision:
               download_button(checked_questions, "Data.docx", "📥 Download (.docx)")
            
        @st.cache
        def convert_df(checked_questions):
            df_checked_questions = pd.DataFrame(checked_questions)
            return df_checked_questions.to_csv().encode('utf-8')

        csv = convert_df(checked_questions)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='large_df.csv',
            mime='text/csv',
            )
        #st.write(type(checked_questions))
        #st.download_button('Download some text', checked_questions)

if __name__ == "__main__":
    main()