#import json
#import requests
import string
import re
#import nltk
import string
import itertools

#from nltk.corpus import stopwords
#from nltk.corpus import wordnet
#import traceback
#from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
#import lt_core_news_sm
#import pke
import spacy
import pandas as pd
import numpy as np
from spacy import displacy



#nlp = lt_core_news_sm.load()
try:
    nlp = spacy.load("lt_core_news_lg")
except:
    print("ERROR")

stop_word_list = [line.strip() for line in open("stopwords-lt.txt", 'r')]

data_test = {'key1': 'value2', 'key2':'value2'}


def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences



def get_noun_adj_verb(text):
    
    doc = nlp(text)
    #html = displacy.render(doc, style="dep", page=True)
    #html = "<div style='max-width:100%; max-height:360px; overflow:auto'>" + html + "</div>"

    out = []

    #for token in doc:
     #   pos_tokens.extend([(token.text, token.pos_), (" ", None)])
    for ent in doc.ents:
        #pos_tokens.extend([(ent.text, ent.label_), (" ", None)])
        out.append(ent.text)
    return out
    #doc = nlp(text_data2)



def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences

def get_fill_in_the_blanks(sentence_mapping):
    out={}
    blank_sentences = []
    processed = []
    keys=[]
    for key in sentence_mapping:
        if len(sentence_mapping[key])>0:
            sent = sentence_mapping[key][0]
            # Compile a regular expression pattern into a regular expression object, which can be used for matching and other methods
            insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
            no_of_replacements =  len(re.findall(re.escape(key),sent,re.IGNORECASE))
            line = insensitive_sent.sub(' _________ ', sent)
            #if (sentence_mapping[key][0] not in processed) and no_of_replacements<3:
            blank_sentences.append(line)
            processed.append(sentence_mapping[key][0])
            keys.append(key)
    out["sentences"]=blank_sentences
    out["keys"]=keys
    return out


def text_analysis(text):
        doc = nlp(text)
        html = displacy.render(doc, style="dep", page=True)
        #html = "<div style='max-width:100%; max-height:360px; overflow:auto'>" + html + "</div>"
        pos_count = {
            "char_count": len(text),
            "token_count": 0,
        }
        pos_tokens = []

        #for token in doc:
        #   pos_tokens.extend([(token.text, token.pos_), (" ", None)])
        for ent in doc.ents:
            pos_tokens.extend([(ent.text, ent.label_), (" ", None)])

        #  pos_tokens.extend([(ent.text, ent.label_), (" ", None)])
        
        return pos_tokens

def fill_blanks_klausimai(text):
    sentences = tokenize_sentences(text)
    noun_verbs_adj = get_noun_adj_verb(text)

    keyword_sentence_mapping_noun_verbs_adj = get_sentences_for_keyword(noun_verbs_adj, sentences)

    fill_in_the_blanks = get_fill_in_the_blanks(keyword_sentence_mapping_noun_verbs_adj)
    results = fill_in_the_blanks

    sakiniai = results['sentences']
    raktazodziai = results['keys']
    
    '''
        sakiniai_su_raktazodziasi = pd.DataFrame(
        {'Raktažodis': raktazodziai,
         'Klausimas': sakiniai
        })
    '''
    sakiniai_su_raktazodziasi = pd.DataFrame(
        {'Raktažodis': raktazodziai,
         'Klausimas': sakiniai
        })


    highlights = text_analysis(text)
    return highlights, sakiniai_su_raktazodziasi