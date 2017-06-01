
import os
import sys

import numpy as np
import spacy
import tqdm
import gensim
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

TRAIN_FILE = "quora/train.csv"
OUTPUT_DIR = "quora/data/"

def load_data():
    df = pd.read_csv(TRAIN_FILE)

    df['question1'] = df['question1'].apply(lambda x: str(x).decode('utf8'))
    df['question2'] = df['question2'].apply(lambda x: str(x).decode('utf8'))

    return df

def create_gensim_model():

    df = load_data()
    questions = list(df['question1'] + df['question2'])

    for c, question in tqdm.tqdm(enumerate(questions)):
        questions[c] = list(gensim.utils.tokenize(question, deacc=True, lower=True))

    print("Generating word2vec model")
    model = gensim.models.Word2Vec(questions, size=300, workers=16, iter=10, negative=20)
    model.init_sims(replace=True)

    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    print("No of tokens in word2vec: ", len(w2v.keys()))

    model.save(os.path.join(OUTPUT_DIR, 'word2vec.mdl'))
    model.wv.save_word2vec_format(os.path.join(OUTPUT_DIR, 'word2vec.bin'), binary=True)

def create_spacy_model():
    # spacy.load('en_core_web_md')

    df = load_data()
    nlp = spacy.load('en')

    vecs1 = [doc.vector for doc in nlp.pipe(df['question1'], n_threads=50)]
    vecs1 = np.array(vecs1)
    df['q1_feats'] = list(vecs1)

    vecs2 = [doc.vector for doc in nlp.pipe(df['question2'], n_threads=50)]
    vecs2 = np.array(vecs2)
    df['q2_feats'] = list(vecs2)

    pd.to_pickle(df, os.path.join(OUTPUT_DIR, 'spacy_df.pkl'))

def calc_tfidf(df, question_id, nlp, word2tfidf):

    vecs1 = []
    for qu in tqdm.tqdm(list(df[question_id])):
	doc = nlp(qu)
	mean_vec = np.zeros([len(doc), 300])
	for word in doc:
	    # word2vec
	    vec = word.vector
	    # fetch df score
	    try:
		idf = word2tfidf[str(word)]
	    except:
		#print word
		idf = 0
	    # compute final vec
	    mean_vec += vec * idf
	mean_vec = mean_vec.mean(axis=0)
	vecs1.append(mean_vec)

    return list(vecs1)

def create_spacy_tf_idf_model():

    df = load_data()
    #nlp = spacy.load('en')
    nlp = spacy.load('en_core_web_md')

    questions = list(df['question1']) + list(df['question2'])

    tfidf = TfidfVectorizer(lowercase=False, )
    tfidf.fit_transform(questions)

    # dict key:word and value:tf-idf score
    word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

    df['q1_feats'] = calc_tfidf(df, 'question1', nlp, word2tfidf)
    df['q2_feats'] = calc_tfidf(df, 'question2', nlp, word2tfidf)

    #pd.to_pickle(df, os.path.join(OUTPUT_DIR, 'spacy_en_tfidf.pkl'))
    pd.to_pickle(df, os.path.join(OUTPUT_DIR, 'spacy_enweb_tfidf.pkl'))

if __name__=="__main__":
    #create_gensim_model()
    #create_spacy_model()
    create_spacy_tf_idf_model()

