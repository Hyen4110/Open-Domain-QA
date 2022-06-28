import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Kkma
from rank_bm25 import BM25Okapi

def get_dataset():
    klue_path = "./data/klue_mrc/parsed_klue_mrc_train.csv"
    klue_df = pd.read_csv(klue_path, index_col=0)
    klue_content = klue_df['content'].to_list()
    klue_query = klue_df['question'].to_list()
    klue_answer = klue_df['answer'].to_list()

    return klue_df, klue_content, klue_query, klue_answer
    # return klue_df[:10], klue_content[:10], klue_query[:10], klue_answer[:10] # for test 


class MyTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        # 1) 토큰화 = 형태소
        # pos = self.tagger.pos(sent)
        # pos = ['{}/{}'.format(word,tag) for word, tag in pos]

        # 2) 토큰화 = 명사
        pos = self.tagger.nouns(sent)

        return pos


def get_embedding(pos_tagger, klue_content, klue_query, version = "tfidf", max_feature= 5000):
    #형태소 분석기 import
    
    if "tfidf" in version:
        my_tokenizer = MyTokenizer(pos_tagger)
        print("max_feature",max_feature)
        encoder = TfidfVectorizer(max_df = 0.8,
                                min_df = 3,
                                max_features = max_feature, # 5000 : 10minute
                                tokenizer = my_tokenizer )
        print(f"start making tfidf vector for content : {datetime.datetime.now()}")
        print("encoder", encoder)

        content_tfidf = encoder.fit_transform(klue_content).toarray()
        print(f"start making tfidf vector for query : {datetime.datetime.now()}")
        query_tfidf = encoder.transform(klue_query).toarray()

        return content_tfidf, query_tfidf, encoder

    elif "bm25" in version:
        tokenized_documents, tokenized_query = [], []
        for i in tqdm(range(len(klue_content))):
            tokenized_documents.append(pos_tagger.nouns(klue_content[i]))
            tokenized_query.append(pos_tagger.nouns(klue_query[i]))

        tokenized_documents, tokenized_query
        return tokenized_documents, tokenized_query
    

def save_embedding(encoder, embedding_content, embedding_query, version = "tfidf_kkma_000"):
    # save
    
    if "tfidf" in version:
        # 1) tfidfVectorizer vocabulary
        tfidfVectorizer_vocab = encoder.vocabulary_
        with open('tfidfVectorizer_vocab_'+version+'.pickle', 'wb') as f:
            pickle.dump(tfidfVectorizer_vocab, f, pickle.HIGHEST_PROTOCOL)
        print('tfidfVectorizer_vocab_'+version+'.pickle -> save finished!')
    
        # 2) tfidfVectorizer idf
        tfidfVectorizer_vocab_idf = encoder.idf_
        with open('tfidfVectorizer_vocab_idf'+version+'.pickle', 'wb') as f:
            pickle.dump(tfidfVectorizer_vocab_idf, f, pickle.HIGHEST_PROTOCOL)
        print('tfidfVectorizer_vocab_idf'+version+'.pickle -> save finished!')

    else:
        # 2) content_tfidf
        with open('content_emb_'+version+'.pickle', 'wb') as f:
            pickle.dump(embedding_content, f, pickle.HIGHEST_PROTOCOL)
        print('content_emb_+'+version+'.pickle -> save finished!')
            
        # 3) query_tfidf
        with open('query_emb_'+version+'.pickle', 'wb') as f:
            pickle.dump(embedding_query, f, pickle.HIGHEST_PROTOCOL)
        print('query_emb_'+version+'.pickle -> save finished!')
    
        

def get_topk_index(embedding_query, embedding_content, version = "tfidf"):
    if "tfidf" in version:
        cos = nn.CosineSimilarity()
        k_ = 100 # top-k '몇 건의 문서를 가져와서 확인할 것인가?'

        top_k_idx = []
        for i in tqdm(range(len(embedding_query))):
            res = torch.Tensor()
            
            j_ = 0
            for j in range(0, len(embedding_query), 100):
                query_tfidf_i = torch.tensor(embedding_query[i:i+1])
                content_tfidf_i  = torch.tensor(embedding_content[j_:j])

                output = cos(query_tfidf_i, content_tfidf_i)
                res = torch.cat((res, output), 0)

                j_ = j

            res_ = np.array(res)
            top_k_idx_ = list(np.argsort(res_)[::-1][:k_])

            top_k_idx.append(top_k_idx_)

        # save (top_k_idx)
        with open('top_k_idx_answer_'+version+'.pickle', 'wb') as f:
            pickle.dump(top_k_idx, f, pickle.HIGHEST_PROTOCOL)
        print('top_k_idx_answer_'+version+'.pickle -> save finished!')

    elif "bm25" in version:
        bm25 = BM25Okapi(embedding_content)

        top_k_idx = []
        for i in tqdm(embedding_query):
            doc_scores = bm25.get_scores(i)
            top_k_idx_ = np.argsort(doc_scores)[::-1][:100]
            top_k_idx.append(top_k_idx_)

        # save (top_k_idx)
        with open('top_k_idx_answer_bm25.pickle', 'wb') as f:
            pickle.dump(top_k_idx, f, pickle.HIGHEST_PROTOCOL)
        print('top_k_idx_answer_bm25.pickle -> save finished!')
    

    return top_k_idx


def get_topk_accracy_tfidf(top_k_idx, max_feature = 5000, topk=1):

    def get_top_k_score(k_for_score):
        for answer, candidate in enumerate(top_k_idx):
            if answer in candidate[:k_for_score]:
                top_k_accuracy.append(1)
            else:
                top_k_accuracy.append(0)
        return sum(top_k_accuracy)/len(top_k_accuracy) 


    top_k_accuracy = []
    top_k_acc = []
    for k_for_score in tqdm(range(1,101)):
        top_k_acc.append(get_top_k_score(k_for_score))

    # save (top_k_idx)
    with open('top_k_acc_tfidf_'+str(max_feature)+'.pickle', 'wb') as f:
        pickle.dump(top_k_acc, f, pickle.HIGHEST_PROTOCOL)
    print('top_k_acc_tfidf_'+str(max_feature)+'.pickle -> save finished!')
    print(f"top_k_acc: {top_k_acc}")

def get_topk_accracy_bm25(top_k_idx, topk=1):
    top_k_accuracy = []
    def get_top_k_score(k_for_score):
        for answer, candidate in tqdm(enumerate(top_k_idx)):
            if answer in candidate[:k_for_score]:
                top_k_accuracy.append(1)
            else:
                top_k_accuracy.append(0)

        return sum(top_k_accuracy)/len(top_k_accuracy) 

    top_k_acc = []
    for k_for_score in tqdm(range(1,101)):
        top_k_acc.append(get_top_k_score(k_for_score))
        
    print(f"top_k_acc: {top_k_acc}")


    # save (top_k_idx)
    with open('top_k_acc_bm25_.pickle', 'wb') as f:
        pickle.dump(top_k_acc, f, pickle.HIGHEST_PROTOCOL)


def tf_idf_retriever(pos_tagger, max_feature, topk=1, version = "tfidf"):
    # version -> (require)tfidf/bm25 , (option) kkma_5000 etc..../
    klue_df, klue_content, klue_query, klue_answer = get_dataset()
    print("data load finished.....")
    embedding_content, embedding_query, encoder = get_embedding(pos_tagger, klue_content, klue_query,version, max_feature)
    print("get embedding finished.....")

    ## save code / 
    # save_embedding(encoder, embedding_content, embedding_query, version = "tfidf_kkma_000")
    top_k_idx = get_topk_index(embedding_query,
                                embedding_content, 
                                version = version)
    get_topk_accracy_tfidf(top_k_idx, max_feature = max_feature, topk=1)


def bm25_retriever(pos_tagger,  topk=1, version = "bm25"):
    klue_df, klue_content, klue_query, klue_answer = get_dataset()
    tokenized_documents, tokenized_query = get_embedding(pos_tagger, 
                                                        klue_content, 
                                                        klue_query,
                                                        version = version,
                                                        max_feature= 0
                                                        )
    ## save code / 
    # save_embedding(encoder, embedding_content, embedding_query, version = "tfidf_kkma_000")
    top_k_idx = get_topk_index(tokenized_query,
                                tokenized_documents, 
                                version = version)

    get_topk_accracy_bm25(top_k_idx, topk=1)

if __name__ == "__main__":
    max_feature = 3000
    pos_tagger = Kkma() # Hannanum()
    
    # TF-IDF
    # print(f"tf idf retriever(max_feature: {max_feature}) started")
    # tf_idf_retriever(pos_tagger, max_feature, topk=1, version = "tfidf_kkma_5000")

    # BM25
    bm25_retriever(pos_tagger,  topk=1, version = "bm25_kkma")
