import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle5 as pickle
import re
import requests
import json
# from konlpy.tag import Kkma
# kkma = Kkma()

def load_topk_tf_idx(version = "bm25_kkma"):    
    file_path = {"bm25_kkma": "./data/topk_docs/top_k_idx_answer_bm25_kkma.pickle",
                "bm25_hannanum" : "top_k_idx_bm25_han.pickle",
                "tfidf_kkma_3000":"top_k_idx_answer_tfidf_kkma_3000.pickle",
                "tfidf_kkma_5000":"top_k_idx_answer_tfidf_kkma_5000.pickle",
                 }

    # load top_k_index
    with open(file_path[version], 'rb') as f:
        top_k_idx = pickle.load(f)
    print(f"---'top_k_idx' {version}] pickle download -> shape : ({len(top_k_idx)} , {len(top_k_idx[0])})") #(8514, 100)
    return top_k_idx

def get_dataset():
    klue_path = "./data/klue_mrc/parsed_klue_mrc_train.csv"
    klue_df = pd.read_csv(klue_path, index_col=0)
    klue_content = klue_df['content'].to_list()
    klue_query = klue_df['question'].to_list()
    klue_answer = klue_df['answer'].to_list()
    return klue_df, klue_content, klue_query, klue_answer

def request_answer(result_documents,query):
    request_text = []
    question = [{"question": query}]

    for i, document in enumerate(result_documents) :
        request_text.append({'context': document, 'questionInfoList': question}) 

    URL = 'http://211.39.140.116:8080/mrc'
    headers={
            'Content-type':'application/json', 
            'Accept':'application/json'}
    
    response = requests.post(URL + '/predict/documents', data=json.dumps(request_text), headers=headers)
    best_answers = []
    counts = 0

    for i in range(len(result_documents)):
        try:
            answers_from_doc = response.json()[i]['answers']
            answer_scores = sorted([[i['text'], i['score']] for i in answers_from_doc if len(i['text'])!=1], key = lambda x: x[1], reverse =True)
            best_answer, best_score = answer_scores[0][0],answer_scores[0][1]
            best_answers.append([best_answer, best_score])
        except:
            pass    

    best_answers = sorted(best_answers, key = lambda x:x[1], reverse= True)

    try: 
        answer = best_answers[0][0]
    except:
        answer = ""    
    return answer

    # model output preprocessing
def get_final_token(answer_string):
    p = re.compile('[0-9a-z가-힣A-Z]+')
    final_token = []
    # result = set(p.findall(answer_string)) | set(kkma.nouns(answer_string))
    result = set(p.findall(answer_string))

    return [i for i in list(result) if len(i)!=1]


def get_f1_score(klue_query, top_k_idx, klue_df,klue_answer, top_k=2): 
    precision, recall, f1_score = [],[],[]
    count = 0
    for idx in tqdm(range(len(klue_query))):
        count+=1
    # for idx in [1856,0]:
        query_ = klue_query[idx]
        
        candidate_doc_idx = top_k_idx[idx][:top_k]
        # print(candidate_doc_idx)
        candidate_doc = klue_df.loc[candidate_doc_idx]['content'].to_list()
        # print(candidate_doc)
        
        prediction = request_answer(candidate_doc, query_)
        # print(f" idx: {idx} prediction {answer}")
        prediction = get_final_token(prediction)
        # print(f" idx: {idx} tokenized prediction {answer}")
        # print(f" idx: {idx} answer {klue_answer[idx]}")
        # ground_truth = klue_groudtruth_answer_token[idx]
        groud_truth = get_final_token(klue_answer[idx])
        # print(f" idx: {idx} tokenized answer", groud_truth)
        
        # print(answer)
        

        try:
            p = len(set(prediction) & set(groud_truth))/len(prediction) # precision
        except ZeroDivisionError:
            p = 0
        
        try:
            r = len(set(prediction) & set(groud_truth))/len(groud_truth) #recall
        except ZeroDivisionError:
            r = 0

        try:
            f1 = 2*p*r/(p+r)
        except ZeroDivisionError:
            f1=0

        precision.append(p)
        recall.append(r)
        f1_score.append(f1)
    return precision, recall, f1_score

        


if __name__ == "__main__":
    '''
    version : "bm25_kkma", "bm25_hannanum", "tfidf_kkma_3000", "tfidf_kkma_5000"
    '''
    top_k_idx = load_topk_tf_idx(version = "tfidf_kkma_5000")
    klue_df, klue_content, klue_query, klue_answer = get_dataset()
    precision, recall, f1_score = get_f1_score(klue_query, top_k_idx, klue_df,klue_answer, top_k=1)
    
    # # 정답으로 top-1 accuracy 구하고 싶을 때!
    # answer_idx = [[i]  for i in range(len(klue_df))]
    # precision, recall, f1_score = get_f1_score(klue_query, answer_idx, klue_df, klue_answer, top_k=1)

    print(f"total score counts = {len(precision)}")
    print(f"precision : {sum(precision)/len(precision)} ")
    print(f"recall : {sum(recall)/len(recall)} ")
    print(f"f1_score : {sum(f1_score)/len(f1_score)} ")

