# coding = utf-8
import os
import pandas as pd
import numpy as np
# from preprocess import Preprocess

import re
import random
import tarfile
import urllib
from torchtext.legacy import data
from datetime import datetime
import pickle
from gensim.models import Word2Vec
import jieba
import traceback



'''
prefix: 
spark: 'SPARK'
eclipse: 'JDT'
map_reduce: 'MAP'
Mozilla_core: 'Core'
'''
def exchange_prefix(prefix):
    dict={'spark':'SPARK','eclipse':'JDT','mapreduce':'MAPREDUCE','core':'Core','firefox':'Firefox','thunderbird':'Thunderbird'}
    return dict[prefix]

def times_window(t1, t2):
    t1 = pd.to_datetime(t1)
    t2 = pd.to_datetime(t2)
    delta = t2 - t1 if t2 > t1 else t1 - t2
    if delta.days < 90:
        return 1
    else:
        return 0

def train_word2vec_model(df):
    '''
    w2v 模型训练句子获取向量
    '''
    corpus = []
    for i, r in df.iterrows():
        try:
            corpus.append(jieba.lcut(r['Title']))
            # print jieba.lcut(r['ques1'])
            corpus.append(jieba.lcut(r['Description']))
        except:
            pass
            # print('Exception: ', r['ques1']
    word2vec_model = Word2Vec(corpus, size=300, window=3, min_count=1, sg=0, iter=100)
    return word2vec_model


def load_data(prefix):
    #
    # df = pd.read_csv(open(data_path, 'rU'))
    data_path='./TSE/dataAll/'+prefix+'/'+prefix+'.csv'
    df = pd.read_csv(data_path, encoding = 'gb18030')
    # df['Title'] = df['Title'] + ' '
    # df['Title'] = df['Title'] + df['Description']
    df.dropna(subset=['Title'], inplace=True)
    
    df['Duplicate_null'] = df['Duplicated_issue'].apply(lambda x : pd.isnull(x))
    
    # prep = Preprocess()
    # df['Desc_list'] = df['Title'].apply(lambda x : prep.stem_and_stop_removal(x))
    
   
    df_data = df[df['Duplicate_null'] == False]


    df_field = df_data[['Issue_id', 'Title', 'Duplicated_issue', 'Resolution']]
    df_field['dup_list'] = df_field['Duplicated_issue'].apply(lambda x: x.split(';'))
    Dup_list = []
    for i,r in df_field.iterrows():
        for dup in r['dup_list']:
            # print(dup)
            if int(r['Issue_id'].split('-')[1]) < int(dup.split('-')[1]):
                if dup.startswith(exchange_prefix(prefix)):
                    Dup_list.append([r['Issue_id'], dup, r['Resolution']])
    df_pairs_pos = pd.DataFrame(Dup_list, columns = ['Issue_id_1', 'Issue_id_2', 'Resolution'])

    
    neg_dup_list = []
    cnt = 0
            
    for i,r in df.iterrows():
        if r['Duplicate_null'] == True:
            j = 1
            while i + j < len(df) and not df.iloc[i+j]['Issue_id'].startswith(exchange_prefix(prefix)):
                j += 1
            if i+j <len(df):
                neg_dup_list.append([r['Issue_id'], df.iloc[i+j]['Issue_id'], r['Resolution']])
                cnt += 1
                
        # if cnt > len(Dup_list):
        #     break

            

    df_pairs_neg = pd.DataFrame(neg_dup_list, columns = ['Issue_id_1', 'Issue_id_2', 'Resolution'])

    df_pairs_neg['Title_1'] = df_pairs_neg['Issue_id_1'].apply(lambda x: list(df[df['Issue_id'] == x]['Title'])[0])
    df_pairs_neg['Title_2'] = df_pairs_neg['Issue_id_2'].apply(lambda x: list(df[df['Issue_id'] == x]['Title'])[0])
    
    df_pairs_pos['Title_1'] = df_pairs_pos['Issue_id_1'].apply(lambda x: list(df[df['Issue_id'] == x]['Title'])[0])
    df_pairs_pos['Title_2'] = df_pairs_pos['Issue_id_2'].apply(lambda x: list(df[df['Issue_id'] == x]['Title'])[0] if len(list(df[df['Issue_id'] == x]['Title'])) > 0 else '')
    # df_pairs_pos['Title_2'] = df_pairs_pos['Issue_id_2'].apply(lambda x: list(df[df['Issue_id'] == x]['Title'])[0])
    
   
    '''
    df_pairs_pos = df_pairs_pos[['Title_1','Title_2']]  
    df_pairs_neg = df_pairs_neg[['Title_1','Title_2']]
     
    
    '''
    df_pairs_neg['Title_1'].apply(lambda x: str(' '.join(x)))
    df_pairs_neg['Title_2'].apply(lambda x: str(' '.join(x)))
    df_pairs_pos['Title_1'].apply(lambda x: str(' '.join(x)))
    df_pairs_pos['Title_2'].apply(lambda x: str(' '.join(x)))
    '''
    '''
    df_pairs_pos.insert(loc=len(df_pairs_pos.columns), column='label', value=1)
    df_pairs_neg.insert(loc=len(df_pairs_neg.columns), column='label', value=0)
    
    df_pairs_pos= df_pairs_pos.loc[:, ['Title_1', 'Title_2','label']]
    df_pairs_neg= df_pairs_neg.loc[:, ['Title_1', 'Title_2','label']]
    
    
    df_pairs_pos['word_count1'] = df_pairs_pos['Title_1'].str.len()
    df_pairs_pos['word_count2'] = df_pairs_pos['Title_2'].str.len()
    

    # 删除某列中包含的单词数小于 3 的行
    df_pairs_pos = df_pairs_pos[df_pairs_pos['word_count1'] > 1]
    df_pairs_pos = df_pairs_pos[df_pairs_pos['word_count2'] > 1]

    # 删除多余的 'word_count' 列
    df_pairs_pos = df_pairs_pos.drop('word_count1', axis=1)
    df_pairs_pos = df_pairs_pos.drop('word_count2', axis=1)

    df_pairs_pos.reset_index(drop=True, inplace=True)
    
    df_pairs_neg['word_count1'] = df_pairs_neg['Title_1'].str.split().str.len()
    df_pairs_neg['word_count2'] = df_pairs_neg['Title_2'].str.split().str.len()

    # 删除某列中包含的单词数小于 3 的行
    df_pairs_neg = df_pairs_neg[df_pairs_neg['word_count1'] >= 1]
    df_pairs_neg = df_pairs_neg[df_pairs_neg['word_count2'] >= 1]
    

    # 删除多余的 'word_count' 列
    df_pairs_neg = df_pairs_neg.drop('word_count1', axis=1)
    df_pairs_neg = df_pairs_neg.drop('word_count2', axis=1)
    
    df_pairs_neg = df_pairs_neg[df_pairs_neg.apply(lambda x: bool(re.search(r'\w', x['Title_1'])), axis=1)]
    df_pairs_neg = df_pairs_neg[df_pairs_neg.apply(lambda x: bool(re.search(r'\w', x['Title_2'])), axis=1)]
    df_pairs_pos = df_pairs_pos[df_pairs_pos.apply(lambda x: bool(re.search(r'\w', x['Title_1'])), axis=1)]
    df_pairs_pos = df_pairs_pos[df_pairs_pos.apply(lambda x: bool(re.search(r'\w', x['Title_2'])), axis=1)]
    
    

    df_pairs_neg.reset_index(drop=True, inplace=True)
    
    df_pairs_neg.to_csv('./TSE/dataAll/'+prefix+'/'+'neg.csv',index=False)#, index=False, header=False)
    df_pairs_pos.to_csv('./TSE/dataAll/'+prefix+'/'+'pos.csv',index=False)#, index=False, header=False)
    '''
    '''




def load_glove_as_dict(filepath):
    word_vec = {}
    with open(filepath,encoding='utf-8') as fr:
        for line in fr:
            line = line.split()
            word = line[0]
            vec = line[1:]
            word_vec[word] = vec
    return word_vec

 
