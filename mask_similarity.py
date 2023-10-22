import numpy as np
import pandas as pd

from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity

bc = BertClient(check_length=False)


def Mask(sentence, tag,index):
    index
    slen = len(sentence)

    sentence_replace = []
    for i in range(slen):
        if tag in sentence[i][1]:
            sentence_replace.append('MASK')
        else:
            sentence_replace.append(sentence[i][0])
    sentence_result = ""
    for wd in sentence_replace:
        sentence_result += " " + wd
    masked_sentence = bc.encode([sentence_result])



    return masked_sentence

def get_mask(dd1,dd2,path,x):
    # sentence1_read = pd.read_csv("path1", header=None, index_col=0)
    # sentence2_read = pd.read_csv("path2", header=None, index_col=0)

    j = 0
    k = 0
    resultcombine = []

    for i in range(x):
        sentence1 = []
        sentence2 = []
        while dd1.values[j][0] != '.':
            temp1 = []
            temp1.append(dd1.values[j][0])
            temp1.append(dd1.values[j][1])
            sentence1.append(temp1)
            j += 1
        j += 1

        while dd2.values[k][0] != '.':
            temp2 = []
            temp2.append(dd2.values[k][0])
            temp2.append(dd2.values[k][1])
            sentence2.append(temp2)
            k += 1
        k += 1


        PL = Mask(sentence1, 'PL',i)
        API = Mask(sentence1, 'API',i)
        Env = Mask(sentence1, 'Env',i)
        UI = Mask(sentence1 ,'UI',i)
        Plat = Mask(sentence1, 'Plat',i)
        Sec = Mask(sentence1, 'Sec',i)
        Stan = Mask(sentence1, 'Stan',i)
        
        sen1=""
        for word in sentence1[0]:
            sen1+= " " + word
        sen1=bc.encode([sen1])
        new_sen1=PL+API+Env+UI+Plat+Sec+Stan+sen1
        
        PL = Mask(sentence2, 'PL',i)
        API = Mask(sentence2, 'API',i)
        Env = Mask(sentence2, 'Env',i)
        UI = Mask(sentence2 ,'UI',i)
        Plat = Mask(sentence2, 'Plat',i)
        Sec = Mask(sentence2, 'Sec',i)
        Stan = Mask(sentence2, 'Stan',i)
        
        sen2=""
        for word in sentence2[0]:
            sen2+= " " + word
        sen2=bc.encode([sen2])
        new_sen2=PL+API+Env+UI+Plat+Sec+Stan+sen1
    
    return new_sen1,new_sen2

        


    
if __name__=='__main__':
    prefix="spark"
    cnn_sim=pd.read_csv('./TSE/dataAll/'+prefix+'/'+prefix+'_sim.csv')
    sim=cnn_sim['sim']
    label=cnn_sim['label']
    l=len(label)
    path1="./TSE/dataAll/"+prefix+"/sentence1_result.csv"
    path2="./TSE/dataAll/"+prefix+"/sentence2_result.csv"
    columns_to_keep = [1, 2]
    dd1= pd.read_csv(path1,usecols=columns_to_keep,header=None)
    dd2=pd.read_csv(path2,usecols=columns_to_keep,header=None)
    get_mask(dd1,dd2,"./TSE/dataAll/"+prefix,l)


