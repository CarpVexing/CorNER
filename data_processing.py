import pandas as pd
from fenci import read_data,tokenize,remove_stop_words,get_pos
from nltk.corpus import stopwords
import csv
import re




def save_token(save_path,lens,nostop_word_pos_tokens_with):
    with open(save_path, 'w', newline='') as f:
        data_writer = csv.writer(f, lineterminator='\n')
        header=['Sentence #','Word','POS']
        data_writer.writerow(header)
        for i in range(lens):
            for wd in nostop_word_pos_tokens_with[i]:
                data_writer.writerow(wd)



    
    
    

def data_process(file_root,file):
    prefix = file_root
    path='./TSE/dataAll/'+prefix
    df_neg = pd.read_csv(path+'/neg.csv',na_values='')

    df_pos = pd.read_csv(path+'/pos.csv',na_values='')

    df = pd.concat([df_neg, df_pos], axis=0)
    
    pairid_list = file['pair_id'].tolist()  # 获取file中的pair_id列表
    df = df.iloc[pairid_list]  # 使用iloc按行号提取对应的行

    # print(df)
    df1=df['Title_1']
    df2=df['Title_2']
    file1=path+'/sentence1'+'.csv'
    file2=path+'/sentence2'+'.csv'
    # df1.to_csv(file1,index=None)
    # df2.to_csv(file2,index=None)
    # df3.to_csv(file3,index=None)

    # get_lines(df1,file1)
    # get_lines(df2,file2)
    
    return df1,df2
    

