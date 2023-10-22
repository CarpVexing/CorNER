from data_processing import data_process
from Method2_RandomForest import rf_train
import pandas as pd
from load_data import load_data
from cnn import train_cnn 
from LR import LR_train

prefix='spark'
# load_data(prefix)
train_cnn(prefix)
# cnn_sim=pd.read_csv('./TSE/dataAll/'+prefix+'/'+prefix+'_sim.csv')
# sim=cnn_sim['sim']
# label=cnn_sim['label']
# title1,title2=data_process(prefix,cnn_sim)
# l=len(label)  
# dd=rf_train(prefix,l) 
# result=pd.concat([sim,dd,label], axis=1)
# dd = dd.reset_index(drop=True)
# label = label.reset_index(drop=True)
# result=pd.concat([sim,dd,label], axis=1)
# result.rename(columns={"label":"是否重复"},inplace=True) 
# # result.rename(columns={"sim":"textcnn"},inplace=True)
# # result.reset_index(drop=True, inplace=True)
# result = result.sample(frac=1).reset_index(drop=True)
# result.insert(0, 'Sentence—pairs #', range(1, len(dd) + 1))
# result.to_excel('./TSE/dataAll/'+prefix+'/'+prefix+'_sim.xlsx', index=False)
# LR_train('./TSE/dataAll/'+prefix+'/',prefix)
