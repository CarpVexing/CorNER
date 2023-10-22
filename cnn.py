# from preprocess import Preprocess
import pandas as pd
import numpy as np
import torch
import torchtext.legacy.data as data
import torchtext.legacy.datasets as datasets
import argparse
from load_data import load_data, load_glove_as_dict


import mydatasets
import os
import datetime
import traceback
import model
import train
import pickle
from gensim.models import Word2Vec
import jieba


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

use_global_w2v = True
glove_path = './wordvec.txt/wordvec.txt'



def train_word2vec_model(df):
    
    corpus = []
    for i, r in df.iterrows():
        try:
            corpus.append(jieba.lcut(r['Title']))
            # print jieba.lcut(r['ques1'])
            corpus.append(jieba.lcut(r['Description']))
        except:
            pass
            # print('Exception: ', r['ques1']
    word2vec_model = Word2Vec(corpus, vector_size=512, window=3, min_count=1, sg=0, epochs=100)
    return word2vec_model





def train_cnn(prefix):
    parser = argparse.ArgumentParser(description='')
    # 参数
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=56, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance') 
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
   
    parser.add_argument('-dropout', type=float, default=0.6, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=300, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    parser.add_argument('-padding', type=str, default='1')

    parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    args = parser.parse_args()
    args.weight=1
    #数据加载
    #load_data(load_path)
    
    '''
    '''
   # print("\n加载 data...")
    issue1_field = data.Field(lower=True)
    issue2_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    pairid_field = data.Field(sequential=False, use_vocab=False, preprocessing=int)

    train_data, dev_data, test_data = mydatasets.MR.splits(prefix,issue1_field, issue2_field, label_field, pairid_field,args)
    
    issue1_field.build_vocab(train_data, dev_data, test_data)
    issue2_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    # pairid_field.build_vocab(train_data, dev_data, test_data)
   # print(len(train_data), len(dev_data), len(test_data))
    train_iter, dev_iter, test_iter = data.Iterator.splits(
                                (train_data, dev_data, test_data), 
                                batch_sizes=(args.batch_size, len(dev_data),args.batch_size ), device = torch.device('cpu'),repeat=False)
    
    args.embed_num = len(issue1_field.vocab) + len(issue2_field.vocab)
    args.class_num = len(label_field.vocab) - 1

    # add
    if use_global_w2v:
        embedding_dict = load_glove_as_dict(glove_path)
    else:
        try:
            embedding_dict = Word2Vec.load('./TSE/dataAll/'+prefix+'/wordvec_save/'+prefix+'_w2v.save')
        except: 
            df = pd.read_csv('./TSE/dataAll/'+prefix+'/'+prefix+'.csv', encoding = 'utf-8')
            word2vec_model = train_word2vec_model(df)
            word2vec_model.save('./TSE/dataAll/'+prefix+'/wordvec_save/'+prefix+'_w2v.save')
            embedding_dict = Word2Vec.load('./TSE/dataAll/'+prefix+'/wordvec_save/'+prefix+'_w2v.save')
    word_vec_list = []
    for idx, word in enumerate(issue1_field.vocab.itos):
        try:
            vector = np.array(embedding_dict[word], dtype=float).reshape(1, args.embed_dim)
        except:
            vector = np.random.rand(1, args.embed_dim)
        word_vec_list.append(torch.from_numpy(vector))
    for idx, word in enumerate(issue2_field.vocab.itos):
        try:
            vector = np.array(embedding_dict[word], dtype=float).reshape(1, args.embed_dim)
        except:
            vector = np.random.rand(1, args.embed_dim)
        word_vec_list.append(torch.from_numpy(vector))
    wordvec_matrix = torch.cat(word_vec_list)
    args.pretrained_weight = wordvec_matrix
    # add_end

    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    
    # model
    cnn = model.CNN_Text(args)
    if args.snapshot is not None:
        print('\nLoading model from {}...'.format(args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot))
    else:
        try:
            train.train(train_iter, dev_iter, cnn, args)
        except KeyboardInterrupt:
            print(traceback.print_exc())
            print('\n' + '-' * 89)
            print('Exiting from training early')

    
    if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()
        
    
    #'''
    # try:
    #     train.eval_test(test_iter, cnn, args) 
    # except:
    #     print('test_wrong')
    train.eval_test(test_iter, cnn, args,prefix) 
    
    
    
        
if __name__ == '__main__':
    train_cnn()

