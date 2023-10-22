import re
import os
import random
import tarfile
import urllib
from torchtext.legacy import data
# from transformers import BertModel



data_path = './TSE/dataAll/'


class TarDataset(data.Dataset):
    

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                tfile.extractall(root)
        return os.path.join(path, '')


class MR(TarDataset):

    #url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar'
    dirname = 'rt-polaritydata'

    @staticmethod
    def sort_key(ex):
        return len(ex.issue1)
    
    # def __getitem__(self, index):
    #     example = self.examples[index]
    #     # 从 example 中提取数据和标签
    #     datasets = example.datasets
    #     return datasets

    def __init__(self,prefix,issue1_field, issue2_field, label_field, pairid_field,args,path=None,examples=None, **kwargs):
        def clean_str(string):
           
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            # string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            string= re.sub(r'[A-Z]', lambda match: match.group(0).lower(), string)
            return string.strip()

        issue1_field.preprocessing = data.Pipeline(clean_str)
        issue2_field.preprocessing = data.Pipeline(clean_str)
        fields = [('issue1', issue1_field), ('issue2', issue2_field), ('label', label_field), ('pairid', pairid_field)]

        if examples is None:
            path = data_path+prefix+'/'
            examples = []
            count = 0
            weight=0
            #'''
            with open(os.path.join(path, 'neg.csv'), errors='ignore') as f:
                next(f)
                for line in f:
                    examples.append(data.Example.fromlist([line.split(',')[0], line.split(',')[1],'negative', str(count)], fields))
                    count += 1
                    weight=count
            with open(os.path.join(path, 'pos.csv'), errors='ignore') as f:
                next(f)
                for line in f:
                    examples.append(data.Example.fromlist([line.split(',')[0], line.split(',')[1],'positive', str(count)], fields))
                    count += 1
            weight=float(weight)
            count=float(count-weight)
            weight=weight/count
            args.weight=weight
            #'''
            '''with open(os.path.join(path, 'cnn_train_new.csv'), errors='ignore') as f:
                 for line in f:
                    if count == 0:
                        count += 1
                        continue
                    
                    examples.append(data.Example.fromlist([line.split(',')[1], line.split(',')[2], line.split(',')[3], str(count)], fields))
                    count += 1
                    if (line.split(',')[1] != '') & (line.split(',')[2] != ''):
                        examples.append(data.Example.fromlist([line.split(',')[1], line.split(',')[2], line.split(',')[3], str(count)], fields))
                        count += 1
                    else:
                        print(line)
            '''
         #   print(count)
            print('-----------------------------------------------------------------')
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, prefix,issue1_field, issue2_field, label_field, pairid_field,args,dev_ratio=.1, shuffle=True, root='.', **kwargs):
        
        # path = cls.download_or_unzip(root)
        examples = cls(prefix,issue1_field, issue2_field, label_field, pairid_field,args, **kwargs).examples
        # for i in examples:
        #     print(i.data)
        # if shuffle: 
        #    random.shuffle(examples)
        if shuffle:
            random.shuffle(examples)  # 在这里打乱 examples 列表
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(prefix,issue1_field, issue2_field, label_field, pairid_field,args, examples=examples[:int(0.5*len(examples))]),
               cls(prefix,issue1_field, issue2_field, label_field, pairid_field,args,examples=examples[int(0.5*len(examples)):int(0.6*len(examples))]),
                cls(prefix,issue1_field, issue2_field, label_field, pairid_field,args,examples=examples[int(0.6*len(examples)):int(len(examples))]))
