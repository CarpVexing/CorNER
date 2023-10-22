import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import os
import random
from joblib import dump, load
from imblearn.over_sampling import SMOTE
import hashlib
from fuzzywuzzy import fuzz
# import matplotlib
# import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from mask_similarity import get_mask
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTENC


class MajorityVotingTagger(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        """
        X: list of words
        y: list of tags
        """
        word2cnt = {}
        self.tags = []
        #word的tag计数
        for x, t in zip(X, y):
            if t not in self.tags:
                self.tags.append(t)
            if x in word2cnt:
                if t in word2cnt[x]:
                    word2cnt[x][t] += 1
                else:
                    word2cnt[x][t] = 1
            else:
                word2cnt[x] = {t: 1}
        self.mjvote = {}

        for k, d in word2cnt.items():
            self.mjvote[k] = max(d, key=d.get)
        # 找出最大词标签

    def predict(self, X, y=None):
        '''
        Predict the the tag from memory. If word is unknown, predict 'O'.
        '''
        return [self.mjvote.get(x, 'O') for x in X]
        # 推测词的最常用标签进行标记



def get_sentences(data):
    agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                        s["POS"].values.tolist(),
                                                        s["Tag"].values.tolist())]

    sentence_grouped = data.groupby("Sentence #",sort=False).apply(agg_func)
    return [s for s in sentence_grouped]



# data.tail(10)
# print(data.tail(10))


def hash_func(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)



# 计算单词与给定实体例子里的所有单词的相似度总得分
def calculate_similarity(entity_examples, word):
    similarity_scores = []
    for example in entity_examples:
        example_words = example.split()
        example_scores = [fuzz.ratio(word, example_word) for example_word in example_words]
        similarity_scores.append(max(example_scores))
    return max(similarity_scores)

    # y.append(x)
def get_all_feature(data,tag_encoder,mv_tagger,entities):
    if "Tag" not in data.columns:
        data.insert(loc=3, column='Tag', value=None)
    sentences=get_sentences(data)

    pos_encoder = LabelEncoder()

    words = data["Word"].values.tolist()
    pos = data["POS"].values.tolist()

    # 通过现有数据，计数出word的tag的数据
    
    pos_encoder.fit(pos)
    word_insert = []
    datasets=[]
    for sentence in sentences:
        for i in range(len(sentence)):
            dataset=[]
            w, p,t= sentence[i][0], sentence[i][1],sentence[i][2]
            is_camel_case = w.isidentifier() and any(c.isupper() for c in w[1:-1])
            hasDot=False
            hasUnderline=False
            hasHyphen=False
            hasSlash=False
            if '.' in w:
                hasDot=True
            if '_' in w:
                hasUnderline=True
            if '-' in w:
                hasHyphen=True
            if '\\' in w:
                hasSlash=True
            prefix = w[:5] if len(w) > 5 else w # 取前3个字符作为前缀
            suffix = w[-5:]  if len(w) > 5 else w# 取后3个字符作为后缀
            if i < len(sentence) - 1:
                # 如果不是最后一个单词，则可以用到下文的信息
                mem_tag_r = tag_encoder.transform(mv_tagger.predict([sentence[i + 1][0]]))[0]
                true_pos_r = pos_encoder.transform([sentence[i + 1][1]])[0]
            else:
                mem_tag_r = tag_encoder.transform(['O'])[0]
                true_pos_r = pos_encoder.transform(['.'])[0]

            if i > 0:
                # 如果不是第一个单词，则可以用到上文的信息
                mem_tag_l = tag_encoder.transform(mv_tagger.predict([sentence[i - 1][0]]))[0]
                true_pos_l = pos_encoder.transform([sentence[i - 1][1]])[0]
            else:
                mem_tag_l = tag_encoder.transform(['O'])[0]
                true_pos_l = pos_encoder.transform(['.'])[0]
            # print (mem_tag_r, true_pos_r, mem_tag_l, true_pos_l)

            dataset.append(np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),is_camel_case,hasDot,hasUnderline,hasHyphen,hasSlash,i,
                                calculate_similarity(entities['API'],w),
                                calculate_similarity(entities['Env'],w),
                                calculate_similarity(entities['PL'],w),
                                calculate_similarity(entities['Plat'],w),
                                calculate_similarity(entities['Sec'],w),
                                calculate_similarity(entities['Stan'],w),
                                calculate_similarity(entities['UI'],w),
                                hash_func(prefix),
                                hash_func(suffix),
                                tag_encoder.transform(mv_tagger.predict([sentence[i][0]])),
                                pos_encoder.transform([p])[0],
                                mem_tag_r,
                                true_pos_r,
                                mem_tag_l,
                                true_pos_l
                                ],dtype=object))
            word_insert.append(w)
            dataset.append(t)
            datasets.append(dataset)
    return datasets,word_insert

def add_sentence_num(data):
    temp=None
    for i in range(len(data)):
        if data['Sentence #'][i]!='':
            temp=data['Sentence #'][i]
        if data['Sentence #'][i]=='':
            data['Sentence #'][i]=temp
    return data

def tags_weight(tags):
    len_tags=len(tags);
    TagsWeight={};
    for tag in tags:
        if tag in TagsWeight:
            TagsWeight[tag]+=1;
        else:
            TagsWeight[tag]=1;
    for tag in TagsWeight:
        TagsWeight[tag]=len_tags//TagsWeight[tag]
    return TagsWeight



def rf_train(path,l):
    data = pd.read_csv("./TSE/dataAll/"+path+"/"+path+"All.csv",na_filter=False,encoding="latin1")
    data = data.fillna(method="ffill")
    data1 = pd.read_csv("./TSE/dataAll/"+path+"/sentence1_tokenize.csv",na_filter=False, encoding="latin1")
    data1 = data1.fillna(method="ffill")
    data2 = pd.read_csv("./TSE/dataAll/"+path+"/sentence2_tokenize.csv", na_filter=False,encoding="latin1")
    data2 = data2.fillna(method="ffill")
    data=add_sentence_num(data)
    data1=add_sentence_num(data1)
    data2=add_sentence_num(data2)
    
    tags=data["Tag"].values.tolist()
    words_len=len(data)*8//10
    words = data["Word"].values.tolist()

    TagsWeight=tags_weight(tags[:words_len])

    mv_tagger = MajorityVotingTagger()
    tag_encoder = LabelEncoder()
    
    mv_tagger.fit(words[:words_len], tags[:words_len])
    tag_encoder.fit(tags)
   
    param_grid = {
    'n_estimators': [160,180,200],  # 树的数量
    'max_depth': [None, 10, 20,30,40],  # 树的最大深度，None 表示不限制深度
    'max_features': ['auto', 'sqrt'],  # 特征的最大数量
    # 其他超参数可以继续添加
    }
    
    entities = {
    'PL': ['Java', 'C#', 'C++', 'Python', 'Ruby', 'JavaScript', 'SQL', 'Clojure'],
    'API': ['package', 'class', 'interface', 'method', 'function', 'events', 'JavaScriptonclickevent','API','api','algorithm','hash','plugge','RDD','rdd',path],
    'Env': ['Eclipse', 'Firebug', 'NumPy', 'jQuery', 'maven', 'Spring', 'Markdown'],
    'UI': ['userinterface', 'pagelayout', 'button','UI','web','page','ui',path],
    'Plat': ['x86', 'AMD64', 'MacHardware', 'Android', 'Windows', 'operating systems'],
    'Sec': ['SQLinjection', 'cookiesdatasecurity', 'URLtampered','security'],
    'Stan': ['.xml', 'TCP', 'singletonpattern', 'AJAX', 'JDBC']
    }
    Dataset,word_insert=get_all_feature(data,tag_encoder,mv_tagger,entities)
    sentence1,word_insert1=get_all_feature(data1,tag_encoder,mv_tagger,entities)
    sentence2,word_insert2=get_all_feature(data2,tag_encoder,mv_tagger,entities)

    if not os.path.exists("./TSE/dataAll/"+path+"/model/Random_Forest.pkl"):
        Dataset= np.array(Dataset)
        n = len(Dataset)*8//10
        train=Dataset[:n]
        test=Dataset[n:]
        np.random.shuffle(train)
        
        x_train = train[:,0].tolist()
        y_train = train[:,1].tolist()
        x_test = test[:,0].tolist()
        y_test = test[:,1].tolist()

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        unique_labels = np.unique(y_train)
        indices_to_remove = []
        
        for label in unique_labels:
            idx = np.where(y_train == label)[0]
            if len(idx) == 1: # 标签只出现了一次
                indices_to_remove.extend(idx)
        
        x_train_d = np.delete(x_train, indices_to_remove, axis=0)
        y_train_d = np.delete(y_train, indices_to_remove, axis=0)

        oversampler = SMOTE(k_neighbors=1,random_state=42)
        x_resampled, y_resampled = oversampler.fit_resample(x_train_d, y_train_d)

        x_train = np.concatenate([x_resampled, x_train[indices_to_remove]])
        y_train = np.concatenate([y_resampled, y_train[indices_to_remove]])
        
        X_train=x_train.tolist();
        Y_train=y_train.tolist();


        TagsWeight=tags_weight(Y_train)
        rlf = RandomForestClassifier(n_estimators=181, class_weight=TagsWeight, max_depth=40, max_features='auto', random_state=90)
        # rlf=RandomForestClassifier(class_weight=TagsWeight,random_state=90)
        # grid_search = GridSearchCV(rlf, param_grid, cv=5)
        # grid_search.fit(X_train, Y_train)
        
        # print("Best parameters found: ", grid_search.best_params_)
        # print("Best score found: ", grid_search.best_score_)
        #score_lt = []
        # # 每隔10步建立一个随机森林，获得不同n_estimators的得分
        # matplotlib.use('TkAgg')
        # for i in range(0,200,10):
        #     rfc = RandomForestClassifier(n_estimators=i+1
        #                                 ,random_state=90)
        #     score = cross_val_score(rfc,X_train,Y_train, cv=2).mean()
        #     score_lt.append(score)
        # score_max = max(score_lt)
        # print('最大得分：{}'.format(score_max),
        #     '子树数量为：{}'.format(score_lt.index(score_max)*10+1))
        # # 绘制学习曲线
        # x = np.arange(1,201,10)
        # plt.subplot(111)
        # plt.plot(x, score_lt, 'r-')
        # plt.savefig('./plt.png')
        # plt.show()
        
        rlf.fit(X_train,Y_train)
        report=classification_report(y_test,rlf.predict(x_test))
        with open("./TSE/dataAll/"+path+"/model/classification_report.txt", 'w') as f:
            f.write(report)
        dump(rlf, "./TSE/dataAll/"+path+"/model/Random_Forest.pkl")
        

    rlf=load("./TSE/dataAll/"+path+"/model/Random_Forest.pkl")

    sen1=np.array(sentence1)
    sen2=np.array(sentence2)
    pred1 = rlf.predict(sen1[:,0].tolist())
    pred2 = rlf.predict(sen2[:,0].tolist())
    datacombine1 = {
    'Word': word_insert1,
    'Tag': pred1 
    }
    datacombine2 = {
    'Word': word_insert2,
    'Tag': pred2
    }
    dd1 = pd.DataFrame(datacombine1, index=None)
    dd2 = pd.DataFrame(datacombine2, index=None)
    dd1.to_csv("./TSE/dataAll/"+path+"/sentence1_result.csv", header=False)
    dd2.to_csv("./TSE/dataAll/"+path+"/sentence2_result.csv", header=False)

    
    return get_mask(dd1,dd2,"./TSE/dataAll/"+path,l)


