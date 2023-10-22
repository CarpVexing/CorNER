import pandas as pd
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# 3.划分数据集与测试集
from sklearn.model_selection import train_test_split
# 4.模型搭建
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# from imblearn.over_sampling import SMOTE

def LR_train(path,prefix):
    df = pd.read_excel(path+prefix+'_sim.xlsx',index_col=0)
    df = df.sample(frac=1, random_state=42)
        
    # 2.划分特征变量与目标变量
    X = df.drop(columns='是否重复')
    y = df['是否重复']



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
        # 使用SMOTE进行过采样
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)
    
    param_grid = {'C': [0.1, 1, 10, 100]}

    model = LogisticRegression(random_state=42, max_iter=5000)
# 使用 GridSearchCV 进行交叉验证调参
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # 打印最佳参数
    print("Best parameters: ", grid_search.best_params_)

    # 使用最佳参数重新训练模型
    model= LogisticRegression(C=grid_search.best_params_['C'])


    model.fit(X_train, y_train)

    # 5.预测分类结果
    y_pred = model.predict(X_test)
    #y_pred[:5]
    # print(y_pred)

    # 6.预测概率
    y_pred_proba = model.predict_proba(X_test)
    #y_pred_proba[:5]
    # print(y_pred_proba)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)#[[TN, FP],
                                               #[FN, TP]]


    # print('准确率:',model.fit(X_train, y_train).score(X_test,y_test))
    # print('召回率:',recall_score(y_test,model.fit(X_train, y_train).predict(X_test)))
    # print('精确率:',accuracy_score(y_test,model.fit(X_train, y_train).predict(X_test)))
    # print('f1-score:',f1_score(y_test,model.fit(X_train, y_train).predict(X_test)))
    # print('混淆矩阵',confusion_matrix(y_test, y_pred))
    
    with open(path+'lr_result.txt', 'a') as f:
        f.write('Accuracy: {}\n'.format(acc))
        f.write('Precision: {}\n'.format(prec))
        f.write('Recall: {}\n'.format(rec))
        f.write('F1 score: {}\n'.format(f1))
        f.write('Confusion matrix:\n')
        f.write('{}\n'.format(conf_mat))

if __name__ == "__main__":
    prefix='spark'
    LR_train('./TSE/dataAll/'+prefix+'/',prefix);