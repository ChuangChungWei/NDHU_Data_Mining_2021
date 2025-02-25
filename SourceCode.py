#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import tree, svm, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


# In[2]:


# 找出最佳參數
def find_best_param(classifier, x, y, param_grid=ParameterGrid({})):
    best_score = 0.0
    best_param = {}
    for params in param_grid:
        clf = classifier(**params)
        score = cross_val_score(clf, x, y, cv=10).mean()
        if score > best_score:
            best_score = score
            best_param = params
    print(clf.__class__.__name__)
    print("Best score is: ", best_score)
    print("Best param is: ", best_param)


# In[3]:


###讀取檔案，並取出特徵名稱
df = pd.read_csv("heart.csv")
X = df[df.columns[:11]]
features = list(df.columns[0:11])

###將St_Slope的有序文字資料進行mapping
mapping = {
    'Down':0,
    'Flat':1,
    'Up':2
}
df['ST_Slope']=df['ST_Slope'].map(mapping)

###將所有無序無字資料進行one-hot encoding
for feature in features:
    dic=np.unique(df[feature].values)
    
    if isinstance (dic[0],str):

        onehot_encoding=pd.get_dummies(df[feature],prefix=feature)
        df=df.drop(feature,1)
        df=pd.concat([onehot_encoding,df],axis=1)


###將進行one-hot encoding後的前18欄訓練資料設給目標標籤X，即表示18種特徵，最後一欄預測資料設給目標標籤y，即表示是否有心臟病。
features = list(df.columns[0:18])
X = df[features]
y = df["HeartDisease"].astype(dtype=np.float32)

###對所有特徵資料正規化置0到1之間。
scaler = MinMaxScaler()
scaler.fit(X)
x = scaler.transform(X)

###對特徵資料進行PCA降維轉換
pca = PCA(n_components=7)
pca.fit(X)


# In[4]:


###Decision tree###
param_grid = ParameterGrid({'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [2,3,4,5,10, 15, 20,25,30,None],
        'min_samples_split': [2, 3, 4,5],
        'min_samples_leaf': [1, 2, 3,4,5],
        'class_weight':['balanced',None,{0:410,1:508}]})
find_best_param(tree.DecisionTreeClassifier, x, y, param_grid)


# In[ ]:


###KNN###
param_grid = ParameterGrid({'n_neighbors': list(range(2, 11)),
              'weights': ['uniform', 'distance'],
              'algorithm': ['ball_tree', 'kd_tree', 'brute'],
              'p': [1, 2]})
find_best_param(KNeighborsClassifier, x, y, param_grid)


# In[ ]:


###Naive Gaussian###
param_grid = ParameterGrid({'var_smoothing': [1e-4,1e-5,1e-7,1e-8,1e-9 
                                              ,1e-10]})
find_best_param(GaussianNB, x, y, param_grid)


# In[ ]:


###SVM###
param_grid = ParameterGrid({'kernel': ['linear', 'poly','rbf','sigmoid'],
        'class_weight':['balanced',None,{0:410,1:508}],
        'C':[2,3,4,5],
        'degree':[3,4,5]})
find_best_param(svm.SVC, x, y, param_grid)


# In[ ]:


###Random Forest###
param_grid = ParameterGrid({'n_estimators': [50,100,200,250,300],
        'max_depth': [2,3,4,5,10, 15, 20,25,30,None],
        'min_samples_split': [2, 3, 4,5],
        'min_samples_leaf': [1, 2, 3,4,5],
        'class_weight':['balanced',None,{0:410,1:508}],
        'criterion':['gini','entropy']})
find_best_param(ensemble.RandomForestClassifier, x, y, param_grid)


# In[ ]:


###Logistic Regression###
class_weight = ('class_weight', ['balanced', None])
C = ('C', [i/10 for i in range(1, 11)])
param_grid = [
    {'penalty': 'l1', 'solver': 'liblinear', 'multi_class': 'ovr'},
    {'penalty': 'l1', 'solver': 'saga', 'multi_class': 'ovr'},
    {'penalty': 'l1', 'solver': 'saga', 'multi_class': 'multinomial'},
    {'penalty': 'elasticnet', 'solver': 'saga', 'multi_class': 'ovr', 'l1_ratio': 0},
    {'penalty': 'elasticnet', 'solver': 'saga', 'multi_class': 'ovr', 'l1_ratio': 1},
    {'penalty': 'elasticnet', 'solver': 'saga', 'multi_class': 'multinomial', 'l1_ratio': 0},
    {'penalty': 'elasticnet', 'solver': 'saga', 'multi_class': 'multinomial', 'l1_ratio': 1},
    {'penalty': 'l2', 'solver': 'newton-cg', 'multi_class': 'ovr'},
    {'penalty': 'l2', 'solver': 'newton-cg', 'multi_class': 'multinomial'},
    {'penalty': 'l2', 'solver': 'lbfgs', 'multi_class': 'ovr'},
    {'penalty': 'l2', 'solver': 'lbfgs', 'multi_class': 'multinomial'},
    {'penalty': 'l2', 'solver': 'sag', 'multi_class': 'ovr'},
    {'penalty': 'l2', 'solver': 'sag', 'multi_class': 'multinomial'},
    {'penalty': 'l2', 'solver': 'saga', 'multi_class': 'ovr'},
    {'penalty': 'l2', 'solver': 'saga', 'multi_class': 'multinomial'},
    {'penalty': 'none', 'solver': 'newton-cg', 'multi_class': 'ovr'},
    {'penalty': 'none', 'solver': 'newton-cg', 'multi_class': 'multinomial'},
    {'penalty': 'none', 'solver': 'lbfgs', 'multi_class': 'ovr'},
    {'penalty': 'none', 'solver': 'lbfgs', 'multi_class': 'multinomial'},
    {'penalty': 'none', 'solver': 'sag', 'multi_class': 'ovr'},
    {'penalty': 'none', 'solver': 'sag', 'multi_class': 'multinomial'},
    {'penalty': 'none', 'solver': 'saga', 'multi_class': 'ovr'},
    {'penalty': 'none', 'solver': 'saga', 'multi_class': 'multinomial'}
]
length = len(param_grid)
for i in range(length):
    for elem in class_weight[1]:
        param_grid[i][class_weight[0]] = elem
        param_grid.append(param_grid[i].copy())
param_grid = param_grid[length:]
count = 0
for item in param_grid:
    if param_grid[count]['penalty'] == 'none':
        break
    for elem in C[1]:
        param_grid[count][C[0]] = elem
        param_grid.append(param_grid[count].copy())
    count += 1
param_grid = param_grid[count:]
find_best_param(LogisticRegression, x, y, param_grid)


###利用voting classifier，分別用不同模型進行預測，投票決定y輸出的種類。
c1 = LogisticRegression(multi_class='multinomial')
c2 = RandomForestClassifier(n_estimators=60)
c3 = GaussianNB()

classifier = VotingClassifier(estimators=[
         ('lr', c1), ('rf', c2), ('gnb', c3)], voting='hard')

val=cross_val_score(classifier, X, y, cv=10)

print(val.mean())

