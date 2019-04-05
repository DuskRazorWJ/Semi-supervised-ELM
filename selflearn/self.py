import pandas as pd
import time

a=time.time()
data=pd.read_excel('C:/Users/Administrator/Desktop/论文草稿/代码+数据/mnist_5_4000.xlsx',header=None)
X1=data.iloc[:,:-1]
y=data.iloc[:,-1]

import numpy as np
X1=np.array(X1)
y=np.array(y)
ytrue1=np.copy(y)
ys=np.array([-1]*len(ytrue1))

import random
random_sample=random.sample( (range(len(ytrue1))),len(ytrue1))

X=X1[random_sample]
ytrue=ytrue1[random_sample]

random_labeled_points=random.sample(list(range(len(ytrue))),int(0.2*(len(ytrue))))
ys[random_labeled_points]=ytrue[random_labeled_points]

import sklearn
from sklearn.linear_model import SGDClassifier
basemodel = SGDClassifier(loss='log', penalty='l1')
basemodel.fit(X[random_labeled_points, :], ys[random_labeled_points])
print ("supervised log.reg. score", basemodel.score(X, ytrue))

import sys
from sys import path
path.append('C:/Users/Administrator/Desktop/论文草稿/代码+数据/selflearn/frameworks/')

from SelfLearning import SelfLearningModel
ssmodel = SelfLearningModel(basemodel)
ssmodel.fit(X, ys)
print ("self-learning log.reg. score", float(100*(ssmodel.score(X, ytrue))))
b=time.time()
print("running time:",float(b-a))
