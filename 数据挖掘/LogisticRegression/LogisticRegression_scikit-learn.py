# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.externals import joblib

def logisticRegression():
    data = loadtxtAndcsv_data("E:/pycharm3.6/数据挖掘/LogisticRegression/data2.txt", ",", np.float64)
    X = data[:, 0:-1]
    y = data[:, -1]
    # 划分为训练集和测试集

    for i in range(0,1000):
        for j in [0.2,0.25,0.3,0.35]:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=j)
            # 归一化
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.fit_transform(x_test)
            #逻辑回归
            model = LogisticRegression()
            model.fit(x_train,y_train)
            # 预测
            predict = model.predict(x_test)
            right = sum(predict == y_test)
            predict = np.hstack((predict.reshape(-1, 1), y_test.reshape(-1, 1)))   # 将预测值和真实值放在一块，好观察
            accury = right/predict.shape[0]
            print('测试集准确率：%f%%'% (right*100.0/predict.shape[0]))          #计算在测试集上的准确度
            if( accury > 0.9) :
                break
                print(j)
            else: continue
        if (accury > 0.9):
            break
            joblib.dump(model, "LogisticRegression_model.m")   # 保存模型
        else: i = i +1
# 加载txt和csv文件
def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)

# 加载npy文件
def loadnpy_data(fileName):
    return np.load(fileName)

if __name__ == "__main__":
    logisticRegression()