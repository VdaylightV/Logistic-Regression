# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:15:37 2019

@author: Administrator
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

matrix = []
filename = 'E:/大学课程/AI程序设计/实验8降维回归和分类/data_akbilgic.csv'

#####################生成矩阵
matrix = []
filename = 'E:/大学课程/AI程序设计/实验8降维回归和分类/data_akbilgic.csv'

for index in range(1,537):
    with open(filename) as file:
        reader = csv.reader(file)
        j = 1
        while j<=index+2:
            row = next(reader)
            j+=1
#         header_title = next(reader)
#         header_row = next(reader)
    #     for line in range(1,539):
#         row = next(reader)
        row_mid = row[1:]
    #     print(row1_mid)
        row_pro = []
        for string in row_mid:
            row_pro.append(eval(string))
        matrix.append(row_pro)
X = np.mat(matrix)  
Y = np.array(matrix) 
#print(Y)

##########################生成DataFrame
DF = pd.DataFrame(Y, index = range(1,537), columns =['ISE_TL','ISE_used','SP', 'DAX', 'FTSE','NIKKEI', 'BOVESPA', 'EU', 'EM'])
#print(DF)
#print(matrix)

#############可视化Seaborn
data = pd.read_csv(filename, engine='python')

sns.pairplot(DF, x_vars = ['ISE_used','SP', 'DAX', 'FTSE','NIKKEI', 'BOVESPA', 'EU', 'EM'], y_vars = 'ISE_TL',kind = 'reg',height = 3, aspect = 0.8)

plt.show()
##############到此已经实现了可视化分析
##############数据集的拆分
#print(DF.describe())

#print(DF[DF.isnull()==True].count())
#print(DF)
X_train,X_test,Y_train,Y_test = train_test_split(DF.iloc[:,1:],DF.ISE_TL,train_size=0.8)
#print(Y_test)



#print("自变量---源数据:",DF.Connect.shape, "；  训练集:",X_train.shape, "；  测试集:",X_test.shape)
#print("因变量---源数据:",DF.Return.shape, "；  训练集:",Y_train.shape, "；  测试集:",Y_test.shape)


#print(DF.corr())
######################3数据集拆分完成

##############线性回归模型：
model = LinearRegression()

model.fit(X_train,Y_train)

a = model.intercept_#截距
b = model.coef_#回归系数

#print("最佳拟合线: Y = ",round(a,2),"+",round(b[0],5),"* X1 + ",round(b[1],5),"* X2 + ",round(b[2],5),"*X3 + ",round(b[3],5),"*X4 + ",round(b[4],5),"*X5 + ",round(b[5],5),"*X6 +",round(b[6],5),"*X7")
Y_pred = model.predict(X_test)

plt.plot(range(len(Y_pred)),Y_pred,'red', linewidth=2.5,label="predict data")
plt.plot(range(len(Y_test)),Y_test,'green',label="test data")
plt.legend(loc=2)
plt.show()#显示预测值与测试值曲线

#print(a)
#print(b)

#################分析数据误差
####泛化误差
Y_pred_array = np.array(Y_pred)
Y_test_array = np.array(Y_test)
MSE_eco = sum(pow(Y_pred_array-Y_test_array,2))/len(Y_pred)
print(MSE_eco)

####训练误差
X_pred = model.predict(X_train)
plt.plot(range(len(X_pred)),X_pred,'red', linewidth=2.5,label="predict data")
plt.plot(range(len(Y_train)),Y_train,'green',label="test data")
plt.legend(loc=2)
plt.show()#显示预测值与测试值曲线

X_pred_array = np.array(X_pred)
Y_train_array = np.array(Y_train)
TE_eco = sum(pow(X_pred_array-Y_train_array,2))/len(X_pred)
print(MSE_eco)
print(TE_eco)

#print(a)
#print(b)

#################分析数据误差
####MSE
#X_pred_array = np.array(X_pred)
#X_test_array = np.array(X_test)
#MSE_eco = sum(pow(X_pred_array-X_test_array,2))/len(X_pred)
#print(X_train)
#print(X_test)