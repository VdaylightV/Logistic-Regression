# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:47:57 2019

@author: Administrator
"""
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D




#####################生成矩阵
matrix = []
filename = 'E:/大学课程/AI程序设计/实验8降维回归和分类/wdbc.csv'

#####################导入数据并形成矩阵
for index in range(1,569):
    with open(filename) as file:
        reader = csv.reader(file)
        j = 1
        while j<=index:
            row = next(reader)
            j+=1
#         header_title = next(reader)
#         header_row = next(reader)
    #     for line in range(1,539):
#         row = next(reader)
        row_mid = row[2:]
    
    #     print(row1_mid)
        row_pro = []
        for string in row_mid:
            row_pro.append(eval(string))
        matrix.append(row_pro)
X = np.mat(matrix)  
Y = np.array(matrix) 
# print(Y)

##################利用PCA把数据降维，降成3维
pca = PCA(n_components = 3)
reduced_X = pca.fit_transform(Y)
# print(reduced_X)
reduced_X_matrix = np.mat(reduced_X)
# print(reduced_X)

################形成列表，为后面哦数据可视化作准备
data_sub1 = []
matrix_result = []
row_mid_result = []
for index in range(1,569):
    with open(filename) as file:
        reader = csv.reader(file)
        j = 1
        while j<=index:
            row_result = next(reader)
            j+=1
#         header_title = next(reader)
#         header_row = next(reader)
    #     for line in range(1,539):
#         row = next(reader)
        row_mid_result = row_result[1]
        if row_result[1]=='M':
            data_sub1.append(1)
        else:
            data_sub1.append(0)
#         data_sub1.append(row_result[1])
    
    #     print(row1_mid)
        row_pro_result = []
        for string in row_mid_result:
            row_pro_result.append(string)
        matrix_result.append(row_pro_result)
# print(matrix_result)
#print(data_sub1)
X_axis = []
Y_axis = []
Z_axis = []

# print(reduced_X[0][2])

for i in range(568):
    X_axis.append(reduced_X[i][0])
    Y_axis.append(reduced_X[i][1])
    Z_axis.append(reduced_X[i][2])

data = {'Analyses':data_sub1,'Factor_1':X_axis, 'Factor_2':Y_axis,'Factor3':Z_axis}
DF = pd.DataFrame(data)
DF_matrix = DF.as_matrix(columns = None)

print(DF.corr())
#print(DF_matrix)

X_axis_0 = []
Y_axis_0 = []
Z_axis_0 = []

X_axis_1 = []
Y_axis_1 = []
Z_axis_1 = []

for i in range(568):
    if DF_matrix[i][0]==1.00000000e+00:
        X_axis_0.append(DF_matrix[i][1])
        Y_axis_0.append(DF_matrix[i][2])
        Z_axis_0.append(DF_matrix[i][3])
    else:
        X_axis_1.append(DF_matrix[i][1])
        Y_axis_1.append(DF_matrix[i][2])
        Z_axis_1.append(DF_matrix[i][3])
        
print(X_axis_0)
print(Y_axis_0)
print(Z_axis_0)
print(X_axis_1)
print(Y_axis_1)
print(Z_axis_1)



ax = plt.figure().add_subplot(111, projection = '3d')
ax.scatter(X_axis_0,Y_axis_0,Z_axis_0, c='b',marker='o')
ax.scatter(X_axis_1,Y_axis_1,Z_axis_1, c='r',marker='D')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()



# from sklearn.linear_model import RandomizedLogisticRegression as RLR
# r1 = RLR
# r1.fit(reduced_X_matrix, matrix_result)
# r1.get_support(indices=True)

###################数据拆分


x_train,x_test,y_train,y_test = train_test_split(reduced_X_matrix ,data_sub1,test_size = 0.2, random_state = 0)
#print(len(x_train))
#print(len(y_train))
#print(len(x_test))
#print(len(y_test))


##########分类，训练，测试
classifier = LogisticRegression()
classifier.fit(x_train,y_train)
y_predict_test = classifier.predict(x_test)
y_predict_train = classifier.predict(x_train)
error_index_test=np.nonzero(y_test - y_predict_test)[0]
print('error index of test:',error_index_test)
error_rate_test = len(error_index_test)/len(y_test)
print('error rate of test:',error_rate_test)
error_index_train=np.nonzero(y_train - y_predict_train)[0]
print('error index of train:',error_index_train)
error_rate_train = len(error_index_train)/len(y_train)
print('error rate of train:',error_rate_train)



plt.scatter(range(len(list(y_predict_test))),list(y_predict_test),c='r',marker='o',label="predict data")
plt.scatter(range(len(list(y_test))),list(y_test),c='b',marker='D',alpha = 0.4,label="test data")
plt.legend(loc=0)
plt.show()#显示预测值与测试值曲线

plt.scatter(range(len(list(y_predict_train))),list(y_predict_train),c='r',marker='o',label="predict data")
plt.scatter(range(len(list(y_train))),list(y_train),c='b',marker='D',alpha = 0.2,label="test data")
plt.legend(loc=0)
plt.show()#显示预测值与测试值曲线



