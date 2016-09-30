#-*- coding: utf-8 -*-
'''
Created on 2016年9月29日

@author: Activity00
'''

from math import log
import math

import numpy as np


#********************数据收集与准备***********
def getData():
    '''[out]:
            lines    numpy.mat    原始数据
            TreatmentPlan  []  治疗方案
            inedex    []    指标
             
    '''
    TreatmentPlan=['药物治疗','体外碎石','腹腔镜手术','传统手术']
    inedex=['治疗时间(天)','治疗费用（元）','根治程度','副作用','安全性']
    
    f1=open('dx.data','r')
    dx=[line.strip().split('\t') for line in f1.readlines()]
    f2=open('data.txt','r')
    lines=[line.strip().split('\t') for line in f2.readlines()]
    f1.close()
    f2.close()
    #定性指标量化
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            try:
                ind=dx[0].index(lines[i][j])
                lines[i][j]=dx[1][ind]
            except(ValueError):
                pass
#     temp=[]
#     for x in lines:
#         temp.append([int(y) for y in x]) 

    return np.mat(lines).astype(int),inedex,TreatmentPlan       
               



#********************数据收集与准备***********

#*********************权重计算方法*************
def getWeights(data):
    '''客观赋权法-熵值法   （除了它还有主观赋值法-德尔菲，相对比较法）
                  1. 对决策矩阵用线性变换做标准化处理
                  2.计算第j个指标的熵值
                  3.计算第j个指标的偏差度
                  4.确定指标权重
        [in]:
            data    numpy.mat    原始数据矩阵
        [out]:weithts    []        权值列表
            
    '''
    #print data
    R=np.zeros((data.shape[0],data.shape[1]))
    RjMin=np.min(data,0)
    RjMax=np.max(data,0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j==0 or j==1 or j==4:
                R[i,j]=RjMin[0,j]*1.0/data[i,j]
            else:
                R[i,j]=data[i,j]*1.0/RjMax[0,j]
    RjSum=np.sum(R, 0)
    P=R/RjSum
    #print P
    Temp=np.zeros((P.shape[0],P.shape[1]))
    k=log(P.shape[0],2.7)**-1
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            Temp[i,j]=-k*P[i,j]*log(P[i,j],math.e)
    shan=np.sum(Temp,0)
    g=1-shan
    weights=g/sum(g)
    #print shan,'\n',g,'\n',weights
    return weights
    
    #shan=- P * log(P,2) #log base 2
#*********************end*************

#**********************核心算法****************

def linearWeightedMethod():
    '''简单线性加权法
       1.读取数据
       2.却确定各决策指标的权重
       3.对决策矩阵进行标准化处理
       4.求可行方案评价指标先行加权和，并以此作为各方案排列依据
    '''
    data,index,TreatmentPlan=getData()
    weights=getWeights(data)
    
    R=np.zeros((data.shape[0],data.shape[1]))
    RjMin=np.min(data,0)
    RjMax=np.max(data,0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j==0 or j==1 or j==4:
                R[i,j]=RjMin[0,j]*1.0/data[i,j]
            else:
                R[i,j]=data[i,j]*1.0/RjMax[0,j]
    U=np.sum(R*weights,1)
    TreatmentPlan
    result=np.argsort(-U)
    return TreatmentPlan[result[0]]
    
#**********************end*******************

print  linearWeightedMethod()
