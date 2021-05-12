# -*- coding:utf-8 -*-
import pystan
import matplotlib.pyplot as plt
import arviz
import numpy as np
# from numpy.ma import log, exp,sqrt
from math import log, exp,sqrt
import pandas as pd
import csv
import operator
from functools import reduce
from scipy.stats import norm
import random
import matplotlib.mlab as mlab
import seaborn as sns


def WPMGenerateRegularLogMatrix():
    SampleTimes=30
    TT=[(t+1)*1 for t in range(0,SampleTimes)]
    print(len(TT))
    SampleNumber=10
    YY=[[0*t for t in range(0,SampleTimes)] for i in range(0,SampleNumber)]
    print(YY)
    delta=0
    a=1
    beta=10
    sigma=0.1
    for i in range(0,SampleNumber):#10个样本
        YY[i][0]=0
        delta=log(beta*TT[1])
        YY[i][1]=YY[i][0]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        print(YY[i][0],YY[i][1])
        for j in range(2,SampleTimes):
            delta=log(beta*TT[j])-log(beta*TT[j-1])
            print(i*30+j,TT[j],TT[j-1],"delta=",delta)
            #print()
            YY[i][j]=YY[i][j-1]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        print(YY[i][j])
    print("生成数据")
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("WPM:Generated Regular Log degradation data successfully")
    # new_list=[TT]+YY
    # print(new_list)
    content=pd.DataFrame(columns=None,data=[TT]+YY)
    content.to_csv(r'E:\data\无应力对数尺度维纳过程模型.csv',index=False)
    #np.savetxt(r'E:\data\无应力对数尺度维纳过程模型.csv',[TT,YY],delimiter=',')
    return YY,TT,SampleNumber,SampleTimes


def WPMLogRegular(path):
    #无应力对数尺度
    regular_log_code="""
    data {
        int<lower=0> SampleNumber;
        int<lower=0> SampleTimes;
        vector[SampleTimes]  t;
        real<lower=0> y[SampleNumber,SampleTimes];
    }
    parameters {
        real<lower=0> a;
        real<lower=0> beta;
        real<lower=0> sigma;
    }
    transformed parameters {
    }
    model {
        a~uniform(0,3);
        beta~uniform(5,15);
        sigma~uniform(0,1);
        for(i in 1:SampleNumber){
            for(j in 3:SampleTimes)
                y[i][j]-y[i][j-1]~normal(a*log(t[j]/t[j-1]),sigma*(log(t[j]/t[j-1]))^0.5);
        }
    }
    """
    #P76reference

   #YY,TT,SampleNumber,SampleTimes=WPMGenerateRegularLogMatrix()
    Data = pd.read_csv(path)
    #print(Data)
    TT=Data.values.tolist()[0]
    #print(TT)
    YY=Data.values.tolist()[1:len(Data.values.tolist())]
    SampleNumber=len(YY)
    SampleTimes=len(YY[0])
    #绘图而已
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    #print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    regular_log_data = {"SampleNumber": SampleNumber,
                        "SampleTimes":SampleTimes,
                        "t": TT,
                        "y": YY}
    sm = pystan.StanModel(model_code=regular_log_code)
    fit = sm.sampling(data=regular_log_data,chains=4, iter=1000)
    print("fit=",fit)
    all_parameters=fit.extract(permuted=True)
    print("all_parameters=",all_parameters)
    a=all_parameters['a']
    print("a=",a) #a的结果
    b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
    print("b=",b)
    arviz.plot_trace(fit)
    plt.show()

    # #保存参数估计结果
    # cc=["a","beta","sigma","lp__"]
    # # print(b[0])
    # print(len(b[0]))
    # bb=np.array(b).reshape(len(b)*len(b[0]),4)
    # print(bb)
    # content=pd.DataFrame(columns=cc,data=bb)
    # print(content)
    # content.to_csv(r'E:\data\无应力对数尺度维纳过程模型参数估计结果.csv',index=False)

    print("model13:successful")


def WPMGenerateSingleLogMatrix():
    SampleTimes=30
    SampleNumber=30
    TT=[(t+1)*1 for t in range(0,SampleTimes)]
    SS=[20+(t%3)*10 for t in range(0,SampleNumber)]#3个档位应力大小每个10个样品
    #print(len(TT))
    print(SS)
    YY=[[0*t for t in range(0,SampleTimes)] for i in range(0,SampleNumber)]
    #print(YY)
    delta=0

    phi0=2
    phi1=0.1
    phi2=20
    beta=10
    sigma=0.1
    a=1 #a为加速因子
    for i in range(0,SampleNumber):#10个样本
        YY[i][0]=0
        delta=log(beta*TT[1])
        a=phi0*(SS[i]**phi1)*exp(-phi2/SS[i])
        YY[i][1]=YY[i][0]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        #print(YY[i][0],YY[i][1])
        for j in range(2,SampleTimes):
            delta=log(beta*TT[j])-log(beta*TT[j-1])
            #print(i*30+j,TT[j],TT[j-1],"delta=",delta)
            #print()
            YY[i][j]=YY[i][j-1]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        #print(YY[i][j])
    print("生成数据")
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("WPM:Generated Single Log degradation data successfully")
    temp_list=[TT]+YY
    print(temp_list)
    ex_SS=[0]+SS
    k=[]
    for i in range(0,len(temp_list)):
        k.append([ex_SS[i]]+temp_list[i])

    content=pd.DataFrame(columns=None,data=k)
    print(content)
    content.to_csv(r'E:\data\单应力对数尺度艾琳维纳过程模型.csv',index=False)
    #np.savetxt(r'E:\data\无应力对数尺度维纳过程模型.csv',[TT,YY],delimiter=',')
    return YY,TT,SS,SampleNumber,SampleTimes

def WPMLogSingle(path):
    #无应力对数尺度
    single_log_code="""
    data {
        int<lower=0> SampleNumber;
        int<lower=0> SampleTimes;
        vector[SampleNumber]  s;
        vector[SampleTimes] t;
        real<lower=0> y[SampleNumber,SampleTimes];
    }
    parameters {
        real<lower=0> phi0;
        real<lower=0> phi1;
        real<lower=0> phi2;
        real<lower=0> beta;
        real<lower=0> sigma;
    }
    transformed parameters {
        vector[SampleNumber]  a;
        for(i in 1:SampleNumber)
            a[i]=phi0*(s[i]^phi1)*exp(-phi2/s[i]);
    }
    model {
        phi0~normal(2,0.5);
        phi1~uniform(0,1);
        phi2~normal(20,2);
        beta~normal(10,5);
        sigma~uniform(0,1);
        for(i in 1:SampleNumber){
            for(j in 3:SampleTimes)
                y[i][j]-y[i][j-1]~normal(a[i]*log(t[j]/t[j-1]),sigma*(log(t[j]/t[j-1]))^0.5);
        }
    }
    """
    #P76reference

    #YY,TT,SS,SampleNumber,SampleTimes=WPMGenerateSingleLogMatrix()

    Data = pd.read_csv(path)
    #print(Data)
    TT=Data.values.tolist()[0]
    del(TT[0])
    #print(TT)
    YY_SS=Data.values.tolist()[1:len(Data.values.tolist())]
    SS=[x[0] for x in YY_SS]
    #print(SS)
    YY=[x[1:len(YY_SS[0])] for x in YY_SS]
    # print(YY)
    # print(len(YY[0]))
    SampleNumber=len(YY)
    SampleTimes=len(YY[0])
    #绘图而已
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    #print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    single_log_data = {"SampleNumber": SampleNumber,
                        "SampleTimes": SampleTimes,
                        "s": SS,
                        "t": TT,
                        "y": YY}
    sm = pystan.StanModel(model_code=single_log_code)
    fit = sm.sampling(data=single_log_data,chains=4, iter=1000)
    print("fit=",fit)
    all_parameters=fit.extract(permuted=True)
    print("all_parameters=",all_parameters)
    phi0=all_parameters['phi0']
    print("phi0=",phi0) #a的结果
    b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
    print("b=",b)
    arviz.plot_trace(fit)
    plt.show()
    print("model15:successful")


def WPMGenerateRegularMiMatrix():
    SampleTimes=30
    TT=[(t+1)*1 for t in range(0,SampleTimes)]
    print(len(TT))
    SampleNumber=10
    YY=[[0*t for t in range(0,SampleTimes)] for i in range(0,SampleNumber)]
    print(YY)
    delta=0
    a=0.6
    beta=0.8
    sigma=0.2
    for i in range(0,SampleNumber):#10个样本
        YY[i][0]=0
        delta=TT[1]**beta
        YY[i][1]=YY[i][0]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        print(YY[i][0],YY[i][1])
        for j in range(2,SampleTimes):
            delta=TT[j]**beta-TT[j-1]**beta
            #print(i*30+j,TT[j],TT[j-1],"delta=",delta)
            #print()
            YY[i][j]=YY[i][j-1]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        print(YY[i][j])
    print("生成数据")
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("WPM:Generated Regular Mi degradation data successfully")
    # new_list=[TT]+YY
    # print(new_list)
    content=pd.DataFrame(columns=None,data=[TT]+YY)
    #content.to_csv(r'E:\data\无应力幂律尺度维纳过程模型.csv',index=False)
    #np.savetxt(r'E:\data\无应力对数尺度维纳过程模型.csv',[TT,YY],delimiter=',')
    return YY,TT,SampleNumber,SampleTimes


def WPMMiRegular(path):
    #无应力幂律尺度
    regular_mi_code="""
    data {
        int<lower=0> SampleNumber;
        int<lower=0> SampleTimes;
        vector[SampleTimes]  t;
        real<lower=0> y[SampleNumber,SampleTimes];
    }
    parameters {
        real<lower=0> a;
        real<lower=0> beta;
        real<lower=0> sigma;
    }
    transformed parameters {
    }
    model {
        a~uniform(0,1);
        beta~uniform(0,1);
        sigma~uniform(0,0.5);
        for(i in 1:SampleNumber){
            for(j in 3:SampleTimes)
                y[i][j]-y[i][j-1]~normal(a*(t[j]^beta-t[j-1]^beta),sigma*((t[j]^beta-t[j-1]^beta)^0.5));
        }
    }
    """
    #P76reference

    #YY,TT,SampleNumber,SampleTimes=WPMGenerateRegularMiMatrix()
    Data = pd.read_csv(path)
    #print(Data)
    TT=Data.values.tolist()[0]
    #print(TT)
    YY=Data.values.tolist()[1:len(Data.values.tolist())]
    SampleNumber=len(YY)
    SampleTimes=len(YY[0])
    #绘图而已
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    #print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    regular_mi_data = {"SampleNumber": SampleNumber,
                        "SampleTimes":SampleTimes,
                        "t": TT,
                        "y": YY}
    sm = pystan.StanModel(model_code=regular_mi_code)
    fit = sm.sampling(data=regular_mi_data,chains=4, iter=1000)
    print("fit=",fit)
    all_parameters=fit.extract(permuted=True)
    print("all_parameters=",all_parameters)
    a=all_parameters['a']
    print("a=",a) #a的结果
    b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
    print("b=",b)
    arviz.plot_trace(fit)
    plt.show()
    print("model21:successful")


def WPMGenerateSingleMiMatrix():
    SampleTimes=30
    SampleNumber=30
    TT=[(t+1)*1 for t in range(0,SampleTimes)]
    SS=[20+(t%3)*10 for t in range(0,SampleNumber)]#3个档位应力大小每个10个样品
    #print(len(TT))
    print(SS)
    YY=[[0*t for t in range(0,SampleTimes)] for i in range(0,SampleNumber)]
    #print(YY)
    delta=0

    phi0=2
    phi1=0.1
    phi2=20
    beta=0.8
    sigma=0.1
    a=1 #a为加速因子
    for i in range(0,SampleNumber):#10个样本
        YY[i][0]=0
        delta=TT[1]**beta
        a=phi0*(SS[i]**phi1)*exp(-phi2/SS[i])
        YY[i][1]=YY[i][0]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        #print(YY[i][0],YY[i][1])
        for j in range(2,SampleTimes):
            delta=TT[j]**beta-TT[j-1]**beta
            #print(i*30+j,TT[j],TT[j-1],"delta=",delta)
            #print()
            YY[i][j]=YY[i][j-1]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        #print(YY[i][j])
    print("生成数据")
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("WPM:Generated Single Mi degradation data successfully")
    temp_list=[TT]+YY
    print(temp_list)
    ex_SS=[0]+SS
    k=[]
    for i in range(0,len(temp_list)):
        k.append([ex_SS[i]]+temp_list[i])

    content=pd.DataFrame(columns=None,data=k)
    print(content)
    content.to_csv(r'E:\data\单应力幂律尺度艾琳维纳过程模型.csv',index=False)
    #np.savetxt(r'E:\data\单应力幂律尺度艾琳维纳过程模型.csv',[TT,YY],delimiter=',')
    return YY,TT,SS,SampleNumber,SampleTimes

def WPMMiSingle(path):
    #单应力幂律尺度
    single_mi_code="""
    data {
        int<lower=0> SampleNumber;
        int<lower=0> SampleTimes;
        vector[SampleNumber]  s;
        vector[SampleTimes] t;
        real<lower=0> y[SampleNumber,SampleTimes];
    }
    parameters {
        real<lower=0> phi0;
        real<lower=0> phi1;
        real<lower=0> phi2;
        real<lower=0> beta;
        real<lower=0> sigma;
    }
    transformed parameters {
        vector[SampleNumber]  a;
        for(i in 1:SampleNumber)
            a[i]=phi0*(s[i]^phi1)*exp(-phi2/s[i]);
    }
    model {
        phi0~normal(2,0.5);
        phi1~uniform(0,1);
        phi2~normal(20,2);
        beta~normal(0.8,0.1);
        sigma~uniform(0,1);
        for(i in 1:SampleNumber){
            for(j in 3:SampleTimes)
                y[i][j]-y[i][j-1]~normal(a[i]*(t[j]^beta-t[j-1]^beta),sigma*(t[j]^beta-t[j-1]^beta)^0.5);
        }
    }
    """

    #YY,TT,SS,SampleNumber,SampleTimes=WPMGenerateSingleMiMatrix()

    Data = pd.read_csv(path)
    #print(Data)
    TT=Data.values.tolist()[0]
    del(TT[0])
    #print(TT)
    YY_SS=Data.values.tolist()[1:len(Data.values.tolist())]
    SS=[x[0] for x in YY_SS]
    #print(SS)
    YY=[x[1:len(YY_SS[0])] for x in YY_SS]
    # print(YY)
    # print(len(YY[0]))
    SampleNumber=len(YY)
    SampleTimes=len(YY[0])
    #绘图而已
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    #print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    single_mi_data = {"SampleNumber": SampleNumber,
                       "SampleTimes": SampleTimes,
                       "s": SS,
                       "t": TT,
                       "y": YY}
    sm = pystan.StanModel(model_code=single_mi_code)
    fit = sm.sampling(data=single_mi_data,chains=4, iter=1000)
    print("fit=",fit)
    all_parameters=fit.extract(permuted=True)
    print("all_parameters=",all_parameters)
    phi0=all_parameters['phi0']
    print("phi0=",phi0) #a的结果
    b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
    print("b=",b)
    arviz.plot_trace(fit)
    plt.show()
    print("model23:successful")

def WPMGenerateRegularExpMatrix():
    SampleTimes=30
    TT=[(t+1)*1 for t in range(0,SampleTimes)]
    print(len(TT))
    SampleNumber=10
    YY=[[0*t for t in range(0,SampleTimes)] for i in range(0,SampleNumber)]
    print(YY)
    delta=0
    a=1
    beta=0.1
    sigma=0.1
    for i in range(0,SampleNumber):#10个样本
        YY[i][0]=0
        delta=exp(beta*TT[1])
        YY[i][1]=YY[i][0]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        print(YY[i][0],YY[i][1])
        for j in range(2,SampleTimes):
            delta=exp(beta*TT[j])-exp(beta*TT[j-1])
            print(i*30+j,TT[j],TT[j-1],"delta=",delta)
            #print()
            YY[i][j]=YY[i][j-1]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        print(YY[i][j])
    print("生成数据")
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("WPM:Generated Regular Exp degradation data successfully")
    # new_list=[TT]+YY
    # print(new_list)
    content=pd.DataFrame(columns=None,data=[TT]+YY)
    #content.to_csv(r'E:\data\无应力指数尺度维纳过程模型.csv',index=False)
    #np.savetxt(r'E:\data\无应力对数尺度维纳过程模型.csv',[TT,YY],delimiter=',')
    return YY,TT,SampleNumber,SampleTimes

def WPMExpRegular(path):
    #无应力指数尺度
    regular_exp_code="""
    data {
        int<lower=0> SampleNumber;
        int<lower=0> SampleTimes;
        vector[SampleTimes]  t;
        real<lower=0> y[SampleNumber,SampleTimes];
    }
    parameters {
        real<lower=0> a;
        real<lower=0> beta;
        real<lower=0> sigma;
    }
    transformed parameters {
    }
    model {
        a~uniform(0,5);
        beta~uniform(0,1);
        sigma~uniform(0,1);
        for(i in 1:SampleNumber){
            for(j in 3:SampleTimes)
                y[i][j]-y[i][j-1]~normal(a*exp(beta*t[j])-a*exp(beta*t[j-1]),sigma*(exp(beta*t[j])-exp(beta*t[j-1]))^0.5);
        }
    }
    """
    #P76reference

    #YY,TT,SampleNumber,SampleTimes=WPMGenerateRegularExpMatrix()
    Data = pd.read_csv(path)
    #print(Data)
    TT=Data.values.tolist()[0]
    #print(TT)
    YY=Data.values.tolist()[1:len(Data.values.tolist())]
    SampleNumber=len(YY)
    SampleTimes=len(YY[0])
    #绘图而已
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    #print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    regular_exp_data = {"SampleNumber": SampleNumber,
                        "SampleTimes":SampleTimes,
                        "t": TT,
                        "y": YY}
    sm = pystan.StanModel(model_code=regular_exp_code)
    fit = sm.sampling(data=regular_exp_data,chains=4, iter=1000)
    print("fit=",fit)
    all_parameters=fit.extract(permuted=True)
    print("all_parameters=",all_parameters)
    a=all_parameters['a']
    print("a=",a) #a的结果
    b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
    print("b=",b)
    arviz.plot_trace(fit)
    plt.show()
    # #保存参数估计结果
    # cc=["a","beta","sigma","lp__"]
    # # print(b[0])
    # print(len(b[0]))
    # bb=np.array(b).reshape(len(b)*len(b[0]),4)
    # print(bb)
    # content=pd.DataFrame(columns=cc,data=bb)
    # print(content)
    # content.to_csv(r'E:\data\无应力指数尺度维纳过程模型参数估计结果.csv',index=False)
    print("model17:successful")


def WPMGenerateSingleExpMatrix():
    SampleTimes=30
    SampleNumber=30
    TT=[(t+1)*1 for t in range(0,SampleTimes)]
    SS=[20+(t%3)*10 for t in range(0,SampleNumber)]#3个档位应力大小每个10个样品
    #print(len(TT))
    print(SS)
    YY=[[0*t for t in range(0,SampleTimes)] for i in range(0,SampleNumber)]
    #print(YY)
    delta=0

    phi0=2
    phi1=0.1
    phi2=20
    beta=0.1
    sigma=0.1
    a=1 #a为加速因子
    for i in range(0,SampleNumber):#10个样本
        YY[i][0]=0
        delta=exp(TT[1]*beta)
        a=phi0*(SS[i]**phi1)*exp(-phi2/SS[i])
        YY[i][1]=YY[i][0]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        #print(YY[i][0],YY[i][1])
        for j in range(2,SampleTimes):
            delta=exp(TT[j]*beta)-exp(TT[j-1]*beta)
            #print(i*30+j,TT[j],TT[j-1],"delta=",delta)
            #print()
            YY[i][j]=YY[i][j-1]+np.random.normal(loc=a*delta,scale=sigma*delta**0.5)
        #print(YY[i][j])
    print("生成数据")
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("WPM:Generated Single Exp degradation data successfully")
    temp_list=[TT]+YY
    print(temp_list)
    ex_SS=[0]+SS
    k=[]
    for i in range(0,len(temp_list)):
        k.append([ex_SS[i]]+temp_list[i])

    content=pd.DataFrame(columns=None,data=k)
    print(content)
    #content.to_csv(r'E:\data\单应力指数尺度艾琳维纳过程模型.csv',index=False)
    #np.savetxt(r'E:\data\单应力幂律尺度艾琳维纳过程模型.csv',[TT,YY],delimiter=',')
    return YY,TT,SS,SampleNumber,SampleTimes

def WPMExpSingle(path):
    #单应力幂律尺度
    single_exp_code="""
    data {
        int<lower=0> SampleNumber;
        int<lower=0> SampleTimes;
        vector[SampleNumber]  s;
        vector[SampleTimes] t;
        real<lower=0> y[SampleNumber,SampleTimes];
    }
    parameters {
        real<lower=0> phi0;
        real<lower=0> phi1;
        real<lower=0> phi2;
        real<lower=0> beta;
        real<lower=0> sigma;
    }
    transformed parameters {
        vector[SampleNumber]  a;
        for(i in 1:SampleNumber)
            a[i]=phi0*(s[i]^phi1)*exp(-phi2/s[i]);
    }
    model {
        phi0~normal(2,0.5);
        phi1~uniform(0,1);
        phi2~normal(20,2);
        beta~uniform(0,1);
        sigma~uniform(0,1);
        for(i in 1:SampleNumber){
            for(j in 3:SampleTimes)
                y[i][j]-y[i][j-1]~normal(a[i]*(exp(t[j]*beta)-exp(t[j-1]*beta)),sigma*(exp(t[j]*beta)-exp(t[j-1]*beta))^0.5);
        }
    }
    """

    #YY,TT,SS,SampleNumber,SampleTimes=WPMGenerateSingleMiMatrix()

    Data = pd.read_csv(path)
    #print(Data)
    TT=Data.values.tolist()[0]
    del(TT[0])
    #print(TT)
    YY_SS=Data.values.tolist()[1:len(Data.values.tolist())]
    SS=[x[0] for x in YY_SS]
    #print(SS)
    YY=[x[1:len(YY_SS[0])] for x in YY_SS]
    # print(YY)
    # print(len(YY[0]))
    SampleNumber=len(YY)
    SampleTimes=len(YY[0])
    #绘图而已
    YY1d=reduce(operator.add,YY)
    ex_TT=[]
    for i in range(0,SampleNumber):
        ex_TT=np.append(ex_TT,TT)
    #print(len(ex_TT))
    plt.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    single_exp_data = {"SampleNumber": SampleNumber,
                      "SampleTimes": SampleTimes,
                      "s": SS,
                      "t": TT,
                      "y": YY}
    sm = pystan.StanModel(model_code=single_exp_code)
    fit = sm.sampling(data=single_exp_data,chains=4, iter=1000)
    print("fit=",fit)
    all_parameters=fit.extract(permuted=True)
    print("all_parameters=",all_parameters)
    phi0=all_parameters['phi0']
    print("phi0=",phi0) #a的结果
    b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
    print("b=",b)
    arviz.plot_trace(fit)
    plt.show()
    print("model19:successful")

def WPMExpRegularRL(path):
    data=pd.read_csv(path,encoding='GBK')
    para=data.values.tolist()
    random.seed=1
    samplesCount=200
    II=random.sample(range(0,len(para)),samplesCount)
    T=[0 for t in range(0,samplesCount)]

    # print(len(para))
    # phi0=2
    # phi1=2.5
    # beta=0.8
    # sigma_eta=0.1
    for j in range(0,samplesCount):
        R=0.8
        delta_t=0.05
        D=5
        init_t=10

        a=para[II[j]][0]
        beta=para[II[j]][1]
        sigma=para[II[j]][2]

        t=init_t
        r=1
        while(r>R):
            t=t+delta_t;
            r=norm.cdf((D-a*exp(beta*t))/(sigma*exp(beta*t)**0.5))
            r1=exp((2*D*a/(sigma**2))*0.5)#否则指数会爆炸
            #print(r1)
            r2=norm.cdf((-D-a*exp(beta*t))/(sigma*exp(beta*t)**0.5))
            r=r-r1*r2
            r=r-r1*r2
            #print("t=",t,"r=",r)
        T[j]=t+0.5*delta_t
    #print(T)
    # mu=np.mean(T)
    # sigma_T=np.std(T)
    sns.distplot(T,kde=True)
    # y=mlab.normpdf(mu,sigma_T)
    # plt.plot(y,'--')
    # plt.show()
    #plt.hist(T,alpha=0.5)
    plt.xlabel("Reliability Lifetime")
    plt.ylabel("Kernel Density")
    plt.title('Distribution')
    #plt.legend()
    plt.show()

def WPMExpRegularMTTF(path):
    data=pd.read_csv(path,encoding='GBK')
    para=data.values.tolist()
    random.seed=1
    samplesCount=200
    II=random.sample(range(0,len(para)),samplesCount)
    T=[0 for t in range(0,samplesCount)]

    para=data.values.tolist()
    random.seed=1
    samplesCount=200
    II=random.sample(range(0,len(para)),samplesCount)
    MTTF=[0 for t in range(0,samplesCount)]

    # print(len(para))
    # phi0=2
    # phi1=2.5
    # beta=0.8
    # sigma_eta=0.1
    for j in range(0,samplesCount):
        delta_t=0.1
        D=5
        init_t=0

        a=para[II[j]][0]
        beta=para[II[j]][1]
        sigma=para[II[j]][2]
        M=3
        T=[0 for i in range(0,M)]
        y=0

        for i in range(0,M):
            y=0
            t=init_t
            t=t+delta_t
            delta=exp(beta*t)
            y=np.random.normal(a*delta,sigma*delta**0.5)
            while(y<D):
                delta=exp(beta*(t+delta_t))-exp(beta*t)
                t=t+delta_t;
                y=y+np.random.normal(a*delta,sigma*delta**0.5)
                #print("t=",t,"r=",r)
            T[i]=t
            print(i,T[i])
        MTTF[j]=(T[0]+T[1]+T[2])/3
    #print(T)
    # mu=np.mean(T)
    # sigma_T=np.std(T)
    print(MTTF)
    sns.distplot(MTTF,kde=True)
    # y=mlab.normpdf(mu,sigma_T)
    # plt.plot(y,'--')
    # plt.show()
    #plt.hist(T,alpha=0.5)
    plt.xlabel("Mean Time To Failure")
    plt.ylabel("Kernel Density")
    plt.title('Distribution')
    #plt.legend()
    plt.show()

def WPMExpRegularRUL(path):
    data=pd.read_csv(r'E:\data\无应力指数尺度维纳过程模型参数估计结果.csv',encoding='GBK')
    para=data.values.tolist()
    random.seed=1
    samplesCount=200
    II=random.sample(range(0,len(para)),samplesCount)
    T=[0 for t in range(0,samplesCount)]

    para=data.values.tolist()
    random.seed=1
    samplesCount=200
    II=random.sample(range(0,len(para)),samplesCount)
    RUL=[0 for t in range(0,samplesCount)]

    # print(len(para))
    # phi0=2
    # phi1=2.5
    # beta=0.8
    # sigma_eta=0.1
    for j in range(0,samplesCount):
        delta_t=0.1
        D=5
        init_t=5
        init_y=1.5

        a=para[II[j]][0]
        beta=para[II[j]][1]
        sigma=para[II[j]][2]
        M=3
        T=[0 for i in range(0,M)]
        y=0

        for i in range(0,M):
            y=init_y
            t=init_t
            t=t+delta_t
            delta=exp(beta*t)
            y=np.random.normal(a*delta,sigma*delta**0.5)
            while(y<D):
                delta=exp(beta*(t+delta_t))-exp(beta*t)
                t=t+delta_t;
                y=y+np.random.normal(a*delta,sigma*delta**0.5)
                #print("t=",t,"r=",r)
            T[i]=t
            if(T[i]==init_t+delta_t): T[i]=init_t
            #print(i,T[i])
        RUL[j]=(T[0]+T[1]+T[2])/3-init_t
    #print(T)
    # mu=np.mean(T)
    # sigma_T=np.std(T)
    #print(RUL)
    sns.distplot(RUL,kde=True)
    # y=mlab.normpdf(mu,sigma_T)
    # plt.plot(y,'--')
    # plt.show()
    #plt.hist(T,alpha=0.5)
    plt.xlabel("Remaining Useful Lifetime")
    plt.ylabel("Kernel Density")
    plt.title('Distribution')
    #mpl.rcParams['font.sans-serif'] = ['SimHei']
    # plt.xlabel(U"剩余寿命")
    # plt.ylabel(U"核密度")
    # plt.title(U'分布')
    #plt.legend()
    plt.show()

if __name__ == '__main__':
    #WPMLogRegular(r'E:\data\无应力对数尺度维纳过程模型.csv')
    #WPMGenerateRegularLogMatrix()
    #WPMGenerateSingleLogMatrix()
    #WPMLogSingle(r'E:\data\单应力对数尺度艾琳维纳过程模型.csv')

    #WPMGenerateRegularMiMatrix()
    #WPMMiRegular(r'E:\data\无应力幂律尺度维纳过程模型.csv')
    #WPMGenerateSingleMiMatrix()
    #WPMMiSingle(r'E:\data\单应力幂律尺度艾琳维纳过程模型.csv')

    #WPMGenerateRegularExpMatrix()
    #WPMExpRegular(r'E:\data\无应力指数尺度维纳过程模型.csv')

    #WPMGenerateSingleExpMatrix()
    #WPMExpSingle(r'E:\data\单应力指数尺度艾琳维纳过程模型.csv')

    WPMExpRegularRL(r'E:\data\无应力指数尺度维纳过程模型参数估计结果.csv')
    #WPMExpRegularMTTF(r'E:\data\无应力指数尺度维纳过程模型参数估计结果.csv')
    #WPMExpRegularRUL(r'E:\data\无应力指数尺度维纳过程模型参数估计结果.csv')



    #print(exp(710))


    # Data = pd.read_csv(r'E:\data\单应力对数尺度艾琳维纳过程模型.csv')
    # print(Data)
    # TT=Data.values.tolist()[0]
    # del(TT[0])
    # print(TT)
    # YY_SS=Data.values.tolist()[1:len(Data.values.tolist())]
    # SS=[x[0] for x in YY_SS]
    # print(SS)
    # YY=[x[1:len(YY_SS[0])] for x in YY_SS]
    # print(YY)
    # print(len(YY[0]))

    print("end")
