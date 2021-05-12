# -*- coding:utf-8 -*-
import pystan
import matplotlib.pyplot as plt
import arviz
import numpy as np
# from numpy.ma import log, exp,sqrt
from math import log, exp,sqrt
import pandas as pd
import csv
from scipy.stats import norm
import random
import matplotlib.mlab as mlab
import seaborn as sns
# from pylab import *



def read_regular_data():
    #读取csv数据
    regular_data=pd.read_csv(r'E:\data\regular.csv',encoding='GBK')
    print(regular_data)
    Number=len(regular_data)
    Y_test=regular_data.values #退化值
    #print(Y_test)

    #获取表格上的时间
    T_test=regular_data.columns.values
    for i in range(0,len(T_test)):
        T_test[i]=int(T_test[i])
    #print(T_test)
    T_extension=T_test

    for i in range(0,Number-1):
        T_extension=np.append(T_extension,T_test)
        #print(T_extension)
    #print(T_extension)
    T_extension=T_extension.tolist()
    #numpy.ndarray转list
    Y_test=[i for j in Y_test for i in j]
    #print(Y_test)
    #print(len(Y_test))
    #print(len(T_extension))
    print("---data finished---")
    return Y_test,T_extension

def read_regular_Mi():
    #读取csv数据
    regular_data=pd.read_csv(r'E:\data\regularMi_1.csv',encoding='GBK')
    print(regular_data)
    Number=len(regular_data)
    Y_test=regular_data.values #退化值
    #print(Y_test)

    #获取表格上的时间
    T_test=regular_data.columns.values
    for i in range(0,len(T_test)):
        T_test[i]=round(float(T_test[i]))
    #print(T_test)
    T_extension=T_test

    for i in range(0,Number-1):
        T_extension=np.append(T_extension,T_test)
        #print(T_extension)
    #print(T_extension)
    T_extension=T_extension.tolist()
    #numpy.ndarray转list
    Y_test=[i for j in Y_test for i in j]
    print(Y_test)
    print(len(Y_test))
    print(len(T_extension))
    print("---data finished---")
    return Y_test,T_extension

def read_single_data():
    # 读取csv数据
    single_data=pd.read_csv(r'E:\data\single.csv',encoding='GBK')
    print("单应力\n",single_data)
    Number=len(single_data)-1
    # print(Number)
    Y_test=single_data.values #退化值np.ndarray
    #print(Y_test)
    #获取表格中的应力
    s=Y_test[0][0]
    #print(s)
    Y_test=np.delete(Y_test,0,axis=0)
    #print(Y_test)
    Y_test=[i for j in Y_test for i in j]
    #print(Y_test)
    #print(len(Y_test))

    #获取表格上的时间
    T_test=single_data.columns.values
    for i in range(0,len(T_test)):
        T_test[i]=int(T_test[i])
    #print(T_test)
    T_extension=T_test
    for i in range(0,Number-1):
        T_extension=np.append(T_extension,T_test)
        #print(T_extension)
    #print(T_extension)
    #numpy.ndarray转list
    T_extension=T_extension.tolist()
    #print(T_extension)
    #print(len(Y_test))
    #print(len(T_extension))
    #print(len(Y_test))
    print("---single data finished---")
    return Y_test,T_extension,s



def model01(path):
    #无应力对数尺度
    regular_log_code="""
    data { 
        int<lower=0> N;
        vector[N] t;
        vector[N] y;
    }
    parameters {
        real a;
        real<lower=0> beta;
        real<lower=0> sigmasq_eta;
    }
    transformed parameters {
        real<lower=0> sigma_eta;
        sigma_eta=sqrt(sigmasq_eta);
    }
    model {
        vector[N] ypred;
        a~uniform(0,20);
        beta~uniform(0,10);
        for(i in 1:N)
            ypred[i]=a*log(beta*t[i]);
        y ~ normal(ypred, sigma_eta);
    }
    """

    data=pd.read_csv(path,encoding='GBK',header=None)
    TT=data.values.tolist()[0]
    YY=data.values.tolist()[1]
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    regular_log_data = {"N": len(YY),
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
    print("model01:successful")


def model02():
    read_single_data()
    single_arrhenius_log_code="""
    data { 
        int<lower=0> N;
        real<lower=0> s;
        vector[N] t;
        vector[N] y;
    }
    parameters {
        real<lower=0> phi0;
        real phi1;
        real<lower=0> beta;
        real<lower=0> sigmasq_eta;
    }
    transformed parameters {
        real<lower=0> sigma_eta;
        sigma_eta=sqrt(sigmasq_eta);
    }
    model {
        vector[N] ypred;
        phi0~uniform(0,10);
        phi1~uniform(0,20);
        beta~uniform(0,0.5);
        for(i in 1:N)
            ypred[i]=phi0*exp(-phi1/s)*log(beta*t[i]);
        y ~ normal(ypred, sigma_eta);
    }
    """
    #phi0=1 phi2=5 beta=0.1
    YY,TT,SS=read_single_data()
    SS=float(SS)
    print(type(SS))
    # err=np.random.normal(0,0.01,size=len(TT)).tolist()
    # print(YY)
    # plt.scatter(TT, YY,c='g',linewidths=0.1)
    # print(type(TT))
    # print(SS)
    # print(err)
    # for i in range(0,len(TT)):
    #     YY[i]=1*exp(-5/60)*log(0.1*TT[i])+err[i]
    # print(YY)
    # print("生成数据")

    plt.scatter(TT, YY,c='blue',linewidths=0.01,alpha=0.5)
    plt.title('one constant-stress accelerating scenario')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()


    single_arrhenius_log_data = {"N": len(YY),
                        "s": SS,
                        "t": TT,
                        "y":YY
                        }
    sm = pystan.StanModel(model_code=single_arrhenius_log_code)
    fit = sm.sampling(data=single_arrhenius_log_data,chains=4, iter=1000)
    print("fit=",fit)
    all_parameters=fit.extract(permuted=True)
    print("all_parameters=",all_parameters)
    phi0=all_parameters['phi0']
    print("phi0=",phi0) #a的结果
    b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
    print("b=",b)
    arviz.plot_trace(fit)
    #,pars={"phi0","phi1","beta","sigma_eta","sigmasq_eta"}
    plt.show()
    print("model02:successful")


def model03(path):
    #read_single_data()
    single_arrhenius_log_code="""
    data { 
        int<lower=0> N;
        vector<lower=0>[N] s;
        vector<lower=10>[N] t;
        vector[N] y;
    }
    parameters {
        real<lower=0> phi0;
        real phi1;
        real<lower=0> beta;
        real<lower=0> sigmasq_eta;
    }
    transformed parameters {
        real<lower=0> sigma_eta;
        sigma_eta=sqrt(sigmasq_eta);
    }
    model {
        vector[N] ypred;
        phi0~uniform(0,5);
        phi1~uniform(0,5);
        beta~uniform(0.1,5);
        for(i in 1:N)
            ypred[i]=phi0*exp(-phi1/s[i])*log(beta*t[i]);
        y ~ normal(ypred, sigma_eta);
    }
    """
    data=pd.read_csv(path,encoding='GBK',header=None)
    TT=data.values.tolist()[0]
    SS=data.values.tolist()[1]
    YY=data.values.tolist()[2]
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    single_arrhenius_log_data = {"N": len(YY),
                                 "s": SS,
                                 "t": TT,
                                 "y":YY
                                 }
    sm = pystan.StanModel(model_code=single_arrhenius_log_code)
    fit = sm.sampling(data=single_arrhenius_log_data,chains=4, iter=1000)
    print("fit=",fit)
    all_parameters=fit.extract(permuted=True)
    print("all_parameters=",all_parameters)
    phi0=all_parameters['phi0']
    print("phi0=",phi0) #a的结果
    b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
    print("b=",b)
    arviz.plot_trace(fit)
    #,pars={"phi0","phi1","beta","sigma_eta","sigmasq_eta"}
    plt.show()
    #保存参数估计结果
    # cc=["phi0","phi1","beta","sigmasq_eta","sigma_eta","lp__"]
    # # print(b[0])
    # # print(len(b[0]))
    # bb=np.array(b).reshape(2000,6)
    # print(bb)
    # content=pd.DataFrame(columns=cc,data=bb)
    # print(content)
    # content.to_csv(r'E:\data\单应力对数尺度阿伦尼斯通用模型参数估计结果.csv',index=False)
    print("model03:successful")


def model05(path):
    #无应力指数尺度
    regular_exp_code="""
    data { 
        int<lower=0> N;
        vector[N] t;
        vector[N] y;
    }
    parameters {
        real<lower=0> a;
        real<lower=0> beta;
        real<lower=0> sigmasq_eta;
    }
    transformed parameters {
        real<lower=0> sigma_eta;
        sigma_eta=sqrt(sigmasq_eta);
    }
    model {
        vector[N] ypred;
        a~uniform(0,10);
        beta~uniform(0,1);
        ypred=a*exp(beta*t);
        y ~ lognormal(log(ypred), sigma_eta);
    }
    """
    data=pd.read_csv(path,encoding='GBK',header=None)
    TT=data.values.tolist()[0]
    YY=data.values.tolist()[1]
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    #YY,TT=GenerateRegularExp()

    regular_exp_data = {"N": len(YY),
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
    print("model05:successful")

def model06(path):
    #无应力指数尺度
    single_exp_code=single_arrhenius_log_code="""
    data { 
        int<lower=0> N;
        vector<lower=0>[N] s;
        vector<lower=0>[N] t;
        vector[N] y;
    }
    parameters {
        real<lower=0> phi0;
        real phi1;
        real<lower=0> beta;
        real<lower=0> sigmasq_eta;
    }
    transformed parameters {
        real<lower=0> sigma_eta;
        sigma_eta=sqrt(sigmasq_eta);
    }
    model {
        vector[N] ypred;
        phi0~normal(2,1);
        phi1~normal(13,3);
        beta~uniform(0,1);
        for(i in 1:N)
            ypred[i]=phi0*exp(-phi1/s[i])*exp(beta*t[i]);
        y ~ normal(ypred, sigma_eta);
    }
    """
    data=pd.read_csv(path,encoding='GBK',header=None)
    TT=data.values.tolist()[0]
    SS=data.values.tolist()[1]
    YY=data.values.tolist()[2]
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    #YY,TT,SS=GenerateSingleExp()

    single_exp_data = {"N": len(YY),
                       "s": SS,
                       "t": TT,
                       "y":YY
                       }
    sm = pystan.StanModel(model_code=single_exp_code)
    fit = sm.sampling(data=single_exp_data,chains=4, iter=1000)
    print("fit=",fit)
    all_parameters=fit.extract(permuted=True)
    print("all_parameters=",all_parameters)
    a=all_parameters['phi0']
    print("phi0=",a) #a的结果
    b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
    print("b=",b)
    arviz.plot_trace(fit)
    plt.show()
    print("model06:successful")


def GenerateRegularLog():
    oriTT=[t*10 for t in range(1,31)]
    TT=[]
    for i in range(0,10):
        TT=np.append(TT,oriTT)
    print(len(TT))
    YY=[0*t for t in range(0,len(TT))]
    err=np.random.normal(0,0.1,size=len(TT)).tolist()
    for i in range(0,len(TT)):
        if 0.5*TT[i]<=1:#否则会小于0
            YY[i]=abs(err[i])
        else:
            YY[i]=3*log(0.5*TT[i])+err[i]
        print(YY[i])
    print("生成数据")
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.3)
    plt.title('Degradation')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("Generated Regular Log degradation data successfully")
    np.savetxt(r'E:\data\无应力对数尺度通用退化轨迹模型.csv',[TT,YY],delimiter=',')
    return YY,TT

def GenerateSingleLog():
    oriTT=[t*10 for t in range(1,31)]
    TT=[]

    for i in range(0,10):
        TT=np.append(TT,oriTT)
    print(len(TT))
    YY=[0*t for t in range(0,len(TT))]
    SS=[60 for i in range(0,len(TT))]
    err=np.random.normal(0,0.1,size=len(TT)).tolist()
    for i in range(0,len(TT)):
        if 0.5*TT[i]<=1:#否则会小于0
            YY[i]=abs(err[i])
        else:
            #ypred[i]=phi0*exp(-phi1/s)*log(beta*t[i]);
            YY[i]=2*exp(-0.5/SS[i])*log(0.8*TT[i])+err[i]
        print(YY[i])
    print("生成数据")
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("Generated Regular Log degradation data successfully")
    #np.savetxt(r'E:\data\单应力对数尺度阿伦尼斯通用轨迹模型.csv',[TT,SS,YY],delimiter=',')
    return YY,TT,SS

def GenerateRegularExp():
    oriTT=[t*1 for t in range(0,31)]
    TT=[]
    for i in range(0,10):
        TT=np.append(TT,oriTT)
    print(len(TT))
    YY=[0*t for t in range(0,len(TT))]
    err=np.random.normal(0,0.3,size=len(TT)).tolist()
    for i in range(0,len(TT)):
        YY[i]=1.5*exp(0.1*TT[i])+err[i]
        print(YY[i])
    print("生成数据")
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.5)
    plt.title('Degradation Data')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("Generated Regular Exp degradation data successfully")
    np.savetxt(r'E:\data\无应力指数尺度通用退化轨迹模型.csv',[TT,YY],delimiter=',')
    return YY,TT

def GenerateSingleExp():
    oriTT=[t*1 for t in range(0,31)]
    TT=[]

    for i in range(0,10):
        TT=np.append(TT,oriTT)
    print(len(TT))
    YY=[0*t for t in range(0,len(TT))]
    SS=[20 for i in range(0,len(TT))]
    err=np.random.normal(0,0.1,size=len(TT)).tolist()
    for i in range(0,len(TT)):
        #YY[i]=2*(TT[i]**0.5)+err[i]
        YY[i]=2*exp(-13/SS[i])*exp(0.1*TT[i])+err[i]
        if(YY[i]<0):
            YY[i]=abs(err[i])
        #print(YY[i])
        #ypred[i]=phi0*exp(-phi1/s)*log(beta*t[i]);

        print(YY[i])
    print("生成数据")
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("Generated Single Exp degradation data successfully")
    np.savetxt(r'E:\data\单应力指数尺度阿伦尼斯通用轨迹模型.csv',[TT,SS,YY],delimiter=',')
    return YY,TT,SS


def GenerateRegularMi():
    oriTT=[t*10 for t in range(0,31)]
    TT=[]
    for i in range(0,10):
        TT=np.append(TT,oriTT)
    print(len(TT))
    YY=[0*t for t in range(0,len(TT))]
    err=np.random.normal(0,0.02,size=len(TT)).tolist()
    for i in range(0,len(TT)):
        #YY[i]=2*(TT[i]**0.5)+err[i]
        YY[i]=0.1*(TT[i]**0.5)+err[i]
        if(YY[i]<0):
            YY[i]=0
        #print(YY[i])
    print("生成数据")
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.5)
    plt.title('Degradation Data')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("Generated Regular Mi degradation data successfully")
    # matrix=[oriTT]
    # tempYY=[]
    # for i in range(0,int(len(TT)/len(oriTT))):
    #     tempYY=[]
    #     for s in range(0,len(oriTT)):
    #         tempYY.append(YY[i*len(oriTT)+s])
    #     matrix.append(tempYY)
    #     #print(matrix)
    # print(matrix)
    #np.savetxt(r'E:\data\regularMi_1.csv',matrix,delimiter=',')
    np.savetxt(r'E:\data\无应力幂律尺度通用退化轨迹模型.csv',[TT,YY],delimiter=',')
    return YY,TT

def GenerateSingleMi():
    oriTT=[t*10 for t in range(0,31)]
    TT=[]

    for i in range(0,10):
        TT=np.append(TT,oriTT)
    print(len(TT))
    YY=[0*t for t in range(0,len(TT))]
    SS=[20 for i in range(0,len(TT))]
    err=np.random.normal(0,0.05,size=len(TT)).tolist()
    for i in range(0,len(TT)):
        #YY[i]=2*(TT[i]**0.5)+err[i]
        YY[i]=2*exp(-13/SS[i])*(TT[i]**0.5)+err[i]
        if(YY[i]<0):
            YY[i]=abs(err[i])
        #print(YY[i])
            #ypred[i]=phi0*exp(-phi1/s)*log(beta*t[i]);

        print(YY[i])
    print("生成数据")
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()
    print("Generated Single Mi degradation data successfully")
    #np.savetxt(r'E:\data\单应力幂律尺度阿伦尼斯通用轨迹模型.csv',[TT,SS,YY],delimiter=',')
    return YY,TT,SS



def model09(path):
    #无应力幂指尺度
    regular_mi_code="""
    data { 
        int<lower=0> N;
        vector<lower=0>[N] t;
        vector[N] y;
    }
    parameters {
        real<lower=0> a;
        real<lower=0> beta;
        real<lower=0> sigmasq_eta;
    }
    transformed parameters {
        real<lower=0> sigma_eta;
        sigma_eta=sqrt(sigmasq_eta);
    }
    model {
        vector[N] ypred;
        a~uniform(0,10);
        beta~uniform(0,5);
        for(i in 1:N)
            ypred[i]=a*(t[i]^beta);
        y ~ normal(ypred, sigma_eta);
    }
    """

    data=pd.read_csv(path,encoding='GBK',header=None)
    TT=data.values.tolist()[0]
    YY=data.values.tolist()[1]
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    regular_exp_data = {"N": len(YY),
                        "t": TT,
                        "y": YY}
    sm = pystan.StanModel(model_code=regular_mi_code)
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
    print("model09:successful")



def model10(path):
    #read_single_data()
    single_arrhenius_mi_code="""
    data { 
        int<lower=0> N;
        vector<lower=0>[N] s;
        vector<lower=0>[N] t;
        vector[N] y;
    }
    parameters {
        real<lower=0> phi0;
        real<lower=0> phi1;
        real<lower=0> beta;
        real<lower=0> sigmasq_eta;
    }
    transformed parameters {
        real<lower=0> sigma_eta;
        sigma_eta=sqrt(sigmasq_eta);

    }
    model {
        vector[N] ypred;
        phi0~normal(2,0.5);
        phi1~normal(13,3);
        beta~normal(0.5,0.1);
        for(i in 1:N)
            ypred[i]=phi0*exp(-phi1/s[i])*(t[i]^beta);
        y ~ normal(ypred, sigma_eta);
    }
    """
    data=pd.read_csv(path,encoding='GBK',header=None)
    TT=data.values.tolist()[0]
    SS=data.values.tolist()[1]
    YY=data.values.tolist()[2]
    plt.scatter(TT, YY,c='blue',s=1,alpha=0.3)
    plt.title('Degradation Diagram')#显示图表标题
    plt.xlabel('time')#x轴名称
    plt.ylabel('degradation data')#y轴名称
    plt.show()

    single_arrhenius_mi_data = {"N": len(YY),
                                 "s": SS,
                                 "t": TT,
                                 "y":YY
                                 }
    sm = pystan.StanModel(model_code=single_arrhenius_mi_code)
    fit = sm.sampling(data=single_arrhenius_mi_data,chains=4, iter=1000)
    print("fit=",fit)
    all_parameters=fit.extract(permuted=True)
    print("all_parameters=",all_parameters)
    phi0=all_parameters['phi0']
    print("phi0=",phi0) #a的结果
    b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
    print("b=",b)
    arviz.plot_trace(fit)
    #,pars={"phi0","phi1","beta","sigma_eta","sigmasq_eta"}
    plt.show()
    print("model10:successful")

def GPMArrenLogRL(path):
    data=pd.read_csv(path,encoding='GBK')
    para=data.values.tolist()
    random.seed=1
    #random.seed(1)
    samplesCount=200
    II=random.sample(range(0,len(para)),samplesCount)
    #II=random.sample(range(0,len(para)),samplesCount)
    T=[0 for t in range(0,samplesCount)]

    # print(len(para))
    # phi0=2
    # phi1=2.5
    # beta=0.8
    # sigma_eta=0.1
    for j in range(0,samplesCount):
        R=0.8
        delta_t=0.05
        D=8
        init_t=50

        phi0=para[II[j]][0]
        phi1=para[II[j]][1]
        beta=para[II[j]][2]
        sigma_eta=para[II[j]][4]

        t=init_t
        r=1
        s=60
        while(r>R):
            t=t+delta_t;
            r=1-norm.cdf((phi0*exp(-phi1/s)*log(beta*t)-D)/sigma_eta)
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
    # plt.xlabel(U"可靠度寿命")
    # plt.ylabel(U"核密度")
    # plt.title(U'分布')
    #plt.legend()
    plt.show()

def GPMArrenLogMTTF(path):
    data=pd.read_csv(path,encoding='GBK')
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
        D=8
        init_t=2

        phi0=para[II[j]][0]
        phi1=para[II[j]][1]
        beta=para[II[j]][2]
        sigma_eta=para[II[j]][4]

        t=init_t
        r=1
        s=60
        M=3
        T=[0 for i in range(0,M)]

        for i in range(0,M):
            y=0
            t=init_t
            while(y<D):
                t=t+delta_t;
                y=phi0*exp(-phi1/s)*log(beta*t)+np.random.normal(0,sigma_eta)
                #print("t=",t,"r=",r)
            T[i]=t
            #print(i,T[i])
        MTTF[j]=(T[0]+T[1]+T[2])/3
    #print(T)
    # mu=np.mean(T)
    # sigma_T=np.std(T)
    #print(MTTF)
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

def GPMArrenLogRUL(path):
    data=pd.read_csv(path,encoding='GBK')
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
        D=8
        init_t=50
        init_y=7

        phi0=para[II[j]][0]
        phi1=para[II[j]][1]
        beta=para[II[j]][2]
        sigma_eta=para[II[j]][4]

        t=init_t
        r=1
        s=60
        M=3
        T=[0 for i in range(0,M)]

        for i in range(0,M):
            y=init_y
            t=init_t
            while(y<D):
                t=t+delta_t;
                y=phi0*exp(-phi1/s)*log(beta*t)+np.random.normal(0,sigma_eta)
                #print("t=",t,"r=",r)
            T[i]=t
            #print(i,T[i])
        RUL[j]=(T[0]+T[1]+T[2])/3-init_t
    #print(T)
    # mu=np.mean(T)
    # sigma_T=np.std(T)
    #print(MTTF)
    sns.distplot(RUL,kde=True)
    # y=mlab.normpdf(mu,sigma_T)
    # plt.plot(y,'--')
    # plt.show()
    #plt.hist(T,alpha=0.5)
    plt.xlabel("Remaining Useful Lifeime")
    plt.ylabel("Kernel Density")
    plt.title('Distribution')
    #plt.legend()
    plt.show()

if __name__ == '__main__':
    #model01(r'E:\data\无应力对数尺度通用退化轨迹模型.csv')
    #model03(r'E:\data\单应力对数尺度阿伦尼斯通用轨迹模型.csv')
    #model09(r'E:\data\无应力幂律尺度通用退化轨迹模型.csv')
    #model10(r'E:\data\单应力幂律尺度阿伦尼斯通用轨迹模型.csv')
    #model05(r'E:\data\无应力指数尺度通用退化轨迹模型.csv')
    #model06(r'E:\data\单应力指数尺度阿伦尼斯通用轨迹模型.csv')
    #GenerateRegularLog()
    #GenerateSingleLog()
    #GenerateSingleMi()
    #GenerateRegularMi()
    #GenerateRegularExp()
    #GenerateSingleExp()

    # b=[[[1,3,2],[1,3,4]],[[2,5,6],[2,7,8]],[[3,5,6],[3,7,8]]]
    # print(len(b))
    # print(len(b[0]))
    # print(len(b[0][0]))
    # print(b[0]+b[1]+b[2])
    # print(len(b[0]))
    # print(np.array(b).reshape(len(b)*len(b[0]),3))


    #read_regular_Mi()

    # data=pd.read_csv(r'E:\data\无应力对数尺度通用退化轨迹模型.csv',encoding='GBK',header=None)
    # TT=data.values.tolist()[0]
    # YY=data.values.tolist()[1]
    # print(TT)
    # print(YY)
    #mpl.rcParams['font.sans-serif'] = ['SimHei']
    #GPMArrenLogRL(r'E:\data\单应力对数尺度阿伦尼斯通用模型参数估计结果.csv')


    GPMArrenLogMTTF(r'E:\bs_data\单应力对数尺度阿伦尼斯通用轨迹模型_result.csv')

    #GPMArrenLogRUL(r'E:\data\单应力对数尺度阿伦尼斯通用模型参数估计结果.csv')

    # regular_data=pd.read_csv(r'E:\data\regular.csv',encoding='GBK')
    # print(regular_data)
    # Number=len(regular_data)
    # Y_test=regular_data.values #退化值
    # #print(Y_test)
    #
    # #获取表格上的时间
    # T_test=regular_data.columns.values
    # for i in range(0,len(T_test)):
    #     T_test[i]=round(float(T_test[i]))
    # #print(T_test)
    # T_extension=T_test
    #
    # for i in range(0,Number-1):
    #     T_extension=np.append(T_extension,T_test)
    #     #print(T_extension)
    # #print(T_extension)
    # T_extension=T_extension.tolist()
    # #numpy.ndarray转list
    # Y_test=[i for j in Y_test for i in j]
    # print(Y_test)
    # print(len(Y_test))
    # print(len(T_extension))
    # plt.scatter(T_extension, Y_test,c='red',alpha=0.5)

    #plt.rcParams['font.family']=['SimHei']
    # plt.title(U'退化数据分布')#显示图表标题
    # plt.xlabel(U'时间')#x轴名称
    # plt.ylabel(U'退化值') #y轴名称
    # plt.title('Distribution')#显示图表标题
    # plt.xlabel('time')#x轴名称
    # plt.ylabel('Degradation Data') #y轴名称
    #
    # plt.show()
    # print("Generated Regular Mi degradation data successfully")
    # print("---data finished---")



    # time=[t*0.1 for t in range(5,200,1)]
    # FF=[norm.cdf((phi0*exp(-phi1/s)*log(beta*t)-D)/sigma_eta) for t in time]
    # print(FF)
    # #YY=[(phi0*exp(-phi1/s)*log(beta*t)-D)/sigma_eta for t in time]
    # plt.scatter(time, FF,c='blue',s=1,alpha=0.3)
    # #plt.scatter(time, YY,c='r',s=1,alpha=0.3)
    # plt.show()

    print("end")


