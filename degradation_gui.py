# -*- coding:utf-8 -*-
from tkinter import *
from PIL import Image,ImageTk
import pystan
import matplotlib
import matplotlib.pyplot as plt
import arviz
import numpy as np
import operator
from functools import reduce
# from numpy.ma import log, exp,sqrt
from math import log, exp,sqrt
import pandas as pd
import csv
from scipy.stats import norm
import random
import matplotlib.mlab as mlab
import seaborn as sns
import tkinter.messagebox
import matplotlib
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
matplotlib.use('TkAgg')

from tkinter import filedialog

def openData():

    file_open_path=filedialog.askopenfilename()
    print(file_open_path)
    #画布大小和分辨率
    f=Figure(figsize=(4,4),dpi=100)
    #利用子图绘图
    a = f.add_subplot(111)
    print(var_allSpace.get())
    if var_allSpace.get()==1:
        #r'E:\data\单应力对数尺度阿伦尼斯通用轨迹模型.csv'
        data=pd.read_csv(file_open_path,encoding='GBK',header=None)
        TT=data.values.tolist()[0]
        SS=data.values.tolist()[1]
        YY=data.values.tolist()[2]
        a.scatter(TT, YY,c='blue',s=1,alpha=0.3)
        a.set_title('Degradation Diagram')#显示图表标题
        a.set_xlabel('time')#x轴名称
        a.set_ylabel('degradation data')#y轴名称

    if var_allSpace.get()==2:
        #r'E:\data\无应力指数尺度维纳过程模型.csv'
        Data = pd.read_csv(file_open_path)
        #print(Data)
        TT=Data.values.tolist()[0]
        YY=Data.values.tolist()[1:len(Data.values.tolist())]
        SampleNumber=len(YY)
        SampleTimes=len(YY[0])
        #绘图而已
        YY1d=reduce(operator.add,YY)
        ex_TT=[]
        for i in range(0,SampleNumber):
            ex_TT=np.append(ex_TT,TT)

        a.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
        a.set_title('Degradation Diagram')#显示图表标题
        a.set_xlabel('time')#x轴名称
        a.set_ylabel('degradation data')#y轴名称

    #创建画布控件
    cavas1=FigureCanvasTkAgg(f,master=window)
    cavas1.draw()
    #显示画布空间
    cavas1.get_tk_widget().place(x=30,y=100)

def estimateData():
    #tkinter.messagebox.showinfo("info","open data file")

    file_open_path=filedialog.askopenfilename()
    print(file_open_path)
    if var_allSpace.get()==1:
        #file_open_path=r'E:\data\单应力对数尺度阿伦尼斯通用轨迹模型.csv'
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
        data=pd.read_csv(file_open_path,encoding='GBK',header=None)
        TT=data.values.tolist()[0]
        SS=data.values.tolist()[1]
        YY=data.values.tolist()[2]

        single_arrhenius_log_data = {"N": len(YY),
                                     "s": SS,
                                     "t": TT,
                                     "y":YY
                                     }
        sm = pystan.StanModel(model_code=single_arrhenius_log_code)
        fit = sm.sampling(data=single_arrhenius_log_data,chains=4, iter=1000)
        print("fit=",fit)
        all_parameters=fit.extract(permuted=True)
        b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
        cc=["phi0","phi1","beta","sigmasq_eta","sigma_eta","lp__"]
        bb=np.array(b).reshape(2000,6)
        content=pd.DataFrame(columns=cc,data=bb)
        content.to_csv(file_open_path.replace('.csv','_result.csv'),index=False)
        print("single Arrenis:successful")
        tkinter.messagebox.showinfo("info","参数估计结果已保存至同一文件夹下")

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
    if var_allSpace.get()==2:
        #YY,TT,SampleNumber,SampleTimes=WPMGenerateRegularExpMatrix()
        Data = pd.read_csv(file_open_path)
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

        regular_exp_data = {"SampleNumber": SampleNumber,
                            "SampleTimes":SampleTimes,
                            "t": TT,
                            "y": YY}
        sm = pystan.StanModel(model_code=regular_exp_code)
        fit = sm.sampling(data=regular_exp_data,chains=4, iter=1000)
        print("fit=",fit)
        all_parameters=fit.extract(permuted=True)
        # print("all_parameters=",all_parameters)
        #a=all_parameters['a']
        # print("a=",a) #a的结果
        b=fit.extract(permuted=False) #b是未排序的参数，用于绘图
        # print("b=",b)
        # arviz.plot_trace(fit)
        # plt.show()
        # #保存参数估计结果
        cc=["a","beta","sigma","lp__"]
        # print(b[0])
        #print(len(b[0]))
        bb=np.array(b).reshape(len(b)*len(b[0]),4)
        #print(bb)
        content=pd.DataFrame(columns=cc,data=bb)
        #print(content)
        content.to_csv(file_open_path.replace('.csv','_result.csv'),index=False)
        print("no-stress exp wiener:successful")
        tkinter.messagebox.showinfo("info","参数估计结果已保存至同一文件夹下")

def estimateRL():
    file_open_path=filedialog.askopenfilename()
    #print(file_open_path)
    # r'E:\data\单应力对数尺度阿伦尼斯通用模型参数估计结果.csv'
    if var_allSpace.get()==1:
        data=pd.read_csv(file_open_path,encoding='GBK')
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
        #画布大小和分辨率
        f2=Figure(figsize=(5,4),dpi=100)
        #利用子图绘图
        b = f2.subplots()

        sns.distplot(T,kde=True,ax=b)
        # y=mlab.normpdf(mu,sigma_T)
        # plt.plot(y,'--')
        # plt.show()
        #plt.hist(T,alpha=0.5)
        # plt.xlabel("Reliability Lifetime")
        # plt.ylabel("Kernel Density")
        # plt.title('Distribution')
        # plt.show()
        # a.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
        b.set_title('Distribution')#显示图表标题
        b.set_xlabel('Reliability Lifetime')#x轴名称
        b.set_ylabel('Kernel Density')#y轴名称
    elif var_allSpace.get()==2:
        data=pd.read_csv(file_open_path,encoding='GBK')
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

        f2=Figure(figsize=(5,4),dpi=100)
        #利用子图绘图
        b = f2.subplots()
        sns.distplot(T,kde=True,ax=b)
        b.set_title('Distribution')#显示图表标题
        b.set_xlabel('Reliability Lifetime')#x轴名称
        b.set_ylabel('Kernel Density')#y轴名称

    #创建画布控件
    cavas2=FigureCanvasTkAgg(f2,master=window)
    cavas2.draw()
    #显示画布空间
    cavas2.get_tk_widget().place(x=460,y=100)

def estimateMTTF():
    file_open_path=filedialog.askopenfilename()
    #print(file_open_path)
    # r'E:\data\单应力对数尺度阿伦尼斯通用模型参数估计结果.csv'
    if var_allSpace.get()==1:
        data=pd.read_csv(file_open_path,encoding='GBK')
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
        #sns.distplot(MTTF,kde=True)
        # y=mlab.normpdf(mu,sigma_T)
        # plt.plot(y,'--')
        # plt.show()
        #plt.hist(T,alpha=0.5)
        #画布大小和分辨率
        f2=Figure(figsize=(5,4),dpi=100)
        #利用子图绘图
        b = f2.subplots()

        sns.distplot(MTTF,kde=True,ax=b)
        # y=mlab.normpdf(mu,sigma_T)
        # plt.plot(y,'--')
        # plt.show()
        #plt.hist(T,alpha=0.5)
        # plt.xlabel("Reliability Lifetime")
        # plt.ylabel("Kernel Density")
        # plt.title('Distribution')
        # plt.show()
        # a.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
        b.set_title('Distribution')#显示图表标题
        b.set_xlabel('Mean Time To Failure')#x轴名称
        b.set_ylabel('Kernel Density')#y轴名称
    elif var_allSpace.get()==2:
        data=pd.read_csv(file_open_path,encoding='GBK')
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

        f2=Figure(figsize=(5,4),dpi=100)
        #利用子图绘图
        b = f2.subplots()
        sns.distplot(MTTF,kde=True,ax=b)
        b.set_title('Distribution')#显示图表标题
        b.set_xlabel('Mean Time To Failure')#x轴名称
        b.set_ylabel('Kernel Density')#y轴名称

    #创建画布控件
    cavas2=FigureCanvasTkAgg(f2,master=window)
    cavas2.draw()
    #显示画布空间
    cavas2.get_tk_widget().place(x=460,y=100)


def estimateRUL():
    file_open_path=filedialog.askopenfilename()
    #print(file_open_path)
    # r'E:\data\单应力对数尺度阿伦尼斯通用模型参数估计结果.csv'
    if var_allSpace.get()==1:
        data=pd.read_csv(file_open_path,encoding='GBK')
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
        print(RUL)
        # mu=np.mean(T)
        # sigma_T=np.std(T)
        #print(MTTF)
        #画布大小和分辨率
        f2=Figure(figsize=(5,4),dpi=100)
        #利用子图绘图
        b = f2.subplots()

        sns.distplot(RUL,kde=True,ax=b)
        # y=mlab.normpdf(mu,sigma_T)
        # plt.plot(y,'--')
        # plt.show()
        #plt.hist(T,alpha=0.5)
        # plt.xlabel("Reliability Lifetime")
        # plt.ylabel("Kernel Density")
        # plt.title('Distribution')
        # plt.show()
        # a.scatter(ex_TT, YY1d,c='blue',s=1,alpha=0.3)
        b.set_title('Distribution')#显示图表标题
        b.set_xlabel('Remaining Useful Lifeime')#x轴名称
        b.set_ylabel('Kernel Density')#y轴名称
    elif var_allSpace.get()==2:
        data=pd.read_csv(file_open_path,encoding='GBK')
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

        f2=Figure(figsize=(5,4),dpi=100)
        #利用子图绘图
        b = f2.subplots()
        sns.distplot(RUL,kde=True,ax=b)
        b.set_title('Distribution')#显示图表标题
        b.set_xlabel('Remaining Useful Lifetime')#x轴名称
        b.set_ylabel('Kernel Density')#y轴名称

    #创建画布控件
    cavas2=FigureCanvasTkAgg(f2,master=window)
    cavas2.draw()
    #显示画布空间
    cavas2.get_tk_widget().place(x=460,y=100)


if __name__ == '__main__':
    #初始化Tk(
    window=Tk()#用window当窗口Tk对象

    winWidth=1100
    winHeight=600

    window.title("性能退化数据分析软件mini")
    window.geometry("%sx%s"%(winWidth,winHeight))
    #window.configure(bg='black')
    window.resizable(0,0)
    topic=Label(window,text='性能退化数据分析软件',fg='black',font=('华文新魏',28),width=20,height=1,relief=FLAT)
    topic.place(x=30,y=40)
    #topic,bg='#d3fbfb'

    var_allSpace=IntVar()#选择不同退化模型
    var_allSpace.set(1)
    rbGPM=Radiobutton(window,text='单应力对数尺度阿伦尼斯通用退化轨迹模型',
                      variable=var_allSpace,value=1)
    rbGPM.place(x=450,y=40)
    rbWPM=Radiobutton(window,text='无应力指数尺度维纳过程退化模型',
                      variable=var_allSpace,value=2)

    rbWPM.place(x=450,y=60)

    button_open = Button(text ="打开文件", command = openData)
    button_open.place(x=980,y=100)

    button_estimate = Button(text ="参数估计", command = estimateData)
    button_estimate.place(x=980,y=150)

    # button_rl = Button(text ="打开参数文件", command = openData)
    # button_rl.place(x=880,y=200)

    button_rl = Button(text ="预测可靠度寿命", command = estimateRL)
    button_rl.place(x=980,y=200)

    button_mttf = Button(text ="预测平均失效时间", command = estimateMTTF)
    button_mttf.place(x=980,y=250)

    button_rul = Button(text ="预测平均剩余寿命", command = estimateRUL)
    button_rul.place(x=980,y=300)

    #初始图标 数据分布图
    image=Image.open('2-1.png')
    image_resize=image.resize((400,400),Image.ANTIALIAS)
    dataPic=ImageTk.PhotoImage(image_resize)
    label=Label(window,image=dataPic)
    label.place(x=30,y=100)


    image2=Image.open('2-3.png')
    image2_resize=image2.resize((500,400),Image.ANTIALIAS)
    dataPic2=ImageTk.PhotoImage(image2_resize)
    label2=Label(window,image=dataPic2)
    label2.place(x=460,y=100)


    # f=Figure(figsize=(30,30),dpi=100)
    # f_plot=f.add_subplot(111)
    #
    # canvs = FigureCanvasTkAgg(f, window)#f是定义的图像，root是tkinter中画布的定义位置
    # canvs.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    window.mainloop()