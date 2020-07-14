# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 23:25:11 2019

@author: Tim Meiwald
"""

import numpy as np 
import matplotlib.pyplot as plt



alpha = 1.5
beta = 1.5
gamma = 1.5 
delta = 1.5

x = np.array([0.3,0.5,0.7],dtype = "float64") 
y = np.array([1.7,1.5,1.3],dtype = "float64") 

t_step = 0.001
N_step = 15000

def LotkaVolterra(t_step,N_step,x,y):
    ResultArr = np.ndarray((N_step,3),dtype = "float64")
    ResultArr[0,0] = x
    ResultArr[0,1] = y
    for i in np.arange(0,N_step,1):
        ResultArr[i,2] = t_step*i
    for i in np.arange(0,N_step-1,1):
        x = ResultArr[i,0]
        y = ResultArr[i,1]
        dx_dt = alpha*x - beta*x*y
        dy_dt = delta*x*y - gamma*y
        
        ResultArr[i+1,0] = x + dx_dt*t_step
        ResultArr[i+1,1] = y + dy_dt*t_step
    return ResultArr
   
    
plt.figure(figsize = (12.0,12.0))
plt.suptitle("Lotka-Volterra with Malthusian term")
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
for i in np.arange(0,3,1):
    Res = LotkaVolterra(t_step,N_step,x[i],y[i])
    x1 = "32%d" % (1+(2*i))
    x2 = "32%d" % (1+(2*i + 1))
    plt.subplot(x1)
    plt.plot(Res[:,0],Res[:,1])
    plt.xlim((0,3))
    plt.ylim((0,3))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Phase for x(0),y(0) = %.2f,%.2f" % (x[i],y[i]) )
    plt.subplot(x2)
    plt.plot(Res[:,2],Res[:,0],label = "x")
    plt.plot(Res[:,2],Res[:,1],label = "y")
    plt.xlabel("Time, t")
    plt.ylim((0,3))
    plt.title(r"Time for x(0),y(0) = %.2f,%.2f" % (x[i],y[i]))
    plt.legend()
plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask2/Figures/Task3LV.png")

alpha = np.array([1.25,2,4,8],dtype = "float64") 
K = np.array([0.2,0.5,0.75,0.875],dtype = "float64") 


beta = 1.
gamma = 1.
delta = 1.


def LotkaVolterraLogistic(t_step,N_step,x,y,K,alpha):
    ResultArr = np.ndarray((N_step,3),dtype = "float64")
    ResultArr[0,0] = x
    ResultArr[0,1] = y
    for i in np.arange(0,N_step,1):
        ResultArr[i,2] = t_step*i
    for i in np.arange(0,N_step-1,1):
        x = ResultArr[i,0]
        y = ResultArr[i,1]
        dx_dt = alpha*x*(1-K*x) - beta*x*y
        dy_dt = delta*x*y - gamma*y
        
        ResultArr[i+1,0] = x + dx_dt*t_step
        ResultArr[i+1,1] = y + dy_dt*t_step
    return ResultArr

for j in np.arange(0,4,1):
    plt.figure(figsize = (12.0,12.0))
    plt.suptitle(r"Lotka-Volterra Logistic, $\alpha$,K = %.3f,%.3f" % (alpha[j],K[j]))
    plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
    for i in np.arange(0,3,1):
        Res = LotkaVolterraLogistic(t_step,N_step,x[i],y[i],K[j],alpha[j])
        x1 = "32%d" % (1+(2*i))
        x2 = "32%d" % (1+(2*i + 1))
        plt.subplot(x1)
        plt.plot(Res[:,0],Res[:,1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0,3))
        plt.ylim((0,3))
        plt.title(r"Phase for x(0),y(0) = %.2f,%.2f" % (x[i],y[i]) )
        plt.subplot(x2)
        plt.plot(Res[:,2],Res[:,0],label = "x")
        plt.plot(Res[:,2],Res[:,1],label = "y")
        plt.xlabel("Time, t")
        plt.ylim((0,3))
        plt.title(r"Time for x(0),y(0) = %.2f,%.2f" % (x[i],y[i]))
        plt.legend()
    plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask2/Figures/Task3LLV%d.png" % j)
