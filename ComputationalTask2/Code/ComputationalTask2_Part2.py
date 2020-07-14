# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:43:30 2019

@author: Tim Meiwald
"""
import numpy as np 
import matplotlib.pyplot as plt
#z + x + y + q = 1 = const




k = np.array([2.5,1,8.5,0.07],dtype = "float64")
k_neg = np.array([2,0.111,0,0.025],dtype = "float64")
# x,y,q,z, x+y+q+z = 1 = const
InitCond1 = np.array([0,0,0,1],dtype = "float64")
InitCond2 = np.array([0.5,0.5,0,0],dtype = "float64")

gamma1 = np.array([2,0,0,-2],dtype = "float64")
gamma2 = np.array([0,1,0,-1],dtype = "float64")
gamma3 = np.array([1,1,0,2],dtype = "float64")
gamma4 = np.array([0,0,1,-1],dtype = "float64")


t_step = 0.01
N_step = 60000
def MAL(t_step,N_step,InitCond,kNegInt):
    k1Neg = np.arange(0,10,1)/3
    k1Neg = k1Neg[kNegInt]
    k_neg[0] = k1Neg
    x = InitCond[0]
    y = InitCond[1]
    q = InitCond[2]
    z = InitCond[3]
    ResultArr = np.ndarray((N_step,5),dtype = "float64")
    ResultArr[0,:4] = InitCond
    ResultArr[0,4] = 0
    for i in np.arange(1,N_step,1):
        dx_dt = 2*k[0]*z**2 - 2*k_neg[0]*x**2 - k[2]*x*y
        dy_dt = k[1]*z - k_neg[1]*y - k[2]*x*y
        dq_dt = k[3]*z - k_neg[3]*q
        dz_dt = -2*k[0]*z**2 + 2*k_neg[0]*x**2 - k[1]*z + k_neg[1]*y + 2*k[2]*x*y - k[3]*z + k_neg[3]*q
        #print(dx_dt,dy_dt,dq_dt,dz_dt)
        x = x + t_step*dx_dt
        y = y + t_step*dy_dt
        q = q + t_step*dq_dt
        z = z + t_step*dz_dt
        
        ResultArr[i,0] = x
        ResultArr[i,1] = y
        ResultArr[i,2] = q
        ResultArr[i,3] = z
        ResultArr[i,4] = i*t_step
        #print(x+y+q+z)
        #print(x,y,q,z)
    return ResultArr
   
plt.figure(figsize = (12.0,12.0))
plt.suptitle("(q,x) plane : x(0),y(0),q(0) = 0,0,0")
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
for i in np.arange(0,10,1):
    Res = MAL(t_step,N_step,InitCond1,i)
    x = "33%d" % i
    plt.subplot(x)
    k1NegVal = np.arange(0,10,1)/3
    plt.title(r"$k_{-1} =$ %.3f" % k1NegVal[i])
    plt.xlabel("q")
    plt.ylabel("x")
    plt.ylim((-0.05,0.6))
    plt.xlim((-0.05,0.6))
    plt.plot(Res[:,2],Res[:,0])
plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask2/Figures/Task2_1.png")

plt.figure(figsize = (12.0,12.0))
plt.suptitle("(x,y) plane : x(0),y(0),q(0) = 0,0,0")
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
for i in np.arange(0,10,1):
    Res = MAL(t_step,N_step,InitCond1,i)
    x = "33%d" % i
    plt.subplot(x)
    k1NegVal = np.arange(0,10,1)/3
    plt.title(r"$k_{-1} =$ %.3f" % k1NegVal[i])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim((-0.05,0.6))
    plt.xlim((-0.05,0.6))
    plt.plot(Res[:,0],Res[:,1])
plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask2/Figures/Task2_2.png")

plt.figure(figsize = (12.0,12.0))
plt.suptitle("(q,x) plane : x(0),y(0),q(0) = 0.5,0.5,0")
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
for i in np.arange(0,10,1):
    Res = MAL(t_step,N_step,InitCond2,i)
    x = "33%d" % i
    plt.subplot(x)
    k1NegVal = np.arange(0,10,1)/3
    plt.title(r"$k_{-1} =$ %.3f" % k1NegVal[i])
    plt.xlabel("q")
    plt.ylabel("x")
    plt.ylim((-0.05,0.6))
    plt.xlim((-0.05,0.6))
    plt.plot(Res[:,2],Res[:,0])
plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask2/Figures/Task2_3.png")

plt.figure(figsize = (12.0,12.0))
plt.suptitle("(x,y) plane : x(0),y(0),q(0) = 0.5,0.5,0")
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
for i in np.arange(0,10,1):
    Res = MAL(t_step,N_step,InitCond2,i)
    x = "33%d" % i
    plt.subplot(x)
    k1NegVal = np.arange(0,10,1)/3
    plt.title(r"$k_{-1} =$ %.3f" % k1NegVal[i])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim((-0.05,0.6))
    plt.xlim((-0.05,0.6))
    plt.plot(Res[:,0],Res[:,1])
plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask2/Figures/Task2_4.png")

   
