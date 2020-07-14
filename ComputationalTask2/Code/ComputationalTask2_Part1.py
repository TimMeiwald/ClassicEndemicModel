# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:35:42 2019

@author: Tim Meiwald
"""

import numpy as np
import matplotlib.pyplot as plt
#Three initial Conditions


s0 = np.array([0.35,0.25,0.4],dtype = "float64")
i0 = np.array([0.65,0.75,0.4],dtype = "float64")

#Constants
gamma = 0.3
mu = 0.016
Beta = np.arange(0.9,1.7,0.1)
#time step Size
t_step = 0.1
#Number of steps
N_step = 2500


#Classic Endemic Model
#Function 1
#ds/dt = -Beta*i*s + mu - mu*s
#Function 2
#di/dt = Beta*i*s - (gamma + mu)*i

def CEM(s0,i0,gamma,Beta,mu,t_step,N_step):
    #Classic Endemic Model
    ds_dt_Results = np.ndarray(N_step,dtype = "float64")
    di_dt_Results = np.ndarray(N_step,dtype = "float64")
    ds_dt_Results[0] = s0
    di_dt_Results[0] = i0
    for k in np.arange(0,N_step-1,1):
        #Assigning to s and i for readability
        s = ds_dt_Results[k]
        i = di_dt_Results[k]
        
        #Classic Endemic Model Functions
        ds_dt = -Beta*i*s + mu - mu*s
        di_dt = Beta*i*s - (gamma + mu)*i
        #Update the s and I values by adding the derivative multipled by the time
        #step, dt = 1 so not shown in code
        #Euler method! Runge_Kutta would give better, more stable results
        s = s + ds_dt*t_step
        i = i + di_dt*t_step
        
        #Add to the Results
        ds_dt_Results[k+1] = s
        di_dt_Results[k+1] = i
        
    return ds_dt_Results, di_dt_Results


for i in np.arange(0,8,1):
    plt.figure(figsize = (12.0,12.0))
    plt.suptitle("Beta = %f" % (Beta[i]))
    plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
    for j in np.arange(0,3,1):
        x = "31%d" % j
        plt.subplot(x)
        T = CEM(s0[j],i0[j],gamma,Beta[i],mu,t_step,N_step)
        plt.plot(T[1],T[0])
        plt.title("i(0), s(0) = %.2f, %.2f" % (i0[j],s0[j]))
        plt.xlabel("i")
        plt.ylabel("s")
        plt.ylim((-0.05,0.6))
        plt.xlim((-0.05,0.9))
    plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask2/Figures/Task1_%d.png" % i)

    


