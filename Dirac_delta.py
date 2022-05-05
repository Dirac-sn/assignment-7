# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:42:44 2022

@author: harsh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

plt.style.use("bmh")

def delta_sin(t):
    return(np.vectorize(lambda x,a: np.sin((x-a)/t)/((x-a)*np.pi)))

def delta_lorr(t):
    return(np.vectorize(lambda x,a: t/(((x-a)**2 + t**2)*np.pi)))

def delta_decay1(t):
    return(np.vectorize(lambda x,a: np.exp(-np.abs(x-a)/t)/(2*t)))

def delta_decay2(t):
    return(np.vectorize(lambda x,a: np.exp(-(x-a)**2/(2*t))/(2*np.sqrt(t*np.pi))))


x_space = np.linspace(-1,1,200)

def makePlots(f,x_space):
    fig,ax = plt.subplots(1,1)
    e_0 = 0.4
    e_arr = e_0*2**(-1*np.arange(1,6,1,dtype=float))
    a = 0.5
    for e in e_arr:
        y = f(e)(x_space,a)
        plt.plot(x_space,y,label="$\epsilon$="+str(e))
        
    print(quad(lambda x:f(e)(x,a),-100,100)[0])
    plt.legend()

makePlots(delta_lorr,x_space)
makePlots(delta_sin,x_space)
makePlots(delta_decay1,x_space)
makePlots(delta_decay2,x_space)
plt.show()