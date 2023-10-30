#!/usr/bin/env python
# coding: utf-8

# In[34]:


# # # # # # # # # # 
#   Suhail Saini  #
#   ECE351-52     #
#   Lab 8         #
#   10/24/2023    #
# # # # # # # # # # 

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

def a(k):
    return 2/(k*np.pi)*np.sin(k*np.pi)

def b(k):
    return 2/(k*np.pi)*(1-np.cos(k*np.pi)) 

print(f"a(1) = {a(1)}")
print(f"b(1) = {b(1)}")
print(f"b(2) = {b(2)}")
print(f"b(3) = {b(3)}")


# In[17]:


N = [1, 3, 15, 50, 150, 1500]
T = 8
steps = 1e-3
t = np.arange(0 , 20 + steps, steps)
y = 0

for h in [1,2]:
    for i in ([1+(h-1)*3, 2+(h-1)*3, 3+(h-1)*3]):
        for k in np.arange(1, N[i-1]+1):
            
            b = 2/(k*np.pi)*(1-np.cos(k*np.pi))
            x = b*np.sin(2*k*np.pi*t/T)
            
            y = y + x
            
        plt.figure(h, figsize = (10, 7))
        plt.subplot(3, 1, i-(h-1)*3)
        plt.plot(t, y)
        plt.grid()
        plt.ylabel(f"N = {N[i-1]}")
        if i == 1 or i == 4:
            plt.title('Fourier Series Approximations of x(t)')
        if i == 3 or i == 6:
            plt.xlabel('t [s]')
            plt.show()
        y = 0

