#!/usr/bin/env python
# coding: utf-8

# In[8]:


# # # # # # # # # # 
#   Suhail Saini  #
#   ECE351-52     #
#   Lab 5         #
#   10/3/2023     #
# # # # # # # # # # 

#PART 1

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

steps = 1e-5
t = np.arange(0 , 1.2e-3 + steps, steps)

r = 1000
l = 0.027
c = 0.0000001

def alpha(r,c):
    return (-1/(2*r*c))

def omega(r,l,c):
    return ((1/2)*(np.sqrt((((1/(r*c))*(1/(r*c)))-(4/(l*c)))+0*1j)))

def g(p,r,c):
    return (p/(r*c))

alpha = alpha(r,c)
omega = omega(r,l,c)
p = alpha + omega

g_value = g(p,r,c)
g_abs = np.abs(g(p,r,c))
g_angle = np.angle(g(p,r,c))

def y(t):
    return ((g_abs/np.abs(omega)*np.exp(alpha*t)*np.sin((np.abs(omega)*t)+g_angle)))

num = [0, 1/(r*c), 0]
den = [1, 1/(r*c), 1/(l*c)]

tout, yout = sig.impulse((num, den), T = t)


# In[9]:


plt.subplot(2,1,1)
plt.plot(t, y(t))
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.show()

plt.subplot(2,1,2)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('yout')
plt.xlabel('tout')
plt.show()


# In[68]:


tout, yout = sig.step((num, den), T = t)

plt.subplot(1,1,1)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('yout')
plt.xlabel('tout')
plt.show()

