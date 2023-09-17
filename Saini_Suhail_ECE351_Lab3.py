#!/usr/bin/env python
# coding: utf-8

# In[132]:


# # # # # # # # # # 
#   Suhail Saini  #
#   ECE351-52     #
#   Lab 3         #
#   09/19/2023    #
# # # # # # # # # # 

#PART 1

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

steps = 1e-2
t = np.arange(0 , 20 + steps, steps)

def step(t):
    y = np.zeros(t.shape)
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def ramp(t):
    y = np.zeros(t.shape)
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

def f1(t):
    return step(t-2) - step(t-9)

def f2(t):
    return np.exp(-t)*step(t)

def f3(t):
    return ramp(t-2)*(step(t-2) - step(t-3)) + ramp(4-t)*(step(t-3) - step(t-4))

a = f1(t)
b = f2(t)
c = f3(t)
          
plt.subplot(3,1,1)
plt.plot(t, a)
plt.ylim((-0.5,1.5))
plt.grid()
plt.ylabel('f1(t)')
plt.xlabel('t')
plt.show()

plt.subplot(3,1,2)
plt.plot(t, b)
plt.ylim((-0.5,1.5))
plt.grid()
plt.ylabel('f2(t)')
plt.xlabel('t')
plt.show()

plt.subplot(3,1,3)
plt.plot(t, c)
plt.ylim((-0.5,1.5))
plt.grid()
plt.ylabel('f3(t)')
plt.xlabel('t')
plt.show()


# In[129]:


# PART 2

def my_conv(f1, f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1extended = np.append(f1, np.zeros((1, Nf2 - 1)))
    f2extended = np.append(f2, np.zeros((1, Nf1 - 1)))
    result = np.zeros(f1extended.shape)
    
    for i in range(Nf2 + Nf1 - 2):
        result[i] = 0
        
        for j in range(Nf1):
            result[i] += f1extended[j]*f2extended[i - j + 1]
    return result


# In[133]:


steps = 1e-2
t = np.arange(0 , 20 + steps, steps)

NN = len(t)

t_conv = np.arange(0,2*t[NN-1] + steps, steps)

c1 = my_conv(a, b)*steps


# In[134]:


plt.subplot(1,1,1)
plt.plot(t_conv, c1)
plt.grid()
plt.ylabel('c1(t)')
plt.xlabel('t_')
plt.show()


# In[106]:


# Check with scipy.signal

import scipy.signal as sig

NN = len(t)

t_conv = np.arange(0,2*t[NN-1] + steps, steps)

def conv1(t_conv):
    return sig.convolve(a, b, mode = 'full', method = 'auto')

def conv2(t_conv):
    return sig.convolve(b, c, mode = 'full', method = 'auto')

def conv3(t_conv):
    return sig.convolve(a, c, mode = 'full', method = 'auto')

g = conv1(t_conv)
h = conv2(t_conv)
k = conv3(t_conv)

plt.subplot(3,1,1)
plt.plot(t_conv, g)
plt.ylim((-500,1500))
plt.grid()
plt.ylabel('conv1(t)')
plt.xlabel('t_conv')
plt.show()

plt.subplot(3,1,2)
plt.plot(t_conv, h)
plt.ylim((-500,1500))
plt.grid()
plt.ylabel('conv2(t)')
plt.xlabel('t_conv')
plt.show()

plt.subplot(3,1,3)
plt.plot(t_conv, k)
plt.ylim((-500,1500))
plt.grid()
plt.ylabel('conv3(t)')
plt.xlabel('t_conv')
plt.show()

