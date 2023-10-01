#!/usr/bin/env python
# coding: utf-8

# In[22]:


# # # # # # # # # # 
#   Suhail Saini  #
#   ECE351-52     #
#   Lab 4         #
#   09/26/2023    #
# # # # # # # # # # 

#PART 1

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

steps = 1e-2
t = np.arange(-10 , 10 + steps, steps)

def step(t):
    y = np.zeros(t.shape)
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def h1(t):
    return np.exp(2*t)*step(1-t)

def h2(t):
    return step(t-2) - step(t-6)

def h3(f,t):
    w = 2*np.pi*f
    return np.cos(w*t)*step(t)

f = 0.25

a = h1(t)
b = h2(t)
c = h3(f,t)
d = step(t)

plt.subplot(3,1,1)
plt.plot(t, a)
plt.grid()
plt.ylabel('h1(t)')
plt.xlabel('t')
plt.show()

plt.subplot(3,1,2)
plt.plot(t, b)
plt.grid()
plt.ylabel('h2(t)')
plt.xlabel('t')
plt.show()

plt.subplot(3,1,3)
plt.plot(t, c)
plt.grid()
plt.ylabel('h3(t)')
plt.xlabel('t')
plt.show()


# In[86]:


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

t = np.arange(-10, 10 + steps, steps)
NN = len(t)
t_conv = np.arange(2*t[0], 2*t[NN-1] + steps, steps)

c1 = my_conv(a, d)*steps
plt.subplot(3,1,1)
plt.plot(t_conv, c1)
plt.grid()
plt.ylabel('c1(t)')
plt.xlabel('t_conv')
plt.xlim([-10, 10])
plt.show()

c2 = my_conv(b, d)*steps
plt.subplot(3,1,2)
plt.plot(t_conv, c2)
plt.grid()
plt.ylabel('c2(t)')
plt.xlabel('t_conv')
plt.xlim([-10, 10])
plt.show()

c3 = my_conv(c, d)*steps
plt.subplot(3,1,3)
plt.plot(t_conv, c3)
plt.grid()
plt.ylabel('c3(t)')
plt.xlabel('t_conv')
plt.xlim([-10, 10])
plt.show()


# In[90]:


def ramp(t):
    y = np.zeros(t.shape)
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

def c1(t):
    return ((np.exp(2*t)/2*step(1-t))+(np.exp(2)/2)*step(t-1))

def c2(t):
    return ramp(t-2)-ramp(t-6)

def c3(t,f):
    w = 2*np.pi*f
    return (np.sin(w*t)/w)*step(t)

g = c1(t)
h = c2(t)
i = c3(t,f)

plt.subplot(3,1,1)
plt.plot(t, g)
plt.grid()
plt.ylabel('c1(t)')
plt.xlabel('t')
plt.show()

plt.subplot(3,1,2)
plt.plot(t, h)
plt.grid()
plt.ylabel('c2(t)')
plt.xlabel('t')
plt.show()

plt.subplot(3,1,3)
plt.plot(t, i)
plt.grid()
plt.ylabel('c3(t)')
plt.xlabel('t')
plt.show()

