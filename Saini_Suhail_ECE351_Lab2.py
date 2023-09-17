#!/usr/bin/env python
# coding: utf-8

# In[16]:


# # # # # # # # # # 
#   Suhail Saini  #
#   ECE351-52     #
#   Lab 2         #
#   09/12/2023    #
# # # # # # # # # # 

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

steps = 1e-2
t = np.arange(0, 5+steps, steps)

print(f"Number of elements: len(t) = {len(t)}\nFirst Element: t[0] = {t[0]}\nLast Element: t[-1] = {t[-1]}")
      
def example1(t):
    y = np.zeros(t.shape)
      
    for i in range(len(t)):
      if i < (len(t) +1)/3:
           y[i] = t[i]**2
      else:
           y[i] = np.sin(5*t[i]) + 2
    return y
      
y = example1(t)
      
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) with Good Resolution')
plt.title('Background - Illustration of for Loops and if/else Statements')
          
t = np.arange(0, 5 + 0.25, 0.25)
y = example1(t)
          
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) with Poor Resolution')
plt.xlabel('t')
plt.show()


# In[44]:


#P1T2

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

steps = 1e-2
t = np.arange(0, 5+steps, steps)

print(f"Number of elements: len(t) = {len(t)}\nFirst Element: t[0] = {t[0]}\nLast Element: t[-1] = {t[-1]}")
      
def func1(t):
    return np.cos(t)
      
y = func1(t)
      
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) with Good Resolution')
plt.title('Background - Illustration of for Loops and if/else Statements')
          
t = np.arange(0, 10 + 0.25, 0.25)
y = func1(t)
          
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) with Poor Resolution')
plt.xlabel('t')
plt.show()


# In[1]:


#P2T1,T2, T3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})

steps = 1e-3
t = np.arange(0, 5 + steps, steps)
 
def step(t):
    y = np.zeros(len(t))
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def ramp(t):
    y = np.zeros(len(t))
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

def func2(t):
    return ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6)

t = np.arange(-5, 10 + steps, steps)
y = func2(t)
          
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.show()


# In[9]:


#P3T1

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})

steps = (1e-3)
t = np.arange(0, 5 + steps, steps)
 
def step(t):
    y = np.zeros(len(t))
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def ramp(t):
    y = np.zeros(len(t))
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

def func2(t):
    return ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6)

t = np.arange(-10, 5 + steps, steps)
y = func2(-t)
          
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.show()


# In[8]:


#P3T2

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})

steps = 1e-3
 
def step(t):
    y = np.zeros(len(t))
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def ramp(t):
    y = np.zeros(len(t))
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

def func2(t):
    return ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6)

t = np.arange(-13, 13 + steps, steps)

y1 = func2(t - 4)
y2 = func2(-t - 4)
          
plt.subplot(2, 1, 2)
plt.plot(t, y1)
plt.plot(t, y2)
plt.grid()
plt.ylim((-1,10))
plt.ylabel('y(t)')
plt.xlabel('t')
plt.show()


# In[7]:


#P3T3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})

steps = 1e-3
t = np.arange(0, 5 + steps, steps)
 
def step(t):
    y = np.zeros(len(t))
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def ramp(t):
    y = np.zeros(len(t))
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

def func2(t):
    return ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6)

t = np.arange(-5, 15 + steps, steps)

y3 = func2(t/2)
y4 = func2(2*t)
          
plt.subplot(2, 1, 2)
plt.plot(t, y3)
plt.plot(t, y4)
plt.grid()
plt.ylim((-5,11))
plt.ylabel('y(t)')
plt.xlabel('t')
plt.show()


# In[16]:


#P3T5

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})

steps = 1e-3
t = np.arange(-5, 10+steps, steps)

def step(t):
    y = np.zeros(len(t))
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def ramp(t):
    y = np.zeros(len(t))
      
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

def func2(t):
    return ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6)

y = func2(t)
dt = np.diff(t)
dy = np.diff(y, axis = 0)/dt

plt.subplot(2, 1, 2)
plt.plot(t[range(len(dy))], dy)
plt.grid()
plt.ylim((-5,6))
plt.ylabel('y(t)')
plt.xlabel('t')
plt.show()

