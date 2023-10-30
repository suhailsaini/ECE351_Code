#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # # # # # # # # # 
#   Suhail Saini  #
#   ECE351-52     #
#   Lab 6         #
#   10/10/2023    #
# # # # # # # # # # 

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

steps = 1e-2
t = np.arange(0 , 2 + steps, steps)

def step(t):
    y = np.zeros(t.shape)

    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y


# In[2]:


def y(t):
    return ((0.5-0.5*np.exp(-4*t)+np.exp(-6*t))*step(t))

plt.subplot(1,1,1)
plt.plot(t, y(t))
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.show()


# In[3]:


num = [1, 6, 12]
den = [1, 10, 24]

tout, yout = sig.step((num, den), T = t)

plt.subplot(1,1,1)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('yout')
plt.xlabel('tout')
plt.show()


# In[4]:


num = [1, 6, 12]
den = [1, 10, 24, 0]

r,p,k = sig.residue(num, den)

residue = r
pole = p
constant = k

print(residue)
print(pole)
print(constant)


# In[5]:


num = [25250]
den = [1, 18, 218, 2036, 9085, 25250, 0]

r,p,k = sig.residue(num, den)

residue = r
pole = p
constant = k

print(residue)
print(pole)
print(constant)


# In[6]:


t = np.arange(0 , 4.5 + steps, steps)
cos_m = 0
for i in range (len(r)):
    alpha = np.real(p[i])
    omega = np.imag(p[i])
    k_mag = np.abs(r[i])
    k_ang = np.angle(r[i])
    
    cos_m += k_mag*np.exp(alpha*t)*np.cos(omega*t+k_ang)*step(t)
    
plt.subplot(1,1,1)
plt.plot(t, cos_m)
plt.grid()
plt.ylabel('cos_m')
plt.xlabel('t')
plt.show()


# In[7]:


num = [25250]
den = [1, 18, 218, 2036, 9085, 25250]

tout, yout = sig.step((num, den), T = t)

plt.subplot(1,1,1)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('yout')
plt.xlabel('tout')
plt.show()

