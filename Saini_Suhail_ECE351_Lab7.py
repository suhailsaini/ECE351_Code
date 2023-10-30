#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # # # # # # # # # 
#   Suhail Saini  #
#   ECE351-52     #
#   Lab 7         #
#   10/17/2023    #
# # # # # # # # # # 

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

def G(s):
    return (s+9)/(s^2 - 6*s - 16)

def A(s):
    return (s+4)/(s^2 + 4*s + 3)

def B(s):
    return (s^2 + 26*s + 168)


# In[2]:


# RPK of G(s)

numg = [1, 9]
deng = sig.convolve([1, -6, -16], [1, 4])

z,p,k = sig.tf2zpk(numg, deng)

zeroes = z
poles = p
gain = k

print(f'Zeros of G(s): {zeroes}')
print(f'Poles of G(s): {poles}')
print(f'Gain of G(s): {gain}')


# In[3]:


# RPK of A(s)

numa = [1, 4]
dena = [1, 4, 3]

z,p,k = sig.tf2zpk(numa, dena)

zeroes = z
poles = p
gain = k

print(f'Zeros of A(s): {zeroes}')
print(f'Poles of A(s): {poles}')
print(f'Gain of A(s): {gain}')


# In[4]:


# RPK of B(s)

numb = [1, 26, 168]
denb = [1]

z,p,k = sig.tf2zpk(numb, denb)

zeroes = z
poles = p
gain = k

print(f'Zeros of B(s): {zeroes}')
print(f'Poles of B(s): {poles}')
print(f'Gain of B(s): {gain}')


# In[19]:


# OPEN LOOP TRANSFER FUNCTION

xn = sig.convolve(numa, numg)
xd = sig.convolve(dena, deng)
t,y = sig.step((xn,xd))

plt.subplot(1,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y')
plt.xlabel('t')
plt.show()

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

print(xn)
print(xd)
print(my_conv(numa,numg))
print(my_conv(dena,deng))


# In[6]:


# CLOSED LOOP TRANSFER FUNCTION

numclosed = sig.convolve(numa, numg)
denclosed = sig.convolve(deng + sig.convolve(numb, numg), dena)

print(numclosed)
print(denclosed)

t,y = sig.step((numclosed, denclosed))

plt.subplot(1,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y')
plt.xlabel('t')
plt.show()


# In[8]:


z,p,k = sig.tf2zpk(numclosed, denclosed)

zeroes = z
poles = p
gain = k

print(zeroes)
print(poles)
print(gain)


# In[9]:


numclosed = sig.convolve(numa, numg)
denclosed = sig.convolve(deng + sig.convolve(numb, numg), dena)

r,p,k = sig.residue(numclosed, denclosed)

residue = r
pole = p
constant = k

print(residue)
print(pole)
print(constant)

