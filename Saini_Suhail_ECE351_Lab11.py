#!/usr/bin/env python
# coding: utf-8

# In[9]:


# # # # # # # # # # 
#   Suhail Saini  #
#   ECE351-52     #
#   Lab 11        #
#   11/14/2023    #
# # # # # # # # # # 

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

num = [2, -40]
den = [1, -10, 16]

y = [r,p,_] = sig.residuez(num,den)

print(y)


# In[10]:


#
#Copyright (c) 2011 Christropher Felton
#
# This program is free software : you can redistribute it and / or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation , either version 3 of the License , or
# ( at your option ) any later version .
#
# This program is distributed in the hope that it will be useful ,
# but WITHOUT ANY WARRANTY ; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE . See the
# GNU Lesser General Public License for more details .
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program . If not , see < http :// www . gnu . org / license s / >.
#
# The following is derived from the slides presented by
# Alexander Kain for CS506 /606 " Special Topics : Speech Signal Processing "
# CSLU / OHSU , Spring Term 2011.
#
#
#
# Modified by Drew Owens in Fall 2018 for use in the University of Idaho ’s
# Department of Electrical and Computer Engineering Signals and Systems I Lab
# ( ECE 351)
#
# Modified by Morteza Soltani in Spring 2019 for use in the ECE 351 of the U of
# I .
#
# Modified by Phillip Hagen in Fall 2019 for use in the University of Idaho ’s
# Department of Electrical and Computer Engineering Signals and Systems I Lab
# ( ECE 351)

def zplane(b, a, filename = None):
    """ Plot the complex z-plane given a transfer function """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import patches
    
    # get a figure/plot
    ax = plt.subplot(1,1,1)
    
    # create the unit circle
    uc = patches.Circle((0,0),radius=1,fill=False,color='black',ls='dashed')
    ax.add_patch(uc)
    
    # the coefficients are less than 1, normalize the coefficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1
        
    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
        
    # get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'o', ms=10,label='Zeros')
    plt.setp(t1,markersize=10.0,markeredgewidth=1.0)
    
    # plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10,label='Poles')
    plt.setp(t2,markersize=12.0,markeredgewidth=3.0)
    
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.legend()
    
    # set the ticks
    
    # r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    # ticks = [-1, -0.5, 0.5, 1]; plt.xticks(ticks); plt.yticks(ticks)
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        
    return z, p, k


# In[3]:


z,p,k = zplane(num, den)


# In[8]:


w, h = sig.freqz(num, den, whole = True)

magh = np.abs(h)
magdb = 20*np.log10(magh)

plt.figure(figsize=(11,3))
plt.subplot(1,1,1)
plt.plot(w/np.pi, magdb)
plt.grid()
plt.ylabel('Magnitude (in dB)')
plt.xlabel('w')
plt.show()

plt.figure(figsize=(11,3))
plt.subplot(1,1,1)
plt.plot(w/np.pi, np.degrees(np.angle(h)))
plt.grid()
plt.ylabel('Phase')
plt.xlabel('w')
plt.show()

