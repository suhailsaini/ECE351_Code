#!/usr/bin/env python
# coding: utf-8

# In[49]:


# # # # # # # # # # 
#   Suhail Saini  #
#   ECE351-52     #
#   Lab 9         #
#   10/31/2023    #
# # # # # # # # # # 

import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9})

fs = 100
steps = 1/fs
t = np.arange(0, 2, steps)

def fast(x,fs):
    N = len(x)
    X_fft = fft.fft(x)
    X_fft_shifted = fft.fftshift(X_fft)

    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    return freq, X_mag, X_phi

def fastc(x,fs):
    N = len(x)
    X_fft = fft.fft(x)
    X_fft_shifted = fft.fftshift(X_fft)

    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)

    for i in range (len(X_phi)):
        if np.abs(X_mag[i]) < 1e-10:
            X_phi[i] = 0

    return freq, X_mag, X_phi


# In[50]:


x = np.cos(2*np.pi*t)
freq, X_mag, X_phi = fast(x,fs)

plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.title('User-Defined FFT of x(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.xlim([0, 2])

plt.subplot(3,2,3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('|x(f)|')
plt.xlim([-40, 40])

plt.subplot(3,2,4)
plt.stem(freq, X_mag)
plt.grid()
plt.xlim([-2, 2])

plt.subplot(3,2,5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')
plt.xlim([-40, 40])

plt.subplot(3,2,6)
plt.stem(freq, X_phi)
plt.grid()
plt.xlabel('f[Hz]')
plt.xlim([-2, 2])

plt.tight_layout()
plt.show()


# In[51]:


x = 5*np.sin(2*np.pi*t)
freq, X_mag, X_phi = fast(x,fs)

plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.title('User-Defined FFT of x(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.xlim([0, 2])

plt.subplot(3,2,3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('|x(f)|')
plt.xlim([-40, 40])

plt.subplot(3,2,4)
plt.stem(freq, X_mag)
plt.grid()
plt.xlim([-2, 2])

plt.subplot(3,2,5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')
plt.xlim([-40, 40])

plt.subplot(3,2,6)
plt.stem(freq, X_phi)
plt.grid()
plt.xlabel('f[Hz]')
plt.xlim([-2, 2])

plt.tight_layout()
plt.show()


# In[52]:


x = 2*np.cos((2*np.pi*2*t)-2)+(np.sin((2*np.pi*6*t)+3))
freq, X_mag, X_phi = fast(x,fs)

plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.title('User-Defined FFT of x(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.xlim([0, 2])

plt.subplot(3,2,3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('|x(f)|')
plt.xlim([-40, 40])

plt.subplot(3,2,4)
plt.stem(freq, X_mag)
plt.grid()
plt.xlim([-2, 2])

plt.subplot(3,2,5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')
plt.xlim([-40, 40])

plt.subplot(3,2,6)
plt.stem(freq, X_phi)
plt.grid()
plt.xlabel('f[Hz]')
plt.xlim([-2, 2])

plt.tight_layout()
plt.show()


# In[53]:


x = np.cos(2*np.pi*t)
freqc, X_magc, X_phic = fastc(x,fs)

plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.title('User-Defined FFT of x(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.xlim([0, 2])

plt.subplot(3,2,3)
plt.stem(freqc, X_magc)
plt.grid()
plt.ylabel('|x(f)|')
plt.xlim([-40, 40])

plt.subplot(3,2,4)
plt.stem(freqc, X_magc)
plt.grid()
plt.xlim([-2, 2])

plt.subplot(3,2,5)
plt.stem(freqc, X_phic)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')
plt.xlim([-40, 40])

plt.subplot(3,2,6)
plt.stem(freqc, X_phic)
plt.grid()
plt.xlabel('f[Hz]')
plt.xlim([-2, 2])

plt.tight_layout()
plt.show()


# In[54]:


x = 5*np.sin(2*np.pi*t)
freqc, X_magc, X_phic = fastc(x,fs)

plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.title('User-Defined FFT of x(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.xlim([0, 2])

plt.subplot(3,2,3)
plt.stem(freqc, X_magc)
plt.grid()
plt.ylabel('|x(f)|')
plt.xlim([-40, 40])

plt.subplot(3,2,4)
plt.stem(freqc, X_magc)
plt.grid()
plt.xlim([-2, 2])

plt.subplot(3,2,5)
plt.stem(freqc, X_phic)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')
plt.xlim([-40, 40])

plt.subplot(3,2,6)
plt.stem(freqc, X_phic)
plt.grid()
plt.xlabel('f[Hz]')
plt.xlim([-2, 2])

plt.tight_layout()
plt.show()


# In[55]:


x = 2*np.cos((2*np.pi*2*t)-2)+(np.sin((2*np.pi*6*t)+3))
freqc, X_magc, X_phic = fastc(x,fs)

plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.title('User-Defined FFT of x(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.xlim([0, 2])

plt.subplot(3,2,3)
plt.stem(freqc, X_magc)
plt.grid()
plt.ylabel('|x(f)|')
plt.xlim([-40, 40])

plt.subplot(3,2,4)
plt.stem(freqc, X_magc)
plt.grid()
plt.xlim([-2, 2])

plt.subplot(3,2,5)
plt.stem(freqc, X_phic)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')
plt.xlim([-40, 40])

plt.subplot(3,2,6)
plt.stem(freqc, X_phic)
plt.grid()
plt.xlabel('f[Hz]')
plt.xlim([-2, 2])

plt.tight_layout()
plt.show()


# In[56]:


k = 15
T = 8
t = np.arange(0 , 16, steps)
            
b = 2/(k*np.pi)*(1-np.cos(k*np.pi))
x = b*np.sin(2*k*np.pi*t/T)

freqc, X_magc, X_phic = fastc(x,fs)

plt.subplot(3,1,1)
plt.plot(t, x)
plt.grid()
plt.title('User-Defined FFT of x(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.xlim([0, 2])

plt.subplot(3,2,3)
plt.stem(freqc, X_magc)
plt.grid()
plt.ylabel('|x(f)|')
plt.xlim([-40, 40])

plt.subplot(3,2,4)
plt.stem(freqc, X_magc)
plt.grid()
plt.xlim([-2, 2])

plt.subplot(3,2,5)
plt.stem(freqc, X_phic)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')
plt.xlim([-40, 40])

plt.subplot(3,2,6)
plt.stem(freqc, X_phic)
plt.grid()
plt.xlabel('f[Hz]')
plt.xlim([-2, 2])

plt.tight_layout()
plt.show()

