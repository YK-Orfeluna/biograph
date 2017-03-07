# -*- coding: utf-8 -*

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


"""First define some input parameters for the signal:"""
A = 2.
w = 1.
phi = 0.5 * np.pi
nin = 1000
nout = 10000
frac_points = 0.9 # Fraction of points to select

"""Randomly select a fraction of an array with timesteps:"""
r = np.random.rand(nin)
x = np.linspace(0.01, 10*np.pi, nin)
x = x[r >= frac_points]
normval = x.shape[0] # For normalization of the periodogram

"""Plot a sine wave for the selected times:"""
y = A * np.sin(w*x+phi)

"""Define the array of frequencies for which to compute the periodogram:"""
f = np.linspace(0.01, 10, nout)

"""Calculate Lomb-Scargle periodogram:"""
import scipy.signal as signal
pgram = signal.lombscargle(x, y, f)

print(x)
#print(y)
#print(f)
#print(pgram)
#exit()

"""Now make a plot of the input data:"""
plt.subplot(2, 1, 1)
plt.plot(x, y, 'b+')

"""Then plot the normalized periodogram:"""
plt.subplot(2, 1, 2)
plt.plot(f, pgram)#np.sqrt(4*(pgram/normval)))
plt.show()

exit()