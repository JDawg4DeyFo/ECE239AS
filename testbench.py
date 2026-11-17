# File: testbench.py
# Author: Jacob Dennon
# Date: November 14th, 2026
# Description: This file will run tests to verify the system works
from OFDM import *
from channel import *
import numpy as np
import matplotlib.pyplot as plt

################################################################################
## Channel tests
################################################################################
# Test Group 0: basic AWGN_channel() tests
# The first test is just plotting retrieved OFDM symbols + noise to see if what
# we get makes sense
K = 100
pilot_spacing = 4
CP = 1/4
M = 4

# Build symbol
OFDM_Symbol, sequence1 = get_OFDM_symbol(K, CP, M, sps=1, pilot_spacing=pilot_spacing, return_sequence=True)
OFDM_Symbol = AWGN_channel(OFDM_Symbol, 1e-7)
retrieved_sequence1 = np.sqrt(K) * np.fft.fft(OFDM_Symbol[math.floor(CP * K):]) # discarding CP

plt.figure()
plt.scatter(np.real(retrieved_sequence1), np.imag(retrieved_sequence1), color = 'red', label='RX')
plt.scatter(np.real(sequence1), np.imag(sequence1), label='truth')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.legend()
plt.title(f"Retrieved OFDM Symbols; AWGN Channel")


# Test Group 1: basic multipath_channel() tests
# Let's see how delay spread messes with a sinc function
fs = 1e7
Ts = 1/fs
B = 1e5
T=1/B
t = np.arange(-50/B, 50/B, 1/fs)
x = np.sinc(t *B * 2)
X = np.fft.fft(x) /fs
X = np.fft.fftshift(X)
# frequencies = np.arange(-len(X) / 2, len(X)/2) * fs / len(x)
frequencies = np.fft.fftshift(np.fft.fftfreq(len(t), d=1/fs))

plt.figure()
plt.title("Frequency Response of sinc")
plt.plot(frequencies*1e-3, np.abs(X))
plt.xlabel("kHz")
plt.ylabel("|X(f)|")
plt.grid(True)

plt.figure()
plt.title("sinc(t)")
plt.plot(t, x)
plt.xlabel("s")
plt.ylabel("x(t)")
plt.grid(True)


# test multipath component
tau_d = math.floor(T / (Ts * 10))
y = multipath_channel(x, 10, tau_d, 1e6)
Y = np.fft.fft(y) / fs
Y = np.fft.fftshift(Y)
frequencies = np.fft.fftshift(np.fft.fftfreq(len(y), d=1/fs))

plt.figure()
plt.title("Frequency Response of multipath sinc")
plt.plot(frequencies*1e-3, np.abs(Y), label="T/10 max delay")
plt.xlabel("kHz")
plt.ylabel("|Y(f)|")
plt.grid(True)

# test wideband
tau_d = math.floor(10*T / (Ts))
y = multipath_channel(x, 10, tau_d, 1e6)
Y = np.fft.fft(y) / fs
Y = np.fft.fftshift(Y)
frequencies = np.fft.fftshift(np.fft.fftfreq(len(y), d=1/fs))
plt.plot(frequencies*1e-3, np.abs(Y), label =f'10T max delay')

tau_d = math.floor(T / (Ts))
y = multipath_channel(x, 10, tau_d, 1e6)
Y = np.fft.fft(y) / fs
Y = np.fft.fftshift(Y)
frequencies = np.fft.fftshift(np.fft.fftfreq(len(y), d=1/fs))
plt.plot(frequencies*1e-3, np.abs(Y), label =f'T max delay')
plt.legend()



plt.show()