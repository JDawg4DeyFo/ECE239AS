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
# Test Group 0: basic AWGN_Channel() tests
# The first test is just plotting retrieved OFDM symbols + noise to see if what
# we get makes sense
K = 100
pilot_spacing = 4
CP = 1/4
M = 4

# Build symbol
OFDM_Symbol, sequence1 = get_OFDM_symbol(K, CP, M, sps=1, pilot_spacing=pilot_spacing, return_sequence=True)
OFDM_Symbol = AWGN_Channel(OFDM_Symbol, 1e-7)
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


plt.show()