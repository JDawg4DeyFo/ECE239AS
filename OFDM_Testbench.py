# File: OFDM_Testbench.py
# Author: Jacob Dennon
# Date: November 13th, 2026
# Description: This file will run tests to verify all functions from OFDM.py
# are working correctly.
from OFDM import *
import numpy as np
import matplotlib.pyplot as plt

test_count = 0
pass_count = 0

################################################################################
## get_subcarrier_indices()
################################################################################




################################################################################
## get_OFDM_symbol()
################################################################################
K = 100
pilot_spacing = 4
CP = 1/4
M = 4
sps = 4
B = 10 # sampling frequency [MHz]

## Test Group 0: Qualitative Plotting
# Just plotting the output of the OFDM symbols
OFDM_Symbol, sequence1 = get_OFDM_symbol(K, CP, M, sps=1, pilot_spacing=pilot_spacing, return_sequence=True)
OFDM_Symbol4, sequence4 = get_OFDM_symbol(K, CP, M, sps=sps, return_sequence=True, pilot_spacing=pilot_spacing)

time1 = np.linspace(0, (len(OFDM_Symbol)/(B * 1)), len(OFDM_Symbol))
time4 = np.linspace(0, (len(OFDM_Symbol)/(B * sps)), len(OFDM_Symbol4))

plt.figure()
plt.title("Complex BB OFDM symbol")
plt.plot(time1, OFDM_Symbol, label="sps = 1")
plt.plot(time4, OFDM_Symbol4, label=f'sps = {sps}')
plt.xlabel("t (us)")
plt.ylabel("x[n]")
plt.grid(True)
plt.legend()

## Test Group 0a: Symbol Retrieval
# Now let's test and see if the symbols can be reconstructed from IFT
# Both symbols should give us what we expect
# Working with symbol 4 will be a good exercise in sps
# FFT to retrieve sequences...
retrieved_sequence1 = np.sqrt(K) * np.fft.fft(OFDM_Symbol[math.floor(CP * K):]) # discarding CP
retrieved_sequence4 = np.sqrt(K*sps) * np.fft.fft(OFDM_Symbol4[math.floor(CP*K*sps):])

test_count+= 1
if np.allclose(retrieved_sequence1, sequence1,rtol=0):
	pass_count += 1
	print("Retrieved sequence for sps 1 matches generated sequence")
else:
	print("Retrieved sequence for sps 1 does not match generated sequence")
	print("Comparing sequences...")
	for i in range(10):
		print(f"Retrieved: {retrieved_sequence1[i]:.4g} Generated: {sequence1[i]:.4g}")

	np.testing.assert_allclose(retrieved_sequence1, sequence1, rtol=0, atol=1e-8)
test_count+= 1
if np.allclose(retrieved_sequence4, sequence4,rtol=0):
	pass_count += 1
	print("Retrieved sequence for sps 1 matches generated sequence")
else:
	print("Retrieved sequence for sps 1 does not match generated sequence")
	print("Comparing sequences...")
	for i in range(10):
		print(f"Retrieved: {retrieved_sequence4[i]:.4g} Generated: {sequence4[i]:.4g}")

	np.testing.assert_allclose(retrieved_sequence4, sequence4, rtol=0, atol=1e-8)

# Now let's qualitativley plot them
frequencies = np.linspace(0, B / 1, len(retrieved_sequence1))
frequencies4 = np.linspace(0, B / sps, len(retrieved_sequence4))

plt.figure()
plt.title("Retrieved Symbols")
plt.plot(frequencies, retrieved_sequence1, label = 'sps = 1')
plt.plot(frequencies4, retrieved_sequence4, label='sps = 4')
plt.xlabel('MHz')
plt.ylabel('s[k]')
plt.grid(True)
plt.legend()


################################################################################
## get_OFDM_frame()
################################################################################
## Test Group 0: Qualatative Plotting
# Just seeing if things look good
OFDM_Frame = get_OFDM_frame(10, K, CP, M)
time = np.linspace(0, len(OFDM_Frame) / B, len(OFDM_Frame))

plt.figure()
plt.title("Complex BB OFDM Frame")
plt.plot(time, OFDM_Frame)
plt.xlabel("t (us)")
plt.ylabel("x[n]")
plt.grid(True)
# plt.legend()







print(f"Results: {pass_count}/{test_count} ({pass_count/test_count} %) tests passed")
plt.show()