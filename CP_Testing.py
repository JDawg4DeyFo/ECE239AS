# File: CP_Testing.py
# Author: Jacob Dennon
# Date: November 24th, 2026
# Description: This file will demonstrate the importance of the cyclic prefix
# for OFDM systems
from remoterf.drivers.adalm_pluto import adi
import numpy as np
import matplotlib.pyplot as plt
import random
import math

################################################################################
## Helper functions
################################################################################
def gen_rand_qam_symbols(N, M):
	"""
	Return N random M-QAM symbols

	Parameters
	----------
	N
		The number of symbols to return
	
	M
		The modulation order
	
	Returns
	-------
	symbols
		A set of N random M-QAM symbols
	
	constellation
		The constellation from which symbols are drawn.
	"""
	constellation = {}
	symbols = []

	# basic defs
	order = math.ceil(math.log(M, 4))
	M_clean = int(math.pow(4, order))
	M_sqrt = int(math.sqrt(M_clean))
	A = math.sqrt(3 / (2 * (M_clean - 1)))


	# defining constellation
	# Note: not gray coded
	time_to_stop = 0
	for i in range(M_sqrt):			# i determines a
		
		# check if we're over user specified M
		if (time_to_stop == 1):
			break

		for j in range(M_sqrt):		# j determines b
			idx = j + (i * M_sqrt)

			if (idx >= M):
				time_to_stop = 1
				break

			a = A * (2 * i + 1 - M_sqrt)
			b = A * (2 * j + 1 - M_sqrt)

			constellation[idx] = complex(a, b)

	# Generate N symbols
	for i in range(N):
		symbols.append(constellation[random.randint(0, M - 1)])

	return np.array(symbols), constellation

################################################################################
# OFDM Subcarriers
# Return lists of indices
# Indices are in fft shifted form
# this function might need some revising
def get_subcarrier_indices(
	num_subcarriers,
	pilot_spacing,
	num_nulls_dc=1,
	num_nulls_lower=1,
	num_nulls_upper=1):
	"""
	This function is a helper function. It facilitates the construction of an
	OFDM symbol by returning appropriate indices

	Parameters
	----------

	num_subcarriers
		Number of sub carriers

	pilot_spacing
		spacing between pilots

	num_nulls_dc
		number of DC nulls

	num_nulls_lower
		number of lower band nulls

	num_nulls_upper
		number of upper band nulls
	
	"""
	indice_list = range(num_subcarriers)
	indice_set = set(indice_list) # will be used for data indices
	
	idx_subs_nulls = [0, num_subcarriers // 2, num_subcarriers // 2 + 1]
	indice_set -= set(idx_subs_nulls)

	idx_subs_pilots = indice_list[3::pilot_spacing]
	try:
		idx_subs_pilots.pop(idx_subs_pilots.index(0)) # remove DC null if there
		
	except AttributeError:
		pass

	indice_set -= set(idx_subs_nulls) # use this for data set

	idx_subs_data = list(indice_set)

	return np.asarray(idx_subs_data), np.asarray(idx_subs_pilots), np.asarray(idx_subs_nulls)

### setup code referenced from https://wireless.ee.ucla.edu/remoterf/
# ---------------------------------------------------------------
# Digital communication system parameters.
# ---------------------------------------------------------------
fs = 1e6     # baseband sampling rate (samples per second)
ts = 1 / fs  # baseband sampling period (seconds per sample)
sps = 10     # samples per data symbol
T = ts * sps # time between data symbols (seconds per symbol)

# ---------------------------------------------------------------
# Pluto system parameters.
# ---------------------------------------------------------------
sample_rate = fs                # sampling rate, between ~600e3 and 61e6
tx_carrier_freq_Hz = 915e6      # transmit carrier frequency, between 325 MHz to 3.8 GHz
rx_carrier_freq_Hz = 915e6      # receive carrier frequency, between 325 MHz to 3.8 GHz
tx_rf_bw_Hz = sample_rate * 1   # transmitter's RF bandwidth, between 200 kHz and 56 MHz
rx_rf_bw_Hz = sample_rate * 1   # receiver's RF bandwidth, between 200 kHz and 56 MHz
tx_gain_dB = -25                # transmit gain (in dB), beteween -89.75 to 0 dB with a resolution of 0.25 dB
rx_gain_dB = 40                 # receive gain (in dB), beteween 0 to 74.5 dB (only set if AGC is 'manual')
rx_agc_mode = 'manual'          # receiver's AGC mode: 'manual', 'slow_attack', or 'fast_attack'
rx_buffer_size = 100e3          # receiver's buffer size (in samples), length of data returned by sdr.rx()
tx_cyclic_buffer = True         # cyclic nature of transmitter's buffer (True -> continuously repeat transmission)

# ---------------------------------------------------------------
# Initialize Pluto object using issued token.
# ---------------------------------------------------------------
# SDR1 -> single tap loopback
# SDR2 -> multi tap loopback
# SDR3 -> multi tap loopback 2
try:
	sdr1 = adi.Pluto(token='Pluto_10_Loopback@ucla') # create Pluto object
	sdr1.sample_rate = int(sample_rate)   # set baseband sampling rate of Pluto
except:
	sdr1 = None
	print("Pluto 10 not available")

try:
	sdr2 = adi.Pluto(token='Pluto_12_MultiTap@ucla') # create Pluto object
	sdr2.sample_rate = int(sample_rate)   # set baseband sampling rate of Pluto
except:
	sdr2 = None
	print("Pluto 12 not available")

try:	
	sdr3 = adi.Pluto(token='Pluto_13_MultiTap@ucla') # create Pluto object
	sdr3.sample_rate = int(sample_rate)   # set baseband sampling rate of Pluto
except:
	sdr3 = None
	print("Pluto 13 not available")

# ---------------------------------------------------------------
# Setup Pluto's transmitter.
# ---------------------------------------------------------------
if sdr1:
	sdr1.tx_destroy_buffer()                   # reset transmit data buffer to be safe
	sdr1.tx_rf_bandwidth = int(tx_rf_bw_Hz)    # set transmitter RF bandwidth
	sdr1.tx_lo = int(tx_carrier_freq_Hz)       # set carrier frequency for transmission
	sdr1.tx_hardwaregain_chan0 = tx_gain_dB    # set the transmit gain
	sdr1.tx_cyclic_buffer = tx_cyclic_buffer   # set the cyclic nature of the transmit buffer

if sdr2:
	sdr2.tx_destroy_buffer()                   # reset transmit data buffer to be safe
	sdr2.tx_rf_bandwidth = int(tx_rf_bw_Hz)    # set transmitter RF bandwidth
	sdr2.tx_lo = int(tx_carrier_freq_Hz)       # set carrier frequency for transmission
	sdr2.tx_hardwaregain_chan0 = tx_gain_dB    # set the transmit gain
	sdr2.tx_cyclic_buffer = tx_cyclic_buffer   # set the cyclic nature of the transmit buffer

if sdr3:
	sdr3.tx_destroy_buffer()                   # reset transmit data buffer to be safe
	sdr3.tx_rf_bandwidth = int(tx_rf_bw_Hz)    # set transmitter RF bandwidth
	sdr3.tx_lo = int(tx_carrier_freq_Hz)       # set carrier frequency for transmission
	sdr3.tx_hardwaregain_chan0 = tx_gain_dB    # set the transmit gain
	sdr3.tx_cyclic_buffer = tx_cyclic_buffer   # set the cyclic nature of the transmit buffer

# ---------------------------------------------------------------
# Setup Pluto's receiver.
# ---------------------------------------------------------------
if sdr1:
	sdr1.rx_destroy_buffer()                   # reset receive data buffer to be safe
	sdr1.rx_lo = int(rx_carrier_freq_Hz)       # set carrier frequency for reception
	sdr1.rx_rf_bandwidth = int(sample_rate)    # set receiver RF bandwidth
	sdr1.rx_buffer_size = int(rx_buffer_size)  # set buffer size of receiver
	sdr1.gain_control_mode_chan0 = rx_agc_mode # set gain control mode
	sdr1.rx_hardwaregain_chan0 = rx_gain_dB    # set gain of receiver

if sdr2:
	sdr2.rx_destroy_buffer()                   # reset receive data buffer to be safe
	sdr2.rx_lo = int(rx_carrier_freq_Hz)       # set carrier frequency for reception
	sdr2.rx_rf_bandwidth = int(sample_rate)    # set receiver RF bandwidth
	sdr2.rx_buffer_size = int(rx_buffer_size)  # set buffer size of receiver
	sdr2.gain_control_mode_chan0 = rx_agc_mode # set gain control mode
	sdr2.rx_hardwaregain_chan0 = rx_gain_dB    # set gain of receiver

if sdr3:
	sdr3.rx_destroy_buffer()                   # reset receive data buffer to be safe
	sdr3.rx_lo = int(rx_carrier_freq_Hz)       # set carrier frequency for reception
	sdr3.rx_rf_bandwidth = int(sample_rate)    # set receiver RF bandwidth
	sdr3.rx_buffer_size = int(rx_buffer_size)  # set buffer size of receiver
	sdr3.gain_control_mode_chan0 = rx_agc_mode # set gain control mode
	sdr3.rx_hardwaregain_chan0 = rx_gain_dB    # set gain of receiver

################################################################################
## Creating OFDM Signals
################################################################################
# Idea: Create OFDM symbols using different lengths of CP. Create a matrix with
# len(CP) columns. Each row entry is a BB OFDM chip. We'll pass each OFDM symbol
# inside matrix through channel. After RX, we can find the EVM for each CP.

# OFDM Parameters
K = 100
M = 16
pilot_spacing = 4,
sps = 1,
num_nulls_dc=1,
num_nulls_lower=1,
num_nulls_upper=1,
CP = np.arange(0, int(K/2)) # Sweep CP from 0 to K /2
# Create baseline no CP OFDM symbol
# Get indices of subcarriers
pilot_spacing = 4 # maybe should make this a parameter
idx_subs_data, idx_subs_pilots, idx_subs_nulls = get_subcarrier_indices(
	K,
	pilot_spacing,
	num_nulls_dc=num_nulls_dc,
	num_nulls_lower=num_nulls_lower,
	num_nulls_upper=num_nulls_upper
)

# Generate symbols, pilots, and empty sequence
symbols,_ = gen_rand_qam_symbols(len(idx_subs_data), M)
pilots,_ = gen_rand_qam_symbols(len(idx_subs_pilots), M)
sequence = np.zeros(K * sps, dtype='complex')

# Populate the sequence
for idx, i in enumerate(idx_subs_data):
	sequence[i*sps:(i+1)*sps] = symbols[idx]

for idx, i in enumerate(idx_subs_pilots):
	sequence[i*sps:(i+1)*sps] = pilots[idx]

# Convert to OFDM
OFDM_baseline = (1/np.sqrt(K * sps)) * np.fft.ifft(sequence)

# Initialize data structures
# is this optimal? maybe not. Do I care? Not unless I'm working for Jane Street
OFDM_symbols = [0] * len(CP)

# populate OFDM structure
for idx, CP_i in enumerate(CP):
	L = math.floor(K * CP_i * sps)
	cyclical_prefix = OFDM_baseline[-L:]
	OFDM_symbols[idx] = np.concatenate((cyclical_prefix, OFDM_baseline))

################################################################################
## Transmit and Receive OFDM Signals
################################################################################
# three sets for three different channels
OFDM_symbols_RX1 = [0] * len(CP)
OFDM_symbols_RX2 = [0] * len(CP)
OFDM_symbols_RX3 = [0] * len(CP)

# Transmit symbols accross 
for idx, O_symbol in enumerate(OFDM_symbols):
	# scale symbol
	O_symbol = O_symbol / np.max(np.abs(O_symbol)) * 2**14
	
	# Transmit across 3 channels
	if sdr1:
		# Simple TX/RX
		sdr1.tx(O_symbol)
		sdr1.rx_destroy_buffer()
		for i in range(1):
			rx_data_ = sdr1.rx()
		rx_signal = sdr1.rx()
		
		# Crude timing sync
		# not sure if this will work under multi-tap.... hopefully it does
		start_idx = np.argmax(np.abs(np.correlate(rx_signal, O_symbol)))
		OFDM_symbols_RX1[idx] = rx_signal[start_idx:start_idx+len(O_symbol)]
		

	if sdr2:
		# Simple TX/RX
		sdr2.tx(O_symbol)
		sdr2.rx_destroy_buffer()
		for i in range(1):
			rx_data_ = sdr2.rx()
		rx_signal = sdr2.rx()
		
		# Crude timing sync
		# not sure if this will work under multi-tap.... hopefully it does
		start_idx = np.argmax(np.abs(np.correlate(rx_signal, O_symbol)))
		OFDM_symbols_RX2[idx] = rx_signal[start_idx:start_idx+len(O_symbol)]

	if sdr3:
		# Simple TX/RX
		sdr3.tx(O_symbol)
		sdr3.rx_destroy_buffer()
		for i in range(1):
			rx_data_ = sdr3.rx()
		rx_signal = sdr3.rx()
		
		# Crude timing sync
		# not sure if this will work under multi-tap.... hopefully it does
		start_idx = np.argmax(np.abs(np.correlate(rx_signal, O_symbol)))
		OFDM_symbols_RX3[idx] = rx_signal[start_idx:start_idx+len(O_symbol)]

################################################################################
## Processing OFDM Signals
################################################################################
for idx, O_symbol in enumerate(OFDM_symbols_RX1):
	# Discard CP
	O_symbol = O_symbol[CP[idx]:]

	# Take DFT
	O_symbol = np.sqrt(K) * np.fft.fft(O_symbol)
	O_symbol = np.fft.fftshift(O_symbol) # need to confirm whether fftshift is needed
	# if it is eneded, revise OFDM.py

	# Channel correction
	# We can find certain values of h[k] from pilotss
	# This becomes a problem of interpolation between known h[k]'s
	

