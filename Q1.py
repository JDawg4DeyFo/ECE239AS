import numpy as np
import matplotlib.pyplot as plt

def linear_interp(array):
	"""
	This function will return a linear interpolation for an array of the form:
	[x, 0, 0, 0, 0, x, 0, 0, 0, 0, x, 0, 0]

	Parameters
	----------
	array
		The array needing interpolation

	Returns
	-------
	array
		The interpolated array
	"""
	y0 = None
	y0_idx = 0
	y1 = None
	y1_idx = 0
	m = 0
	
	for idx, i in enumerate(array):

		# check if we're on last entry
		if idx == len(array):
			array[y1_idx:len(array)] = y1
			continue

		# if we run into non-zero value then we need to update anchor values
		if i != 0:
			# check if y0, y1 have been populated yet.
			if y0 == None:
				y0 = i
				# populate any values before y0 if idx(y0) != 0
				if idx != 0:
					array[0:idx] = [y0] * idx
					y0_idx = idx # store for when we find y1
				continue

			# if y1, then update and populate backlog
			elif y1 == None:
				y1 = i
				y1_idx = idx
				# update any previous  uninterp'd values
				m = (y1 - y0) / (idx - y0_idx)
				array[y0_idx:idx] = np.arange(0, idx - y0_idx) * m + y0
				continue

			# Normal operation
			y0 = y1
			y0_idx = y1_idx
			y1 = i
			y1_idx = idx

			# interpolate
			m = (y1 - y0) / (idx - y0_idx)
			array[y0_idx:idx] = np.arange(0, idx - y0_idx) * m + y0

	return array

################################################################################
## Part a: SNR1 = SNR 2 = 10 dB uplink
################################################################################
SNR1 = 10
SNR2 = 10
a = np.linspace(0.01, 0.99, 100)

plt.figure()
plt.title(f"Uplink SE Regions: SNR1 = {SNR1} SNR2 = {SNR2}")
plt.xlabel("C1 SNR = 10 dB")
plt.ylabel("C2 SNR = 10 dB")
plt.grid(True)

############ TDMA
C1 = a * np.log2(1 + SNR1)
C2 = (1- a) * np.log2(1 + SNR2)

# plt main
plt.plot(C1, C2, label="TDMA")

# plot max pt
max_idx = np.argmax(C1 + C2)
plt.scatter(C1[max_idx], C2[max_idx], marker='*')

############ FDMA
C1 = a * np.log2(1 + SNR1/ a)
C2 = (1 - a) * np.log2(1 + SNR2/(1 - a))

# plt main
plt.plot(C1, C2, label="FDMA")
plt.xlabel("C1 (bps/Hz)")
plt.ylabel("C2 (bps/Hz)")

# plot max pt
max_idx = np.argmax(C1 + C2)
plt.scatter(C1[max_idx], C2[max_idx], marker='*')

############ NOMA-SIC
C1_degrade = np.log2(1 + SNR1/(1 + SNR2))
C1_full = np.log2(1+ SNR1)

C2_degrade = np.log2(1 + SNR2 / (SNR1 + 1))
C2_full = np.log2(1 + SNR2)

C1 = np.concatenate((np.linspace(0.01, C1_degrade, 25) , [0] * 25 , [C1_full] * 25))
C2 = np.concatenate(([C2_full] * 25 , [0] * 25 , np.linspace(C2_degrade, 0.01, 25)))
C1 = linear_interp(C1)
C2 = linear_interp(C2)

plt.plot(C1, C2, label = "NOMA-SIC")
plt.legend()
################################################################################
## Part b: SNR1 = 20 dB SNR2 = 0 dB uplink
################################################################################
# SNR1 = 100
# SNR2 = 1

# # find caps
# C1 = a * np.log2(1 + SNR1/ a)
# C2 = (1 - a) * np.log2(1 + SNR2/(1 - a))

# # plt main
# plt.figure()
# plt.title("Spectral Efficiency Region")
# plt.plot(C1, C2)
# plt.xlabel("C1 SNR = 10 dB")
# plt.ylabel("C2 SNR = 10 dB")
# plt.grid(True)

# # plot max pt
# max_idx = np.argmax(C1 + C2)
# plt.scatter(C1[max_idx], C2[max_idx], marker='*')
################################################################################
## Part c: SNR1 = 10 dB SNR2 = 10 dB downlink
################################################################################
SNR1 = 10
SNR2 = 10



################################################################################
## Part d: SNR1 = 20 dB SNR2 = 0 dB downlink
################################################################################
SNR1 = 100
SNR2 = 1





plt.show()