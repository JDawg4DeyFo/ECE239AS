import numpy as np
import matplotlib.pyplot as plt

################################################################################
## Part a: SNR1 = SNR 2 = 10 dB uplink
################################################################################
SNR1 = 10
SNR2 = 10
a = np.linspace(0, 1, 100)

plt.figure()
plt.title(f"SE Region SNR1 = {SNR1} SNR2 = {SNR2}")
plt.xlabel("C1 SNR = 10 dB")
plt.ylabel("C2 SNR = 10 dB")
plt.grid(True)

############ TDMA
C1 = a * np.log2(1 + SNR1)
C2 = (1- a) * np.log2(1 + SNR2)

# plt main
plt.plot(C1, C2)

# plot max pt
max_idx = np.argmax(C1 + C2)
plt.scatter(C1[max_idx], C2[max_idx], marker='*')

############ FDMA
C1 = a * np.log2(1 + SNR1/ a)
C2 = (1 - a) * np.log2(1 + SNR2/(1 - a))

# plt main
plt.plot(C1, C2)
plt.xlabel("C1 SNR = 10 dB")
plt.ylabel("C2 SNR = 10 dB")

# plot max pt
max_idx = np.argmax(C1 + C2)
plt.scatter(C1[max_idx], C2[max_idx], marker='*')

############


################################################################################
## Part b: SNR1 = 20 dB SNR2 = 0 dB uplink
################################################################################
SNR1 = 100
SNR2 = 1

# find caps
C1 = a * np.log2(1 + SNR1/ a)
C2 = (1 - a) * np.log2(1 + SNR2/(1 - a))

# plt main
plt.figure()
plt.title("Spectral Efficiency Region")
plt.plot(C1, C2)
plt.xlabel("C1 SNR = 10 dB")
plt.ylabel("C2 SNR = 10 dB")
plt.grid(True)

# plot max pt
max_idx = np.argmax(C1 + C2)
plt.scatter(C1[max_idx], C2[max_idx], marker='*')
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