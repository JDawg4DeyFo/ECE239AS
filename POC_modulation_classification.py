# File: POC_modulation_classification.py
# Author: Jacob Dennon
# Date: November 25th, 2026
# Description: This file provides a proof of concept for various MC techniques.
# from remoterf.drivers.adalm_pluto import adi
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from OFDM import *
# Feature-based: cumulant approach
def cumulant_MC(frame):


	return M

# Average Likelihood Ratio Test
def ARLT_MC(frame):

	return M

# Parameters
N = 5
K = 1024
CP = 1/4
M = 16

# basic gen and guess
frame = get_OFDM_frame(N, K, CP, M)
M_cumulant = cumulant_MC(frame)
M_ARLT = ARLT_MC(frame)


# print results
print(f"Actual Modulation Scheme: {M}")
print(f"ARLT Guess: {M_ARLT}")
print(f"Cumulant Guess: {M_cumulant}")
