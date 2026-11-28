import numpy as np
import matplotlib.pyplot as plt

def get_ula_response(N, az):

	i = np.arange(N)

	a = np.exp(-1j * np.pi * i * np.sin(az))

	return a

def get_upa_response(M, N, az, el):

	a = np.zeros((M, N), dtype = complex)

	for r in range(M):
		for c in range(N):
			a[r][c] = np.exp(-1j * np.pi *(r * np.sin(el) + c * np.sin(az)))

	return a

def get_ula_gain(N, w, az):
	g = np.zeros(len(az), dtype='complex')
	for idx, az_i in enumerate(az):
		g[idx] = np.sum(np.conj(w) * get_ula_response(N, az))

	return g

def get_upa_gain(M, N, w, az, el):
	g = np.zeros((len(az), len(el)))

	for idx, az_i in enumerate(az):
		for jdx, el_i in enumerate(el):
			g[idx][jdx] = np.sum(np.conj(w) * get_upa_response(M, N, az_i, el_i))

	return g