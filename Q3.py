import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm
import matplotlib.colors as mcolors


from Q2 import *

################################################################################
## Part a: Unweighted array gain
################################################################################
az = np.linspace(-180, 180, 600)
N = 8
w = np.ones(N)

gain = get_ula_gain(N, w, np.deg2rad(az))

plt.figure()
plt.title("Unweighted Array Gain")
plt.plot(az, np.abs(gain))
plt.xlabel("Azimuth")
plt.ylabel("Gain")
plt.grid(True)

plt.figure()
plt.polar(np.deg2rad(az), np.abs(gain))
plt.title("Unwieghted Array Gain ")


################################################################################
## Part b: Unweighted array gain; steered 30 degrees
################################################################################
w = -get_ula_response(N, np.deg2rad(30))
gain = get_ula_gain(N, w, np.deg2rad(az))

plt.figure()
plt.title("30 Degree Steering Gain")
plt.plot(az, np.abs(gain))
plt.xlabel("Azimuth")
plt.ylabel("Gain")
plt.grid(True)

plt.figure()
plt.polar(np.deg2rad(az), np.abs(gain))
plt.title("30 Degree Steering Gain")

################################################################################
## Part c: 4x8 unweighted
################################################################################
M = 4
N = 8
phi = np.linspace(-90, 90, 200)

## plot 1 phi  = 0
w = np.ones((4,8))
gain = get_upa_gain(M, N, w, np.deg2rad(az), [0])

plt.figure()
plt.title("Unweighted 4x8 Gain")
plt.plot(np.deg2rad(az), np.abs(gain))
plt.xlabel("Azimuth")
plt.ylabel("Gain")
plt.grid(True)

plt.figure()
plt.polar(np.deg2rad(az), np.abs(gain))
plt.title("Unweighted 4x8 Gain")

## Plot 2 theta = 0
gain = get_upa_gain(M, N, w, [0], np.deg2rad(phi))[0]

plt.figure()
plt.title("Unweighted 4x8 Gain")
plt.plot(np.deg2rad(phi), np.abs(gain))
plt.xlabel("Elevation")
plt.ylabel("Gain")
plt.grid(True)

plt.figure()
plt.polar(np.deg2rad(phi), np.abs(gain))
plt.title("Unweighted 4x8 Gain")

################################################################################
## Part d: steered planar
################################################################################
## plot 1 phi  = 0
w = -get_upa_response(M, N, np.deg2rad(30), np.deg2rad(45))
THETA, PHI = np.meshgrid(az, phi)
G = np.zeros(np.shape(THETA), dtype='complex')

for idx, phi in enumerate(PHI):
	G[idx] = get_upa_gain(M,N, w, [THETA[idx][0]], phi)[0]

# G = get_upa_gain(M, N, w, np.deg2rad(THETA), np.deg2rad(PHI))
# convert to cartesian

G = np.abs(G)

X = G * np.cos(PHI) * np.cos(THETA)
Y = G * np.cos(PHI) * np.sin(THETA)
Z = G * np.sin(PHI)
# create colors
plt.show() # pause program here
colorfunction = np.sqrt(X**2 + Y**2 + Z**2)
normcolor = mcolors.Normalize(colorfunction.min(), colorfunction.max())
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
plot = ax.plot_surface(X, Y, Z, facecolors=cm.jet(normcolor(colorfunction)), rstride=1, cstride=1, linewidth=0)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title("Weighted 4x8 Gain")




plt.show()