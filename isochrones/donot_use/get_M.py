import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table
import matplotlib
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True

def flux_mag(mag, zp):
	newflux = [zp[n] * 10 ** (-mag[n]/2.5) for n in range(len(zp))]
	return newflux #outputs f_nu (Jy) as a vector

def CIT_to_jy(mags):
	zp = [1670, 980, 620, 280]
	newflux = flux_mag(mags, zp)
	return newflux

S_A = [8.60, 7.50, 6.56, 5.06] #prato et al 2003

S_A = CIT_to_jy(S_A)

S_B = [9.37, 8.28, 7.27, 6.07] #prato et al 2003

S_B = CIT_to_jy(S_B)

VV_ne = [12.07, 9.69, 7.33, 3.70] #prato et al 2003

VV_ne = CIT_to_jy(VV_ne)

VV_sw = [9.60, 8.38, 7.27, 6.28] #prato et al 2003

VV_sw = CIT_to_jy(VV_sw)

js = [S_A[0], S_B[0], VV_sw[0], VV_ne[0]]

Aj = [0.548, 0.548, 0.466, 2.796]

hs = [S_A[1], S_B[1], VV_sw[1], VV_ne[1]]
rs = [4.2, 2.3, 1.8, 3.1]

hs = [hs[n] * (1 + rs[n]) for n in range(len(rs))]

Ah = [0.34, 0.34, 0.29, 1.74]

for n, j in enumerate(js):
	ab_m = -2.5 * np.log10(j/3631)

	print('J: ', M)

	'''
	ab_m = -2.5 * np.log10(hs[n]/3631)

	M = ab_m - (5 *(np.log10(150) - 1)) - Ah[n]

	print('H: ', M)
	'''	