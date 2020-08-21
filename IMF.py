"""
.. module:: IMF
   :platform: Unix, Windows
   :synopsis: Synthetic population production

.. moduleauthor:: Kendall Sullivan <kendallsullivan@utexas.edu>

Dependencies: numpy, matplotlib, astropy, scipy, model_fit_tools_v2, MPI4py
"""

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os 
from glob import glob
from astropy import units as u
from matplotlib import rc
rc('text', usetex=True)
from scipy.stats import chisquare
from scipy.interpolate import interp1d, SmoothBivariateSpline, interp2d, griddata
from scipy.optimize import root, minimize, leastsq
from scipy.integrate import trapz, simps
import model_fit_tools_v2 as mft
import multiprocessing as mp
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.cm as cm
import extinction
from mpi4py import MPI
import time 
# from labellines import labelLine, labelLines

def redres(wl, spec, factor):
	"""Reduces spectral resolution to mimic pixellation at the surface of a CCD
	Args:
		wl (array): wavelength array 
		spec (array): flux array
		factor (int): factor by which to reduce the resolution

	Note:
		wl and spec must be the same length. For consistency in reducing the wavelength array it should be evenly spaced.

	Returns:
		wlnew, specnew (tuple): two lists, where the first is the reduced wavelength array and the second is the reduced-resolution flux array

	"""
	#initialize new arrays
	wlnew = []
	specnew = []

	#go through the wavelength array
	for i in range(len(wl)):
		#if I'm at the start of a new block of length (factor), add the wavelength value to the new array
		#and add a new entry to the spectrum array
		if i%factor == 0:
			wlnew.append(wl[i])
			specnew.append(spec[i])
		#otherwise, add the spectrum value to the pre-existing most recent entry in the spectrum list
		#basically we're integrating over a factor-length block of spectrum
		else:
			idx = int((i - i%factor)/factor)
			specnew[idx] += spec[i]

	#return the reduced-resolution spectrum and wavelength array 
	return wlnew, specnew

def make_salpeter_imf(massrange, exponent = 1.3):
	'''Makes a non-normalized power-law IMF using a given mass range and exponent. Not 100% convinced this works

	Args:
		massrange (array): range of masses for which to calculate the IMF
		exponent (float): the exponent of the power law. Enter the POSITIVE value - e.g., for a classic Salpeter IMF, enter 1.3. Default is 1.3.

	Note:
		This uses the logarithmic form of the IMF, so returns an IMF in terms of dN/d(log m), and uses a classic Salpeter exponent of 1.3,
		not the linear exponent equivalent value of 2.3.

	Returns:
		imf (array): an array of the IMF values for the mass range entered.

	'''
	#initialize the IMF array
	imf = []
	#for each mass, calculate the IMF value, given that dN/d(log m) = m^exponent
	for n, m in enumerate(massrange):
		imf.append(m**(-1 * exponent))
	#return the IMF array
	return np.array(imf)

def make_chabrier_imf(massrange):
	'''Makes a lognormal/power law IMF as per Chabrier (2003, PASP), where the IMF follows a lognormal below 1 solar mass
	and is a Salpeter power law above 1 solar mass. 

	Args: 
		massrange (array): range of masses for which to calculate the IMF

	Note:
		This returns an IMF that is not normalized. 

	Returns:
		imf (array): an array of the Chabier IMF values for the masses entered.

	'''

	#initialize the IMF array
	p = []
	#go through the entered mass array
	for m in massrange:
		#if the mass is less than or equal to a solar mass, calculate a lognormal using values presented by Chabrier 2003
		if m <= 1:
			#do lognormal
			p.append(0.097*np.exp(-((np.log(m) - np.log(0.3))**2)/(2 * 0.55**2)))
		else:
			#do Salpeter power law: x = 1.3 for the log version
			p.append(0.0095*m**-1.3)
	#return the IMF array
	return np.array(p)

def calc_pct(imf, wh = 'chabrier'):
	'''Calculates an IMF as a percentage range from 0.09 to 100 solar masses

	Args:
		imf (string): Can be 'c', 's', or 'pct'. If 'c' or 's', returns the mass range and a Chabrier or Salpeter normalized IMF. if 'pct' 
			uses wh to determine which IMF to use to calculate a percentage array.
		wh (string): can be 'chabrier' or 'salpeter'. creates a percent likelihood array.

	Returns:
		x, imf (arrays): if imf = 'c' or 's', returns the mass range array and the IMF array. If 'pct', returns either the Chabrier or Salpeter IMF as a percentage array.

	'''
	#initialize a mass range array
	x = np.arange(0.09, 100, 0.05)

	#make the two initial IMFs
	total_chab = make_chabrier_imf(x)
	total_sal = make_salpeter_imf(x)

	#normalize by the value at 1 solar mass for each IMF
	total_chab = [tc/total_chab[19] for tc in total_chab]
	total_sal = [ts/total_sal[19] for ts in total_sal]

	#calculate the percent of the total area under the lognormal and the power law
	chab = np.trapz(total_chab[0:19], x[0:19])/np.trapz(total_chab, x)
	sal = np.trapz(total_sal[19:-1], x[19:-1])/np.trapz(total_sal, x)
	#and sum the total area
	total = chab+sal

	#if the user is requesting a chabrier imf
	if imf == 'c':
		#return the mass range, and the chabrier IMF, normalized at 1 solar mass and again by its area
		return x, total_chab/np.trapz(total_chab, x)
	#if the user is requesting a salpeter IMF, do the same thing
	elif imf == 's':
		return x, total_sal/np.trapz(total_chab, x)
	
	#if the request is for a percentage of the IMF curve covered by either Chabrier or Salpeter
	elif imf == 'pct':
		#if the user is looking for the Chabrier number
		if wh == 'chabrier':
			#return the area covered by the Chabrier IMF, divided by the total area
			return chab/total
		#if they're looking for the same thing with Salpeter, return the area covered by the Salpeter IMF as a fraction of the total area
		elif wh == 'salpeter':
			return sal/total
		#otherwise, reject the keyword
		else:
			return "You messed something up" 

def get_params(mass, age, which = 'parsec'):
	'''Convert a mass and an age to a log(g), temperature, luminosity, and VRIJHK magnitudes by interpolating stellar evolutionary models. 
	Currently supports parsec or baraffe BT-Settl isochrones. STRONGLY RECOMMEND using PARSEC - the Baraffe model portion needs to be 
	rewritten.

	Args:
	mass (float): mass of the requested star in solar masses
	age (float): age of the star in megayears
	which (string): which model set to use. Default is 'parsec', options are 'baraffe' or 'parsec'

	Returns:
		temperature, log(g), luminosity, magnitudes (floats): returns all four parameters in the same line. Temperature is in
		Kelvin, log(g) is in dex, luminosity in log(luminosity), and an array of VRIJHK magnitudes.

	Note:
		In the same directory as the code, there must be a directory named "isochrones" containing the Baraffe isochrones, with each age 
		as a separate file, and another directory named "phoenix_isos" containing the phoenix BT-Settl isochrones, with a separate file for each age,
		and/or a document named "parsec_isos.dat" that contains the output from a run of the PARSEC evolutionary models, where the columns are 
		[Zini, logAge, Mini, int_IMF, Mass, logL, logTe, logg, label, mbolmag, Umag, Bmag, Vmag, Rmag, Imag, Jmag, Hmag, Kmag]. Not sure what exactly I
		did with the Baraffe models to use two different isochrones, but I recommend using the PARSEC models, as they have been much more extensively tested.
 
	'''
	#if the user requested the baraffe models
	if which == 'baraffe':
		#find all the isochrone files
		isos = glob('isochrones/*.txt')
		#initialize an ages array to hold the various options
		ages = []
		#retrieve the ages using the filenames 
		for file in isos:
			#the age in the filename is in 100's of kilo-years, so we need to divide by 10 to get Myr
			ages.append(int((file.split('_')[1]))/10)
		#sort that because glob doesn't pull files in order
		ages = np.sort(ages)

		#find the nearest age in the ages array to the requested age
		a1 = mft.find_nearest(ages, age)

		#and find the nearest age on the other side 
		#if the nearest age is greater than the requested age, the second grid point is the point before it in the ages array
		if ages[a1] > age:
			a2 = a1 - 1
		#otherwise it's the point after it
		else:
			a2 = a1 + 1

		#so now we have the two grid points between which the requested age falls
		#sort them to make sure that we have them in (min, max) order
		aa1 = ages[min(a1, a2)]
		aa2 = ages[max(a1, a2)]

		#read in the mass, luminosity, and radius arrays from the isochrones for the two ages, accounting for the weird formatting of the files I have
		m1, lum1, radius1 = np.genfromtxt(glob('isochrones/*{}*.txt'.format(str(int(aa1 * 10)).zfill(5)))[0], usecols =(0, 2, 4), comments = '!', unpack = True, autostrip = True)
		m2, lum2, radius2 = np.genfromtxt(glob('isochrones/*{}*.txt'.format(str(int(aa2 * 10)).zfill(5)))[0], usecols =(0, 2, 4), comments = '!', unpack = True, autostrip = True)

		#make an age vector with the single age value for each grid point that's the length of the mass array
		#this is just so we can interpolate more easily later
		aaa1, aaa2 = np.full(len(m1), aa1), np.full(len(m2), aa2)

		#do a 2d interpolation of luminosity in terms of mass and age to get a luminosity for the requested mass and age
		#do the same thing for radius
		a_l = griddata((np.hstack((m1, m2)), np.hstack((aaa1, aaa2))), np.hstack((lum1, lum2)), (mass, age))
		a_r = griddata((np.hstack((m1, m2)), np.hstack((aaa1, aaa2))), np.hstack((radius1, radius2)), (mass, age))

		#units: solar masses, kelvin, solar luminosity, log(g), giga-centimeters (NOT SOLAR RADII)
		#THUS assume solar radius = 6.957e5 km = 6.957e10 cm = 69.75 Gcm
		#retrieve the mass, effective temperature, radius, luminosity, and log(g) from the other isochrone set 
		m_real1, teff1, lu1, logg1, rad1 = np.genfromtxt(glob('phoenix_isos/*{}*.txt'.format(str(int(aa1*10)).zfill(5)))[0], \
			usecols = (0, 1, 2, 3, 4), autostrip = True, unpack = True)

		m_real2, teff2, lu2, logg2, rad2 = np.genfromtxt(glob('phoenix_isos/*{}*.txt'.format(str(int(aa2*10)).zfill(5)))[0], \
			usecols = (0, 1, 2, 3, 4), autostrip = True, unpack = True)

		#trim off the lowest mass value
		teff1, lu1, logg1 = teff1[1:-1], lu1[1:-1], logg1[1:-1]
		#convert radius to solar raidus
		rad1 = [np.around(r/69.75, 2) for r in rad1] #convert to solar radius

		#do this for both age points
		teff2, lu2, logg2 = teff2[1:-1], lu2[1:-1], logg2[1:-1]
		rad2 = [np.around(r/69.75, 2) for r in rad2] #convert to solar radius

		#make an age vector that's the length of the luminosity etc. vectors
		aaa1, aaa2 = np.full(len(lu1), aa1), np.full(len(lu2), aa2)

		#if the luminosity is in the range of the luminosity array
		if a_l >= lu1[0] and a_l <= lu1[-1] and a_l >= lu2[0] and a_l <= lu2[-1]:
			#get a temperature and a log(g) from interpolating them as a function of luminoisity and age
			temp = griddata((np.hstack((lu1, lu2)), np.hstack((aaa1, aaa2))), np.hstack((teff1, teff2)), (a_l, age))
			log_g = griddata((np.hstack((lu1, lu2)), np.hstack((aaa1, aaa2))), np.hstack((logg1, logg2)), (a_l, age))
			#and return the temperature, log(g), and luminosity
			return temp, log_g, a_l

		#if the luminosity is out of the range of one of the age's luminosity arrays
		else:
			#if it's in the range of the first grid point, use the closest value within that age's luminosity array
			if a_l > lu1[0] and a_l < lu1[-1]:
				idx = mft.find_nearest(a_l, lu1)
				temp, log_g = teff1[idx], logg1[idx]
			#if it's in the range of the second grid point, use the nearest value within that age's luminosity array
			elif a_l > lu2[0] and a_l < lu2[-1]:
				idx = mft.find_nearest(a_l, lu2)
				temp, log_g = teff2[idx], logg2[idx]
			#otherwise use the maximum luminosity value within that age and return the associated temperature and log(g)
			else:
				print('luminosity is out of range, using maximum')
				idx = np.where(np.hstack((lu1, lu2)) == max(np.hstack((lu1, lu2))))
				temp, log_g = np.hstack((teff1, teff2))[idx], np.hstack((logg1, logg2))[idx]
			#return the temperature, log(g), luminosity
			return temp, log_g, a_l

	#if I want to use the PARSEC models (recommended)
	if which == 'parsec':
		#reformat the age for easier file fiddling
		age = np.log10(age * 1e6)
		#read in the full isochrone file - this thing is huge and contains all age tracks in a single file
		matrix = np.genfromtxt('parsec_isos.dat', autostrip = True)
		#get just the age column - this has many duplicate ages, because each mass associated with a given track has that age assigned to it
		#so we want to go through and get just unique age values for each track
		ages = matrix[:, 1]
		#initialize an array to hold the unique age values and put the lowest age value into it to start
		aa = [ages[0]]
		#if I haven't already appended a given age, add it to the unique age array
		for a in ages:
			if a != aa[-1]:
				aa.append(a)
		#just to be safe, sort that array
		aa = np.sort(aa)
		#get the mass, luminosity, temperature, and log(g) values from the matrix
		#don't love that this is hardcoded but the parsec GUI outputs a standard format so hopefully won't be a problem for other users
		ma, logl, logt, lg = matrix[:, 4], matrix[:, 5], matrix[:, 6], matrix[:,7]

		#find the nearest grid point to the input age
		a1 = mft.find_nearest(aa, age) 
		#if that grid point is larger than the age, make the other grid point fall below the requested age
		if aa[a1] > age:
			a2 = a1 - 1
		#if the grid point that I've already found is younger than the requested age, place the other one larger than the requested age
		elif aa[a1] < age:
			a2 = a1 + 1
		#just in case, if something falls exactly on a grid point, set the two values equal
		#in its current iteration, the code doesn't like when the requested age is exactly on a grid point
		else:
			a1 = a2

		#as long as it's not a grid point
		if a1 != a2:
			#sort the ages to make sure everything is in order
			age1 = aa[min(a1, a2)]
			age2 = aa[max(a1,a2)]

			#get age, mass, log(temperature), log(g), and log(luminosity) vectors for the desired ages from the large (all-age) mass, temp, lum, and log(g) vectors
			ages1, ages2 = ages[np.where(ages == age1)], ages[np.where(ages == age2)]
			ma1, ma2 = ma[np.where(ages == age1)], ma[np.where(ages == age2)]
			lt1, lt2 = logt[np.where(ages == age1)], logt[np.where(ages == age2)]
			lg1, lg2 = lg[np.where(ages == age1)], lg[np.where(ages == age2)]
			logl1, logl2 = logl[np.where(ages == age1)], logl[np.where(ages == age2)]

			#get the magnitude arrays for the desired ages, and convert them into fluxes so we're not interpolating in log space
			mag1 = [10 ** (-0.4 * matrix[:, n][np.where(ages == age1)]) for n in range(12, 18)] #VRIJHK
			mag2 = [10 ** (-0.4 * matrix[:, n][np.where(ages == age2)]) for n in range(12, 18)]

			#convert log(luminosity) and log(temperature) into linear space so we're not interpolating in log space
			logl1, logl2 = [10 ** l for l in logl1], [10 ** l for l in logl2]
			lt1, lt2 = [10 ** l for l in lt1], [10 ** l for l in lt2]

			#interpolate temperature as a function of mass and age to retrieve a temperature for the input age and mass
			a_t = griddata((np.hstack((ages1, ages2)), np.hstack((ma1, ma2))), np.hstack((lt1, lt2)), (age, mass))
			#do the same for log(g), luminosity, and the magnitudes
			#interpolating each filter "magnitude" (in flux space currently) separately
			logg = griddata((np.hstack((ages1, ages2)), np.hstack((ma1, ma2))), np.hstack((lg1, lg2)), (age, mass))
			a_l = griddata((np.hstack((ages1, ages2)), np.hstack((ma1, ma2))), np.hstack((logl1, logl2)), (age, mass))
			mm = [griddata((np.hstack((ages1, ages2)), np.hstack((ma1, ma2))), np.hstack((mag1[n], mag2[n])), (age, mass)) for n in range(len(mag1))]

			#convert the magnitudes back from fluxes to mags
			mm = [-2.5 * np.log10(mm[n]) for n in range(len(mm))]
			#convert the luminosity back into a log(luminosity) because that's what the rest of the code is set up to handle
			a_l = np.log10(a_l)

			#return the temperature, log(g), log(luminosity) and VRIJHK magnitudes
			return a_t, logg, a_l, mm

		#if this falls at a grid point, we don't need to do a 2d interpolation, we just need to interpolate along a single isochrone
		else:
			#define an ages vector so I can pick out the other vectors more easily
			ages1 = ages[np.where(ages == age1)]
			#get the mass, temp, log(g), and log(lum) vectors for the correct age
			ma1, lt1, lg1, logl1 = ma[np.where(ages == age1)], logt[np.where(ages == age1)], lg[np.where(ages == age1)], logl[np.where(ages == age1)]
			#get the magnitudes for the right age and convert them to fluxes
			mag1 = [10 ** (-0.4 * matrix[:, n][np.where(ages == age1)]) for n in range(12, 18)]
			#convert the luminosity and temperature to linear values so I'm interpolating correctly
			logl1, lt1 = [10 ** l for l in logl1], [10 ** l for l in lt1]

			#interpolate everything to get new temperature, log(g), and luminosity values
			t_int = interp1d(ma1, lt1); a_t = t_int(mass)
			g_int = interp1d(ma1, lg1); logg = g_int(mass)
			l_int = interp1d(ma1, logl1); a_l = l_int(mass)
			#initialize a magnitudes list
			mm = []
			#loop through the mags, interpolate each filter individually to get a best magnitude
			for m in mag1:
				mag_int = interp1d(ma1, m)
				mm.append(mag_int(mass))
			#convert the magntiudes back to magnitudes from fluxes
			mm = [-2.5 * np.log10(p) for p in mm]
			#take the log of the luminosity to return because that's what the rest of the code expects
			a_l = np.log10(a_l)

			#return the temperature, log(g), log(luminosity), and magnitudes
			return a_t, logg, a_l, mm

def match_pars(temp1, temp2, lf, lg = 4):
	'''For a given pair of temperatures and a set luminosity ratio, produces a composite spectrum. 
	Uses a default log(g) of 4, and assumes that both spectra have the same log(g)

	Args:
		temp1 (float): primary spectrum temperature
		temp2 (float): secondary spectrum temperature
		luminosity_ratio (float): the fraction of the primary's brightness to make the secondary: this is the value the secondary flux is multiplied by.
		log_g (float): the desired log(g) of the spectrum. Values between 3 and 5.5 are allowed, default is 4.

	Note:
		Assumes both spectra are on the exact same wavelength grid. 

	Returns:
		wl, spec (lists): a wavelength array and a composite spectrum array. 

	'''

	#Retrieve the two spectra using the provided temperature and log(g)
	#get a generous portion of spectrum 
	wl1, spec1 = mft.get_spec(temp1, lg, [0.45, 2.5], normalize = False, reduce_res = False, resolution = 3000)
	wl2, spec2 = mft.get_spec(temp2, lg, [0.45, 2.5], normalize = False, reduce_res = False, resolution = 3000)

	#reduce the secondary flux by the provided factor
	spec2 *= lf

	#and add the two spectra
	spec = [spec1[n] + spec2[n] for n in range(len(spec1))]

	#return the wavelength and composite spectrum arrays
	return wl1, spec

def get_secondary_mass(pri_mass, power_law = True):
	'''Given a primary mass, randomly assigns it a secondary star mass using the values in the histogram from Raghavan et al. (2010) Fig. 16.
	Could probably stand to be updated to better statistics for Taurus at some point - these values are for field FGK stars.

	Args:
		primary_mass (float): Mass of the primary star, in whatever units you want.

	Note:
		While the secondary mass is less than the classical hydrogen burning limit (0.08 solar masses), the mass is drawn again, so I don't have to 
		deal with parameters that fall below the PARSEC isochrone limits.

	Returns:
		sm (float): The secondary mass, in whatever units the input primary mass was given.

	'''
	#set an initial secondary mass variable to 0
	sm = 0

	#if we want to use a more precise mass ratio distribution, we can model it as a power law (Moe and de Stefano 2017)
	#right now, use the upper sco power law from Tokovinin and Briceno 2020, which has an exponent of 0.4
	#technically it's only tested between 0.4 and 1.5 msun, and if we want to be more precise we could make it separation-dependent, but I'm going to wait on that	
	if power_law == True:
		#make the full possible range of mass ratio distributions
		q_range = np.linspace(1e-3, 1, 1000)
		#then make the probability distribution
		q_dist = [q ** (0.4 * pri_mass) for q in q_range] #alter based on primary mass - A stars have a steeper distribution than M stars
		#from the PDF, make the CDF by summing
		q_cdf = [np.sum(q_dist[0:n]) for n in range(len(q_dist))]
		#then normalize the CDF
		q_cdf_norm = [q/np.sum(q_cdf) for q in q_cdf]
		#while we're not in the stellar regime - this is ok because we don't care about BD or planetary companions
		while sm < 0.09:
			#draw a random number and find which box of the CDF it falls into
			#then find the corresponding q
			rn = np.random.rand()
			mr = 0
			for n in range(1, len(q_cdf_norm)):
				if rn > q_cdf_norm[n-1] and rn < q_cdf_norm[n]:
					mr = q_range[n]
			sm = pri_mass * mr

	#otherwise, if we just want to use the almost flat mass ratio distribution from Raghavan, we can do that
	else:
		#we want to redraw if the computed secondary has a substellar mass
		while sm < 0.09:
			#make a new random seed and draw a random number to select what regime of the Figure 16 histogram we fall into
			r = np.random.RandomState()
			rn = r.random_sample(size=1)[0]
			#if the random number is in the lowest bin, draw a small mass ratio
			if rn < (5/110):
				mr = r.uniform(low=0.05, high = 0.2)
			#if it's particularly large, draw a large mass ratio
			elif rn > 0.9:
				mr = r.uniform(low=0.95, high = 1)
			#otherwise just draw a number wherever else in the range of mass ratios
			else:
				mr = r.uniform(low=0.2, high = 0.95)
			#the secondary mass is just the primary mass times the mass ratio
			sm = pri_mass * mr
		#once it's above 0.09 solar masses, return the secondary mass

	return sm

def get_distance(sfr):
	'''Assigns a random distance within some set range to a star to mimic the random spatial distribution of stars in a star forming region

	Args:
		sfr (string): which star forming region to simulate. Currently supports "taurus", which has a random uniform distance range of 125 - 165 pc,
		and 'usco' (upper sco), which has a Gaussian distance distribution centered at 140 pc with a spread of 15 pc

	Returns:
		dist (float): random distance within the predefined bounds of the SFR

	'''
	#initialize a random seed
	r = np.random.RandomState()
	#check which star forming region is requested
	if sfr == 'taurus':
		#use the distance range from Galli 2019
		return r.randint(low=125, high=165)
	elif sfr == 'usco':
		return r.normal(loc = 140, scale = 20)

def make_mass(n_sys, reduce_m = False, m_reduce = 0.8):
	'''For a given number of systems, make a mass distribution using a Chabrier IMF

	Args:
		n_sys (int): number of systems to simulate

	Returns: 
		masses (array): array of masses, which has a length equal to the number of requested systems.

	'''
	#initialize a random number generator
	r = np.random.RandomState()
	#make an acceptable range of masses to draw from 
	#this could be done more nicely but for now it's where I functionally set the temperature limits for systems 
	#which is why the upper limit is 2.5 solar masses - corresponds to ~ 7000 K, or an early G - way more massive SpT
	# than what's found in Taurus at this age, even if stars that massive are forming there
	masses = np.linspace(0.09, 2, 5000)
	#make the probability distribution associated with a Chabrier IMF
	prob_dist = make_chabrier_imf(masses)

	#to reduce the number of M stars, we want to modify the PDF so that only 1/4 of the standard number of M stars is drawn
	if reduce_m == True:
		#find the region of the PDF that is below some maximum reduction mass
		p = prob_dist[np.where(masses <= m_reduce)]
		#then just cut the pdf into a quarter there - the normalization will ensure that this doesn't wreck anything else
		pp = [r/24 for r in p]
		#then just plug back in
		prob_dist[np.where(masses <= m_reduce)] = pp

	#normalize the probability distribution by its area, so that it sums to 1
	pd = [d/np.sum(prob_dist) for d in prob_dist]
	#create a cumulative distribution function by summing progressively larger regions of the CDF 
	cumu_dist = [np.sum(pd[:n]) for n in range(len(pd))]
	#then draw n_sys random numbers
	r_num = r.random_sample(size = n_sys)
	#initialize a mass array
	mass = []
	#then for each random number
	for rn in r_num:
		#initialize an index for going through the mass distribution later
		idx = 0
		#run through the CDF and check if the random number falls inside each box
		for n in range(0, len(cumu_dist) -1):
			#if it's inside that box, set idx to the correct index value for the mass array
			if rn > cumu_dist[n] and rn <= cumu_dist[n + 1]:
				idx = n
		#add the correct mass to the returned mass list
		mass.append(masses[idx])
	#return the array of masses created from the CDF of the IMF
	return mass

# m_nocutoff = make_mass(4000)
# mc = np.array(make_mass(1200, reduce_m = True))
# ms = mc[np.where(mc <= 0.8)]
# n = 0
# while n < 4:
# 	mc = np.hstack((mc,ms))
# 	n+=1
# m_cutoff = make_mass(1000, reduce_m = True)
# print(len(mc), len(m_nocutoff))
# plt.hist(mc, label = 'reduced, corrected')
# plt.hist(m_nocutoff, label = 'no reduction', alpha = 0.9)
# plt.hist(m_cutoff, label = 'reduced')
# plt.legend(loc='best')
# plt.savefig('correction_validation.png')

def unres_bin(sep, dist):
	if sep/dist <= 2:
		return True
	else:
		return False

def make_binary_sys(n, n_sys, multiplicity, mass_bins, age, av, run, sfr = 'usco', model = 'parsec', reg = [0.45, 0.9], res = 3000, reduce_ms = False, m_reduce = 0.7):
	'''Creates a population of single and binary star composite spectra, and records their physical properties. Optimized for handling 1 system at a time
	- would require a little fiddling to get it to work with multiple systems at once. Uses values from Raghavan + 2010. From that paper, eccentricity 
	is a uniform distribution, the period distribution can be approximated as a Gaussian with a peak at log(P) = 5.03 and sigma_log(P) = 2.28, and the 
	separation between 10 and 1000 AU is uniform. To better model Taurus, these statistics should be replaced by the values in something like 
	Duchene + Kraus 2013 at some point. 

	Args:
		n (int): number relative to total number of systems (i.e. system n of total)
		n_sys (int): number of systems to simulate in this instance of the function
		multiplicity (array): array of multiplicity fractions for the user-defined mass bins. must have len = len(mass_bins) + 1
		mass_bins (array): defines the points where the multiplicity fraction changes in solar masses
		age (array): array of length 2 - the low and high ends of the desired age distribution, which will be uniform between the two points.
		av (float): extinction value for the system(s) in magnitudes
		run (string): directory to write the output files to
		sfr (string): which star forming region to use for distance determination. Default (and current only accepted value) is 'taurus'
		model (string): which set of models to use. Default is 'parsec'. 'baraffe' is allowed but not recommended.
		reg (array): Spectral region to use in units of angstroms. Default is 4500 to 9000 A
		res (int): spectral resolution. Default is 3000

	Note:
		Does not return a spectrum but does write the system spectrum (so either a single star or the composite binary spectrum) to a "spec" file.

	Returns:
		pri_pars, sec_pars (arrays): If system is a binary, returns both primary and secondary star attributes. Otherwise returns the primary star 
		properties and 0 for the secondary star properties. The values of the primary array are, in order, 
		[num, p_mass (solar masses), multiplicity (1 or 0), age (Myr), av (mag), temp (K), logg, luminosity (solar), distance (pc), V mag, R mag, I mag, J mag, H mag, K mag].
		The secondary array contains the following: [num, s_mass, sep (AU), age, eccentricity, log(period) (days), temp, logg, log(luminosity), V mag, R mag, I mag, J mag, H mag, K mag],
		where if units are not noted they are the same as the primary array. The magnitudes given are extincted apparent magnitudes, 
		using Cardelli et al 1989 to determine the extinction in each filter.

	'''

	#initialize a random seed
	r = np.random.RandomState()

	#get an age from the given range
	#this is uniform right now but if a population is more clustered you'd want to change that
	age = r.normal(np.mean(age), (max(age) - min(age))/2)

	#define the different columns for the output files
	#I don't actually write these to any files, but I keep them here to retain a note of what I'm writing to and where
	#probably want to add them to a file at some point to facilitate usability. 
	#these two are for the primary star
	pri_array_keys = ['num', 'p_mass', 'multiplicity', 'age', 'av', 'temp', 'logg', 'luminosity', 'distance', 'v mag', 'r mag', 'i mag', 'j mag', 'h mag', 'k mag']
	p_arr_keys2 = ['#a multiplicity of 1 indicates a multiple at all - it\'s a flag, not a number of stars in the system\n #distance is in pc\
	\n #magnitudes are apparent at the distance assigned']

	#this defines the range of contents for the secondary array
	kw = ['num', 's_mass', 'sep', 'age', 'eccentricity', 'period', 'temp', 'logg', 'luminosity', 'v mag', 'r mag', 'i mag', 'j mag', 'h mag', 'k mag']

	#now make arrays to actually hold the primary and secondary system values
	#we obviously want them to be the same length as the number of keywords we're going to add to them later
	pri_pars = np.empty(len(pri_array_keys))
	sec_pars = np.empty(len(kw))

	#now make the primary star(s) by drawing from the IMF
	mass = make_mass(n_sys, reduce_m = reduce_ms, m_reduce = m_reduce)
	
	#now open a file to write the spectrum to in a few steps
	#name it spec_n.txt where n is the "system number" input above. This is useful because I typically use this to create 1 system at a time,
	#but am making many systems, so this is how I keep track of which system I'm actually creating
	spec_file = open(os.getcwd() + '/' + run + '/specs/spec_{}.txt'.format(n), 'w')

	#now get the observables using the age and mass.
	#depending on which model was requested
	ptemp, plogg, plum, mags = get_params(mass, age, which = model)

	#then get the system distance 
	dist = get_distance(sfr)
	#and calculate the distance modulus for modifying the magnitudes later
	dm = 5 * np.log10(dist/10) - 5
	#start writing system properties to a list
	pri_par = [n, mass[n_sys -1], 0, age, av, float(ptemp), float(plogg), float(plum), dist]
	#including adding the magnitudes
	[pri_par.append(float(mags[n])) for n in range(len(mags))]
	#then add it to the full original primary parameters array - this is a residual from when I was making multiple systems at this stage. Could be cleaned up.
	pri_pars = np.vstack((pri_pars, np.array(pri_par)))

	#now make sure that the mass function actually returned a number
	if type(pri_par[1]) != None:
		#make the primary wavelength array and spectrum
		#for now I'm hard coding the wavelength regime, and get_spec defaults to a reduced-resolution, broadened spectrum of R ~ 3000
		pw, ps = mft.get_spec(ptemp, plogg, reg, normalize = False, resolution = res)
		#convert the spectrum, which for some reason was getting returned as strings, into floats
		#not sure if this is still necessary but it basically takes zero time so not going to worry about it right now
		pri_wl, pri_spec = [],[]
		for k in range(len(pw)):
			pri_wl.append(float(pw[k]))
			pri_spec.append(float(ps[k]))

		#calculate a stellar radius using the temperature and luminosity: we know 4 pi r^2 L = sigma T^4 
		rad = (1 /(2*np.sqrt(np.pi * 5.67e-5))) * np.sqrt((10**plum) * 3.826e33)/ptemp**2
		#The spectra are retrieved in units per surface area, so here I'm multiplying by the stellar surface area to get a total surface integrated luminosity
		pri_spec = [float(ps * 4 * np.pi * rad**2) for ps in pri_spec]

		#initialize a "combined spectrum" variable that is currenty just the primary star spectrum
		comb_spec = pri_spec
		#now figure out which multiplicity fraction I need to use by checking which mass regime the assigned stellar mass falls into, and grabbing that multiplicity fraction
		mf = 0

		if mass[n_sys - 1] > mass_bins[-1]:
			mf = multiplicity[-1]
		else:
			for k, m in enumerate(mass_bins):
				if mass[n_sys - 1] < m:
					mf = multiplicity[k]

		#get a random number
		num_rand = r.random_sample()
		sep = 10 ** (r.normal(3 * mass[n_sys - 1] - 0.79, 1)) #from fitting a line to the results from De Rosa+ 2014, Raghavan+ 2010, and Winters+ 2019
		#and if the random number is smaller than the multiplicity fraction, we assign it as a binary and now have to assign the secondary some properties
		if mf >= num_rand and unres_bin(sep, dist) == True:
			pri_pars[n_sys][2]= 1

			sec_par = np.empty(len(kw))
			
			#give the secondary the same number as the primary so we can match them up again later
			sec_par[0] = n
			#get the secondary mass using the slightly-non uniform mass ratio distribution from Raghavan
			sec_par[1] = get_secondary_mass(mass[n_sys - 1])
			#assign a separation in AU
			sec_par[2] = sep
			#record the age in a second place
			sec_par[3] = age
			#assign an eccentricity
			sec_par[4] = r.uniform(0, 1)
			#and assign a period in units of log(days)
			sec_par[5] = r.normal(5.03, 2.28)

			#then, like above, use the mass and age to infer observables
			stemp, slogg, slum, smags = get_params(sec_par[1], age, which = model)
			#write everything to the array 
			sec_par[6:9] = stemp, slogg, slum
			sec_par[9:] = smags

			#and write it to the master array for secondary parameters
			sec_pars = np.vstack((sec_pars, sec_par))

			#get the spectrum and wavelength vector and make sure they're floats
			sw, ss = mft.get_spec(stemp, slogg, reg, normalize = False, resolution = res)
			sec_wl, sec_spec = [],[]
			for n in range(len(pw)):
				sec_wl.append(float(sw[n]))
				sec_spec.append(float(ss[n]))

			#calculate the secondary radius and multiply the spectrum by the surface area
			rads = (1 /(2* np.sqrt(np.pi * 5.6704e-5))) * np.sqrt((10**slum) * 3.826e33)/stemp**2
			sec_spec = [float(ps * 4 * np.pi * rads**2) for ps in sec_spec]

			#find the smallest wavelength spacing in the primary spectrum wavelength array
			mindiff = np.inf
			for k in range(1, len(pri_wl)):
				if pri_wl[k] - pri_wl[k-1] < mindiff:
					mindiff = pri_wl[k] - pri_wl[k-1]

			#define a new, common, wavelength vector, accounting for the fact that the primary and secondary arrays might start and end in slightly different places
			wl = np.arange(max(pri_wl[0], sec_wl[0]), min(pri_wl[-1],sec_wl[-1]), mindiff)

			#and interpolate them to that common wavelength array to make sure I can add them without messing anything up
			i1, i2 = interp1d(pri_wl, pri_spec), interp1d(sec_wl, sec_spec)
			new_pri, new_sec = i1(wl), i2(wl)

			#now add the primary and secondary spectra to get a combined spectrum
			#the flux ratio has been set by the multiplication by surface area
			pri_wl, comb_spec = wl, [new_pri[n] + new_sec[n] for n in range(len(wl))]

		#regardless of the extinction value, we can do this extinction step - if it's zero, it'll just add the distance modulus

		#first define the factors to use for the broadband filter extinction calculation
		factor = [1, 0.751, 0.479, 0.282, 0.190, 0.114] #cardelli et al. 1989

		#calculate an apparent magnitude for the 6 broadband magnitudes I'm measuring using the distance modulus and av*factor for the cardelli factors 
		#do this for the secondary too if I have one
		for n in range(len(mags)):
			pri_pars[n_sys][n - 6] = (av * factor[n]) + dm + mags[n] 

			if len(sec_pars.shape) == 1:
				sec_pars[n-6] = (av * factor[n]) + dm + sec_pars[n-6]

			else:
				sec_pars[n_sys][n-6] = (av * factor[n]) + dm + sec_pars[n_sys][n-6]

		#but we don't need to manually extinct the spectrum unless the av is > 0
		if av > 0:
			#make sure the spectrum and wavelength vector are numpy objects, then extinct the spectrum
			pri_wl, comb_spec = np.array(pri_wl), np.array(comb_spec)
			comb_spec = mft.extinct(pri_wl, comb_spec, av)

		#make sure nothing weird happened and the spectrum is negative - shouldn't be! 
		comb_spec = [np.abs(c) for c in comb_spec]

		#write the wavelength and spectrum to the spectrum file
		#since it's not a numpy write function, I have to make the values strings before I can write 
		#not sure this is actually true but I trust past me....enough to leave this as is
		for k in range(len(comb_spec)):
			spec_file.write(str(pri_wl[k]) + ' ' + str(comb_spec[k]) + '\n')
		spec_file.close()

		n += 1
	
	#if for some reason the mass generation didn't work, try again
	else:
		mass[n] = make_mass(1)

	#now check if there's a secondary
	#if there is, return both the primary and secondary properties
	if pri_pars[n_sys][2] == 1:
		return pri_pars[1], sec_pars[1]
	#otherwise return the primary properties
	else:
		return pri_pars[1]

def find_extinct(model_wl, model, data):
	'''DEPRECATED. Uses a basic simplex fit to find a best-fit extinction value. NOT UPDATED TO REFLECT CURRENT CODE REQUIREMENTS

	Args:
		model_wl (array): wavelength vector
		model (array): test spectrum
		data (array): input spectrum (the one I'm trying to find the best fit for)

	Returns: 
		model (array): test spectrum, and best extinction value (float) as a tuple

	'''
	#begin with some initial extinction guess, and define an initial step size
	init_guess = 2
	step = 0.5
	#also initialize a count
	niter = 0

	#these numbers are changeable depending on goal
	#but basically while the values are greater/less than these numbers the fitting routine will keep going, trying to find the best fit
	while step > 0.2 and niter < 10:
		#first print the current extinction value and which step we're on
		print('extinction: ', init_guess, step)
		#convert magnitudes of extinction to flux
		#making three different guesses: one at the guess, one 1 step above, one 1 step below the guess
		#this should be changed to using mft.extinct before trying to use this function
		extinct_minus = 10 ** (-0.4 * extinction.fm07(model_wl, init_guess - step))
		extinct_init = 10 ** (-0.4 * extinction.fm07(model_wl, init_guess))
		extinct_plus = 10 ** (-0.4 * extinction.fm07(model_wl, init_guess + step)) 

		#extinct the test spectrum using each of the three possible extinction vectors
		model_minus = [model[n] - extinct_minus[n] for n in range(len(model))]
		model_init = [model[n] - extinct_init[n] for n in range(len(model))]
		model_plus = [model[n] - extinct_plus[n] for n in range(len(model))]

		#and calculate a variance for each of them
		minus_var = np.mean([m * 0.01 for m in model_minus])
		init_var = np.mean([m * 0.01 for m in model_init])
		plus_var = np.mean([m * 0.01 for m in model_plus])

		#calculate the chi square for each guess
		xs_minus = mft.chisq(data, model_minus, minus_var)
		xs_init = mft.chisq(data, model_init, init_var)
		xs_plus = mft.chisq(data, model_plus, plus_var)

		niter += 1
		#now if the current guess is better than each step, reduce the step size and start over
		if xs_init < xs_minus and xs_init < xs_plus:
			step *= 0.5
			niter = 0
		#if the step below is better, change the guess to that value 
		#frankly I have no idea what I was doing with these lines of code but they seem to have worked to basically decide whether to step up or down or just reduce the step size
		elif xs_init > xs_minus and xs_init < xs_plus:
			init_guess = init_guess - (init_guess * step)
		elif xs_init < xs_minus and xs_init > xs_plus:
			init_guess = init_guess + (init_guess * step)
		else:
			if xs_minus < xs_plus:
				init_guess = init_guess - (init_guess * step)
			else:
				init_guess = init_guess + (init_guess * step)

	#extinct the model by the best fit extinction guess I've found
	extinct = 10 ** (-0.4 * extinction.fm07(model_wl, init_guess))
	model = [model[n] - extinct[n] for n in range(len(model))]

	#return the test value and the best fit extinction value
	return model, init_guess

def find_norm(model_wl, model, data):
	'''DEPRECATED. Uses a basic simplex fit to find a best-fit normalization value. NOT UPDATED TO REFLECT CURRENT CODE REQUIREMENTS

	Args:
		model_wl (array): wavelength vector
		model (array): test spectrum
		data (array): input spectrum (the one I'm trying to find the best fit for)

	Note:
		Both spectra should have a common wavelength vector (e.g. both be interpolated to model_wl)

	Returns: 
		best normalization value (float) and best extinction value (float)

	'''
	#define an initial normalization guess
	init_guess = max(data)/max(model)

	#make sure the test wavelength is an array
	model_wl = np.array(model_wl)
	#define an initial step size that's 10% of the initial guess
	step = 0.1 * init_guess

	#as long as the step is larger than 0.1% of the original guess
	while step > 0.001 * max(data)/max(model):
		#print the normalization value we're trying
		print('norm: ', init_guess, step)
		#calculate the normalized test spectra - one at the guess, and two each one step away in opposite directions
		model_minus = [model[n] * (init_guess - (init_guess * step)) for n in range(len(model))]
		model_init = [model[n] * init_guess for n in range(len(model))]
		model_plus = [model[n] * (init_guess + (init_guess * step)) for n in range(len(model))]

		#now find the best fit extinction for each test normalization
		model_minus, extinct_minus = find_extinct(model_wl, model_minus, data)
		model_init, extinct_init = find_extinct(model_wl, model_init, data)
		model_plus, extinct_plus = find_extinct(model_wl, model_plus, data)

		#then calculate a variance that's just 1% of the spectrum value
		minus_var = np.mean([m * 0.01 for m in model_minus])
		init_var = np.mean([m * 0.01 for m in model_init])
		plus_var = np.mean([m * 0.01 for m in model_plus])
		
		#calculate the chi square
		xs_minus = mft.chisq(data, model_minus, minus_var)
		xs_init = mft.chisq(data, model_init, init_var)
		xs_plus = mft.chisq(data, model_plus, plus_var)

		#then decide what to do - stay where you are and just make the step size smaller to refine things, or accept a higher or lower guess and start over
		if xs_init < xs_minus and xs_init < xs_plus:
			step *= 0.5
		elif xs_init > xs_minus and xs_init < xs_plus:
			init_guess = init_guess - (init_guess * step)
		elif xs_init < xs_minus and xs_init > xs_plus:
			init_guess = init_guess + (init_guess * step)
		else:
			if xs_minus < xs_plus:
				init_guess = init_guess - (init_guess * step)
			else:
				init_guess = init_guess + (init_guess * step)

	#reutrn the best fit normalization and extinction
	return init_guess, extinct_init

def even_simpler(filename, t_guess, lg_range):
	'''***DEPRECATED*** 
	Finds the best fit temperature, log(g), normalization, and extinction for a spectrum given input temperature and log(g) guesses using nested 1D simplex fits.

	Args:
		filename (string): input spectrum file name
		t_guess (float): initial temperature guess
		lg_range (array): list of log(g) values to try

	Returns:
		Creates a file with each simplex fit step (best fit extinction, normalization, teff for each step, log(g), chi square, and step number)

	'''
	#read in the test spectrum
	wl, spec = np.genfromtxt(filename, unpack = True)

	#initialize some arrays to save results to as we go through
	xs = []
	temp = []
	logg = []
	norm = []
	st = []
	extinct = []

	#iterate through each log(g) values
	for l in lg_range:
		#set the initial test temperature as the guess temperature
		t_init = t_guess

		#and set the iniital step size to be 50K - this is something that is adjustable based on how large you want your initial steps to be
		step = 50
		#initialize a counter
		niter = 0

		#this cutoff is determined by how precise you want your temperature fit to be - here I've decided 10K is good enoough
		while step >= 10:
			#fist, print out the step size and the temperature I'm trying
			print(step, t_init)
			#then make the two test temperatures - one step above and below the test value
			t_minus = t_init - step
			t_plus = t_init + step

			#make all the test spectra
			ww_minus, ss_minus = mft.get_spec(t_minus, l, [0.45, 2.5], normalize = False)
			ww_init, ss_init = mft.get_spec(t_init, l, [0.45, 2.5], normalize = False)
			ww_plus, ss_plus = mft.get_spec(t_plus, l, [0.45, 2.5], normalize = False)

			#create an evenly spaced common wavelength vector 
			second_wl = np.linspace(max(wl[0], ww_init[0], ww_minus[0], ww_plus[0]), min(wl[-1], ww_init[-1], ww_minus[-1], ww_plus[-1]), len(ww_init) * 2)

			#interpolate everything to a common wavelength vector so that they're directly comparable
			di2 = interp1d(wle, spece)
			mi2_minus = interp1d(wwe_minus, sse_minus)
			mi2_init = interp1d(wwe_init, sse_init)
			mi2_plus = interp1d(wwe_plus, sse_plus)

			spece2 = di2(second_wl)
			sse_minus = mi2_minus(second_wl)
			sse_init = mi2_init(second_wl)
			sse_plus = mi2_plus(second_wl)

			#find the best fit normalization and extinction for this particular teff/log(g) combination, as well as the two test values
			n_minus, extinct_minus = find_norm(wwe_minus, sse_minus, spece2)
			n_init, extinct_init = find_norm(wwe_init, sse_init, spece2)
			n_plus, extinct_plus = find_norm(wwe_plus, sse_plus, spece2)

			#normalizae the spectrum - I think extincting the spectrum got left out here
			sse_minus = [s * n_minus for s in sse_minus]
			sse_init = [s * n_init for s in sse_init]
			sse_plus = [s * n_plus for s in sse_plus]

			#calculate the variance for the chi square by assuming it is 1% of the input spectrum
			var = [sp * 0.01 for sp in spece2]
			
			#calculate the chi square for each guess
			cs_minus = mft.chisq(sse_minus, spece2, var)
			cs_init = mft.chisq(sse_init, spece2, var)
			cs_plus = mft.chisq(sse_plus, spece2, var)

			#increase the step number by 1
			niter += 1

			#save everything to the array we'll eventually write to the output file
			st.append(step)
			xs.append(cs_init)
			temp.append(t_init)
			logg.append(l)
			norm.append(n_init)
			extinct.append(extinct_init)

			#decide whether I'm going to stay where I am and decrease the step size, or take a step up or down
			#this is JUST for the temperature, because I find the other best fit values for each test temperature
			if cs_init < cs_minus and cs_init < cs_plus:
				step *= 0.5
				niter = 0
			elif cs_init > cs_minus and cs_init < cs_plus:
				t_init = t_init - step
			elif cs_init < cs_minus and cs_init > cs_plus:
				t_init = t_init + step
			else:
				if cs_minus < cs_plus:
					t_init = t_init - step
				else:
					t_init = t_init + step

			#impose a cutoff in case it gets stuck - if it stays in the same spot for 10 steps without changing, decrease the step size and try again
			#this is a parameter that should be fiddled with depending on how quickly you want to wait for convergence
			if niter > 10:
				step *= 0.5

	#save everything to an output file
	#there is a nicer way to do this that doesn't require the loop to complete before writing everything out
	#but since this code isn't used anymore not going to bother modifying
	print('saving')
	np.savetxt(filename.split('/')[0] + '/results/params_' + filename.split('/')[-1], np.column_stack((st, xs, temp, logg, norm, extinct)), \
		header = '#step size, chi square, temperature, log(g), normalization, extinction')

	return

def find_best(pars, run, number):
	'''DEPRECATED Streamlined fitting function for use with a simplex or levenberg-marquardt canned function (from scipy or lmfit).

	Args:
		pars (list): list of test values for temperature, normalization, and extinction. log(g) is assumed to be fixed at a value of 4.
		run (string): directory for results to be written to
		number (float): system number (for keeping track of multiple systems in a population)

	Note:
		Requires a data spectrum to exist in {run}/specs/ and to be named "spec_{number}.txt".

	Returns:
		chi square array containing the values from the comparison between the data spectrum and the test spectrum.

	'''

	#retrieve the data spectrum 
	filename = run + '/specs/spec_{}.txt'.format(number)
	wl, spec = np.genfromtxt(filename, unpack = True)

	#unpack the parameters array to get the test temperature, normalization, and extinction values
	teff, normalization, extinct = pars[0], pars[1], pars[2]
	#fix the log(g) to 4 - you could also make this one of the parameters if you wanted to vary it 
	logg = 4
	#set a temperature limit mandated by the range of spectral types you're open to
	#in this case I'm not generating anything below 2700 K or above 7000 K so I set those limits to stop the fit from wandering outside those valid bounds
	if teff < 7000 and teff > 2500:
		#get the test spectrum 
		ww_init, ss_init = mft.get_spec(teff, logg, [min(wl), max(wl)], normalize = False)

		#create a new evenly spaced wavelength vector that only spans the common range of the test spectrum and the data spectrum
		second_wl = np.linspace(max(wle[0], wwe_init[0]), min(wle[-1], wwe_init[-1]), len(ww_init))

		#interpolate both spectra so they're on the same scale for direct comparison
		di2 = interp1d(wl, spec)
		mi2_init = interp1d(ww_init, ss_init)
		spece2 = di2(second_wl)
		sse_init = mi2_init(second_wl)

		#extinct the test spectrum
		sse_init = mft.extinct(second_wl, sse_init, extinct)
		#normalize the test spectrum
		sse_init = [s * normalization for s in sse_init]

		#use 1% of the data spectrum as the variance
		var = [sp * 0.01 for sp in spece2]

		#calculate the chi square value 
		cs_init = mft.chisq(sse_init, spece2, var)
		#write the result to a parameters file so I can keep track of what the code is doing
		f = open(run + '/results/parvals_{}.txt'.format(number), 'a')
		f.write('{} {} {} {} {}\n'.format(np.sum(cs_init)/len(cs_init), teff, logg, normalization, extinct))
		f.close()
	#if the temperature that was tried is outside the allowed bounds	
	else:
		#return an infinite valued chi square so that we reject this guess
		cs_init = np.full(len(wle), np.inf)
		#but save the tested parameters to the output file just to have a complete record
		f = open(run + '/results/parvals_{}.txt'.format(number), 'a')
		f.write('{} {} {} {} {}\n'.format(np.sum(cs_init)/len(cs_init), teff, logg, normalization, extinct))
		f.close()
	#as long as the chi square is valid, return it	
	try:
		return cs_init
	#if there's a problem for some reason, return an inf so the canned function doesn't accept this guess
	except:
		return np.full(len(wle), np.inf)

def simplex_fit(teff, logg, normalization, extinct, filename):
	'''Use a canned simplex or least squares or Levenberg-Marquardt fitting routine to find the best fit parameters to a spectrum.

	Args:
		teff (float): initial temperature guess
		logg (float): initial log(g) guess. Code rewrites this to a value of 4, but it's left here for completeness.
		normalization (float): normalization initial guess. The test spectrum is MULTIPLIED by this value. 
		extinct (float): initial extinction guess
		filename (string): full file name (including path, if in a subdirectory/other directory than where the code is being run) for the data spectrum

	Returns:
		File with best fit parameters

	'''
	#split the filename to get the run name
	run = filename.split('/')[0]
	#and split it in a different place to get the number of the system
	number = int(filename.split('_')[1].split('.')[0])

	#this whole section has a bunch of different techniques that I tried to fit with. They're all formatted to work as is, but I just moved away from this path
	#just uncomment whichever one you want

	#simp = np.array(([teff + 100, logg, normalization, extinct],
	#	[teff, logg, normalization + normalization * 0.05, extinct],
	#	[teff, logg, normalization, extinct + 0.5],
	#	[teff, logg - 0.2, normalization, extinct],
	#	[teff - 100, logg + 0.2, normalization, extinct - 0.5]))

	#a = minimize(find_best, np.array([teff, logg, normalization, extinct]), args = (run, number), method = 'Nelder-Mead',\
	#	 options = {'adaptive': True, 'initial_simplex': simp}, tol = 10)
	#a = leastsq(find_best, [teff, logg, normalization, extinct], args = (run, number))
	a = root(find_best, [teff, normalization, extinct], args = (run, number), method = 'lm', options = {'ftol':1e-1})
	#par = lmfit.Parameters()
	#par.add_many(('teff', teff, True, 2800, 6500),
	#			('logg', logg, True, 3.5, 5),
	#			('norm', normalization, True, 1e22, 1e26),
	#			('extinct', extinct, True, 0, 5))
	#
	#a = lmfit.minimize(find_best, par, args = (run, number), options = {'ftol': 10})
	print('fitting done')
	#save the results to a text file
	np.savetxt(run + '/results/{}_simplex.txt'.format(number), a.x)
	return

def fit_test(teff, logg, norm, extinct, filename, res = [3000], nsteps = 50, reg = [[0.6, 0.9]], degrade_res = True, perturb_init = True, cutoff = True):
	'''Uses an initial temperature, log(g), normalization, and extinction guess to find a best fit spectrum to an input data spectrum using a modified Gibbs sampler.

	Args:
		teff (float): initial test temperature
		logg (float): initial log(g) - currently always held constant at 4
		norm (float): initial normalization guess
		extinct (float): initial extinction guess
		filename (string): file containing data spectrum, with the filename containing the full path from the running directory to the file

	Note:
		The anticipated file structure is that the code is running in a top-level directory, which has a subdirectory named "runN" where N is some number. That directory
		should have "specs" and "results" subdirectories, the first of which contains system reference spectra and the second is where results will be written to.
		If running this using run_pop, that infrastructure should already have been set up.

	Returns:
		Reduced chi-square value for the best fit spectrum. Also writes out two files: One with all test values tried, the other with the best fit parameters for 
		each step, to a results subdirectory.

	'''
	#find the name of the run directory
	run = filename.split('/')[0]
	#and get the system number
	number = int(filename.split('_')[1].split('.')[0])
	#hardcode the log(g) values because we don't want to fit for them
	logg, logg_try = 4, 4

	#read in the file containing the data wavelength and spectrum
	wl, spec = np.genfromtxt(filename, unpack = True)

	#if we want to perturb the initial guesses (usually we do)
	#perturb the initial temperature and extinction guesses using a normally distribution with appropriate FWHM
	if perturb_init == True:
		teff += np.random.normal(scale = 100)
		extinct += np.random.normal(scale = 0.1)

		#if the perturbation makes the extinction negative, keep perturbing until it's positive
		while extinct < 0:
			extinct += np.random.normal(scale = 0.1)

	#get an initial test spectrum using the initial (perturbed, if selected) guess values
	ww_init, ss_init = mft.get_spec(teff, logg, [(min(wl)-5)/1e4, (max(wl) + 5)/1e4], normalize = False, resolution = max(res))

	norm_array, var_array = [], []
	wls, specs = list(np.arange(len(res))), list(np.arange(len(res)))
	cs_init = 0

	for n, r in enumerate(reg):

		iwl, ispec = wl[np.where((wl >= min(r) * 1e4) & (wl <= max(r) * 1e4))], spec[np.where((wl >= min(r) * 1e4) & (wl <= max(r) * 1e4))]

		iwl, ispec = mft.broaden(iwl, ispec, res[n])
		res_element = np.mean(iwl)/res[n]
		spec_spacing = iwl[1] - iwl[0]
		if 3 * spec_spacing < res_element:
			factor = (res_element/spec_spacing)/3
			iwl, ispec = mft.redres(iwl, ispec, factor)

		wi, si = mft.broaden(ww_init, ss_init, res[n])
		res_element = np.mean(wi)/res[n]
		spec_spacing = wi[1] - wi[0]
		if 3 * spec_spacing < res_element:
			factor = (res_element/spec_spacing)/3
			wi, si = mft.redres(wi, si, factor)

		#find the smallest wavelength space separation between steps, and make a wavelength array with half that spacing so we're oversampled
		min_diff = np.inf
		for k in range(1, len(ispec)):
			if iwl[k] - iwl[k-1] < min_diff:
				min_diff = iwl[k] - iwl[k-1]

		second_wl = np.arange(max(iwl[0], ww_init[0]) + 5, min(iwl[-1], ww_init[-1]) - 5, min_diff/2)
		wls[n] = second_wl

		#interpolate the test and data spectra to the common wavelength vector
		dd2 = interp1d(iwl, ispec)
		md2_init = interp1d(wi, si)
		spec2 = dd2(second_wl)
		ssi = md2_init(second_wl)

		specs[n] = spec2

		#extinct the test spectrum by the initial guess amount
		ssi = mft.extinct(second_wl, ssi, extinct)

		if perturb_init == True:
			#calculate an initial normalization guess by making it the ratio of the average values of the two spectra
			normalization = np.mean(spec2)/np.mean(ssi)

			#then perturb it with a random value drawn from a normal distribution with FWHM of 2% of the initial normalization guess
			normalization += np.random.normal(scale = np.abs(normalization * 0.02))
		else:
			normalization = norm[n]

		norm_array.append(normalization)
		#normalize the guess spectrum

		ssi = [s * normalization for s in ssi]

		#make a variance array by first taking a baseline variance that's 1% of the average value of the data spectrum
		ref_variance = 0.01 * spec2

		#then calculate the variance of each pixel as the reference variance times the square root of the ratio of the data pixel value to the mean data pixel value
		#the goal here is to give high values....larger error than low values? I think this got flipped...
		#***SHOULD THE FRACTION INSIDE THE SQUARE ROOT BE INVERTED?****
		vary = [ref_variance[n] * np.sqrt(np.median(spec2)/spec2[n]) for n in range(len(spec2))]
		var_array.append(vary)

		#calculate the chi square of the initial test and the input data spectrum
		ci = mft.chisq(ssi, spec2, vary)
		cs_init += np.sum(ci)/len(ci)

		# # plot the two spectra where the test spectrum has been extincted and normalized
		# plt.figure()
		# plt.plot(ssi, label = 'initial test')
		# plt.plot(spec2, label = 'data')
		# plt.legend()
		# plt.savefig(run + '/results/initial_test.png')
		# plt.close()


	#initialize a couple counters
	niter = 0
	total_iter = 0
	csqs = [cs_init]
	vt, vnorm, vext = [teff], [norm_array], [extinct]

	#write the initial guesses, the step number, and the reduced chi square to an output file
	f = open(run + '/results/parvals_{}.txt'.format(number), 'a')
	f.write('{} {} {} {} {} {}\n'.format(niter, cs_init, teff, logg, norm_array[0], extinct))
	f.close()

	#modified gibbs sampler fitting
	#change niter based on how long you want the fitting to try new steps before terminating
	while niter < nsteps * 2 and total_iter < nsteps * 100:
		#inititalize a parameters array to make varying the parameters a little easier
		#it just contains the current best guess temperature, normalization, and extinction
		parvar = np.asarray([teff, np.array(norm_array), extinct])

		#if we're in the early stage of the ift, we want to take large steps to try to fully explore the parameter space
		if niter < nsteps:
			#define a variance array that's three different draws from normal distributions with different FWHM values: 100 K, 1%, and 0.05 mag
			var = np.asarray([np.random.normal(scale = 100), np.array([np.random.normal(scale=np.abs(n*0.01)) for n in norm_array]), np.random.normal(scale = 0.05)])
			#if the extinction variance would make the total extinction negative, resample until it's positive
			while parvar[2] + var[2] < 0:
				var[2] = np.random.normal(scale = 0.05)

		#if we're further along in the fitting process, make the steps smaller - but the idea is exactly the same
		else:
			var = np.asarray([np.random.normal(scale = 5), np.array([np.random.normal(scale=np.abs(n*0.005)) for n in norm_array]), np.random.normal(scale = 0.01)])
			while parvar[2] + var[2] < 0:
				var[2] = np.random.normal(scale = 0.01)

		#add the variation values to the guesses to make the new guesses
		if len(csqs) >= 2:
			if csqs[-1] < csqs[-2]:
				var = np.asarray([vt[-1], vnorm[-1], vext[-1]])
			else:
				var = var
		else:
			var = var

		while parvar[2] + var[2] < 0:
			var[2] += np.random.normal(scale = 0.02)
		parvar += var

		vt.append(var[0]), vnorm.append(var[1]), vext.append(var[2])

		#then unpack that array into the new trial values 
		teff_try, norm_arr_try, extinct_try = parvar
		cs_try = 0

		#as long as we're inside a valid temperature range (so things can't go completely off the rails)
		if teff_try < 10000 and teff_try > 2300:

			#make a new test spectrum with the new temperature
			ww_i, ss_i = mft.get_spec(teff_try, logg, [(min(wl) - 5)/1e4, (max(wl) + 5)/1e4], normalize = False, resolution = max(res))

			for n, r in enumerate(reg):
				iwl, ispec = ww_i[np.where((ww_i > min(r) * 1e4 - 5) & (ww_i < max(r) * 1e4 + 5))], ss_i[np.where((ww_i > min(r) * 1e4 - 5) & (ww_i < max(r) * 1e4 + 5))]
				iwl, ispec = mft.broaden(iwl, ispec, res[n])
				res_element = np.mean(iwl)/res[n]
				spec_spacing = iwl[1] - iwl[0]
				if 3 * spec_spacing < res_element:
					factor = (res_element/spec_spacing)/3
					iwl, ispec = mft.redres(iwl, ispec, factor)

				mi2_init = interp1d(iwl, ispec)
				sse_i = mi2_init(wls[n])
				
				#extinct and normalize the test spectrum using the test values
				sse_i = mft.extinct(wls[n], sse_i, extinct_try)
				sse_i = [s * norm_arr_try[n] for s in sse_i]

				#calculate the chi square
				ct = mft.chisq(sse_i, specs[n], var_array[n])
				cs_try += np.sum(ct)/len(ct)

				# # plot the test and data and save the figure
				# fig, ax2 = plt.subplots(nrows = 1)
				# ax2.plot(wls[n], sse_i, label = 'test, chisq = {:.0f}'.format(np.sum(ct)/len(ct)))
				# ax2.plot(wls[n], specs[n], label = 'spectrum')
				# ax2.legend(loc = 'best')
				# plt.savefig(run + '/results/test{}_{}_{}.png'.format(number, total_iter, n))
				# plt.close()

			#if the reduced chi square of the guess is better than the previous best value
			if cs_try < cs_init:
				#accept the new parameters
				teff, logg, norm_array, extinct = teff_try, logg_try, norm_arr_try, extinct_try
				#set the new reference chi square to be the one we just calculated
				cs_init = cs_try
				#then if we're in the small jump regime, go to the beginning of that again
				if niter >= nsteps + 1:
					niter = nsteps + 1
				#otherwise go back to the beginning
				else:
					niter = 0
			#if the test fit isn't better, add 1 to the counter 
			else:
				niter += 1

			total_iter += 1
			#if we're at the maximum number of tests without change and the chi square is still lousy
			#send it back to close to the beginning to force more steps
			if cutoff == True and niter == nsteps * 2 and cs_init > 1.5:
				niter = int(nsteps/2) - 1

			#save the current best fit values to an output file
			f = open(run + '/results/parvals_{}.txt'.format(number), 'a')
			f.write('{} {} {} {} {} {}\n'.format(niter, cs_init, teff, logg, norm_array[0], extinct))
			f.close()

			#and save the attempted guess to another output file, even if it isn't better than the current guess
			f = open(run + '/results/testpars_{}.txt'.format(number), 'a')
			f.write('{} {} {} {} {} {}\n'.format(1, cs_try, teff_try, logg_try, norm_arr_try[0], extinct_try))
			f.close()

			csqs.append(cs_init)
		#if the guess temperature is outside the accepted regime, vary everything until it's accepted
		else:
			teff_try = teff_try + np.random.normal(scale = 200)
	#return the reduced chi square value for the best fit test spectrum at the end of fitting
	return cs_try
	
def fit_lum(mag, ts):
	'''Use a set of absolute VRIJHK magnitudes (e.g. the apparent magnitudes output by system creation, corrected for distance and extinction) to infer a system
	luminosity by interpolating evolutionary models.

	Args:
		mag (array): 6-element array containing VRIJHK absolute magnitudes
		ts (int): measured temperature from spectral fitting or by some other means

	Note:
		Requires the parsec isochrone file in the same directory as the code.

	Returns:
		log10(luminosity) in solar luminosities.

	'''
	#read in the evolutionary models
	matrix = np.genfromtxt('parsec_isos.dat', autostrip = True)

	#read in the age column and convert it into megayears
	age = matrix[:, 1]
	age = [(10**a)/1e6 for a in age]
	#remove the redundant entries
	a1 = [age[0]]
	for n in range(len(age)):
		if age[n] != a1[-1]:
			a1.append(age[n])

	#read in the luminosity and log T columns, and convert log T to linear units
	lum, teff = matrix[:, 5], matrix[:, 6]
	teff = np.asarray([10**(t) for t in teff])
	#read in the magnitudes
	mags = matrix[:, -6:] #[V R I J H K]

	#initialize a list for putting potential luminosities into
	lum_cand = []
	mag_cand = np.empty(6)

	#convert the input magnitudes into fluxes so we're not doing anything in log space
	mag = [float(10** (-0.4 * (m))) for m in mag]
	# mag = [float(m) for m in mag]
	# print(mag)
	#go through each age except for time == 0
	for k, a in enumerate(a1[1:]):
		#find the range of temps, mags, and luminosities for the given age
		t = teff[np.where(age == a)]
		m = mags[np.where(age == a), :][0]
		#and convert log(L) into L
		l = 10 ** (lum[np.where(age == a)])
		# print(min(t), max(t), ts, a)

		# print(np.shape(m), m)

		#for each filter and each set of mags, convert into flux units
		for n, row in enumerate(m):
			m[n] = [10 ** (-0.4 * r) for r in row]

		#initialize an array to recieve the interpolated luminosity
		interpolated_mag = []
		#go through each filter and get the interpolated magnitude for the given temperature
		for n in range(len(mag)):
			try:
				mag_interpolate = interp1d(t, m[:, n])
				interpolated_mag.append(float(mag_interpolate(ts)))
			except:
				interpolated_mag.append(-99)

		#get the luminosity for the given teff
		lum_interpolate = interp1d(t, l)

		#save the luminosity and magnitude combination for the given age
		try:
			lum_cand.append(float(lum_interpolate(ts)))
		except:
			lum_cand.append(-99)
		if k == 0:
			mag_cand = interpolated_mag
		else:
			mag_cand = np.vstack((mag_cand, interpolated_mag))


	# #Now I have an array of the predicted (mags, luminosity) for a given teff (the input teff) for each age
	# #now i need to interpolate mags as a fn of luminosity, and plug in the given mags to get the right luminosity
	l_try = []
	for n,m in enumerate(mag):
		if m > -99:
			i = interp1d(mag_cand[:, n], lum_cand, fill_value = 'extrapolate')
			l_try.append(i(m))

	#return the log of the median of the candidates
	return np.log10(np.median(l_try))

def analyze_sys(runfolder, model = 'parsec'):
	'''
	Args: 
		runfolder (string): path to follow
		model (string): which models to use. Default (and only supported value at the moment) is 'parsec'
		
	Note:
		Requires files contining chi square values and values from each step for each system in a subfolder of "runfolder/results/parvals[n].txt" where n is the system number.

	Returns:
		Mass and Luminosity derived from teff and log(g).

	'''
	#collect all the files with the fit results
	csqs = glob(runfolder + '/results/parvals*.txt')#glob(runfolder + '/results/params*.txt')

	#read in the initial primary system parameters
	#the magnitudes are APPARENT magnitudes: they have been modified by a distance modulus and the assigned system extinction value
	pnum, pmass, multiplicity, page, av, ptemp, plogg, pluminosity, distance, pvmag, prmag, pimag, pjmag, phmag, pkmag = np.genfromtxt(runfolder + '/ppar.txt', unpack = True)

	#and sort them so they're in order, because ppar won't write them out in order
	pnum, pmass, multiplicity, page, av, ptemp, plogg, pluminosity, distance, pvmag, prmag, pimag, pjmag, phmag, pkmag = pnum[np.argsort(pnum)], pmass[np.argsort(pnum)],\
		 multiplicity[np.argsort(pnum)], page[np.argsort(pnum)], av[np.argsort(pnum)], ptemp[np.argsort(pnum)], plogg[np.argsort(pnum)], pluminosity[np.argsort(pnum)], \
		 distance[np.argsort(pnum)], pvmag[np.argsort(pnum)], prmag[np.argsort(pnum)], pimag[np.argsort(pnum)], pjmag[np.argsort(pnum)], phmag[np.argsort(pnum)], pkmag[np.argsort(pnum)]

	try:
		#read in all the secondary parameters
		snum, s_mass, sep, sage, seccentricity, period, stemp, slogg, sluminosity, svmag, srmag, simag, sjmag, shmag, skmag = np.genfromtxt(runfolder + '/spar.txt', unpack = True)

		#and sort those too
		snum, s_mass, sep, sage, seccentricity, period, stemp, slogg, sluminosity, svmag, srmag, simag, sjmag, shmag, skmag = snum[np.argsort(snum)], s_mass[np.argsort(snum)], \
			sep[np.argsort(snum)], sage[np.argsort(snum)], seccentricity[np.argsort(snum)], period[np.argsort(snum)], stemp[np.argsort(snum)], slogg[np.argsort(snum)],\
		 	sluminosity[np.argsort(snum)], svmag[np.argsort(snum)], srmag[np.argsort(snum)], simag[np.argsort(snum)], sjmag[np.argsort(snum)], shmag[np.argsort(snum)], skmag[np.argsort(snum)]

	except:
		snum = []
		sage = []
		pass
	#Not sure why this luminosity was being converted to log(luminosity) out of what was assumed to be magnitudes
	#it's definitely not in magnitudes 
	#but going to keep this in here for posterity in case i've missed something
	# sluminosity = np.asarray([np.log10(10**(-0.4 * s)) for s in sluminosity])
	#make the magnitudes into one nice array
	pmags = np.column_stack((pvmag, prmag, pimag, pjmag, phmag, pkmag))
	try:
		smags = np.column_stack((svmag, srmag, simag, sjmag, shmag, skmag))
	except:
		pass

	#initialize arrays for system luminosity, component temperature difference, and total system magnitudes 
	sl = np.zeros(len(pluminosity))
	tdiff = np.zeros(len(pluminosity))
	sys_mag = np.zeros(np.shape(pmags))

	#for each system
	for n in range(len(pluminosity)):
		#if it's a binary
		if n in snum:
			#calculate the total system luminosity by addint together in linear units then converting back to log10 afterward
			sl[n] = np.log10(10**pluminosity[np.where(pnum == n)] + 10**sluminosity[np.where(snum == pnum[n])])
			# print(sl[n])
			# print(ptemp[n] - stemp[np.where(snum == pnum[n])], ptemp[n], stemp[np.where(snum == pnum[n])], np.where(snum == pnum[n]), pnum[n], snum[np.where(snum == pnum[n])])
			#enter the component temperature difference into the tdiff array
			tdiff[n] = ptemp[n] - stemp[np.where(snum == pnum[n])]
			#calculate the magnitude of the system as a whole by converting mags to fluxes, adding them, then going back to mags
			for k in range(len(pmags[0, :])):
				sys_mag[n][k] = -2.5 * np.log10((10 ** (-0.4 * pmags[n][k])) + (10 ** (-0.4 * smags[np.where(snum == pnum[n])[0][0]][k])))
		#if it's not a binary just add the primary components and set the temp difference to -1 so we can weed it out later
		else:
			sl[n] = pluminosity[n]
			tdiff[n] = -1
			for k in range(len(pmags[0, :])):
				sys_mag[n][k] = pmags[n][k]

	cmap = cm.plasma(tdiff[np.where(tdiff > 0)])

	#Initialize a whole bunch of arrays for keeping track of different values
	#this first bunch is for output values
	masses = np.zeros(len(csqs))
	lums = np.zeros(len(csqs))
	ages = np.zeros(len(csqs))

	#keeps track of system number
	num = np.zeros(len(csqs))

	#keeps track of input age, temperature, log(g), extinction, and normalization
	inp_age = np.zeros(len(csqs))
	inp_t = np.zeros(len(csqs))
	inp_l = np.zeros(len(csqs))
	inp_e = np.zeros(len(csqs))
	inp_n = np.zeros(len(csqs))

	#keeps track of output temperature, log(g), extinction, and normalization
	out_t = np.zeros(len(csqs))
	out_l = np.zeros(len(csqs))
	out_e = np.zeros(len(csqs))
	out_n = np.zeros(len(csqs))	

	#records the temperature difference between the two components 
	#not sure why I have a second one of these when i already have tdiff but i'll keep it in so I don't accidentally mess something up
	#i assume it has to do with counting consistency
	td = np.zeros(len(csqs))

	#read in the evolutionary models
	matrix = np.genfromtxt('parsec_isos.dat', autostrip = True)
	#get the age and convert it into megayears from log(years)
	aage = matrix[:, 1]
	aage = [(10**a)/1e6 for a in aage]
	#get the mass, log(luminosity), effective temp, and log(g)
	ma = matrix[:,4]
	lll = matrix[:, 5]
	teff = matrix[:, 6]
	llog = matrix[:, 7]


	#remove redundant ages from the age vector
	a1 = [aage[0]]
	for n in range(len(aage)):
		if aage[n] != a1[-1]:
			a1.append(aage[n])

	#make three lists where the first is the first value in the non-redundant age array, and the second and third lists are the luminosity
	#and temperature values for that single age
	#this is ONLLY for the first age, and we're doing it so we can plot isochrones
	aa1, ll1, tt1 = [np.full(len(np.where(aage == a1[0])[0]), a1[0])], [lll[np.where(aage == a1[0])]], [teff[np.where(aage == a1[0])]]
	#switch from log(temp) to linear temp
	tt1 = [10 ** t for t in tt1]
	#plot the age = 0 temperature vs. luminosity 
	fig, ax = plt.subplots()
	ax.plot(tt1[0], ll1[0], label = '0')

	#now for all the otehr ages fill an array with the single valued age, get the temperature and convert it from log
	#then plot it versus the correct luminosity
	#tagging each one with the age and color coding it 
	for n in range(1, len(a1)):
		a2 = np.full(len(np.where(aage == a1[n])[0]), a1[n])
		tt2 = teff[np.where(aage == a1[n])]
		tt2 = [10 ** t for t in tt2]

		ax.plot(tt2, lll[np.where(aage == a1[n])], label = '{}'.format(int(np.around(a1[n]))), color = cm.plasma(a1[n]/10), zorder = 0)

	#make the isochrone labels show up between the 6000 and 7000 K temperatures, except for the zero age one which shows at 3500 K
	#labelLines does exactly that 
	#the zorder of 2.5 here makes the labels plot above the lines - fontsize and alpha can be tuned as desired
	xvals = list(np.linspace(7000,8500, len(a1[1:])))
	xvals.insert(0, 3500)
	labelLines(plt.gca().get_lines(), align = False, xvals = xvals, fontsize = 9, alpha = 0.8, zorder = 2.5)

	#this code plots isomass lines on top of the isochrones 
	# m1 = [ma[0]]
	# for n in ma:
	# 	if n not in m1:
	# 		m1.append(n)

	# mm1, ll1, tt1 = [np.full(len(np.where(ma == m1[0])[0]), m1[0])], [lll[np.where(ma == m1[0])]], [teff[np.where(ma == m1[0])]]
	# tt1 = [10 ** t for t in tt1]
	# ax.plot(tt1[0], ll1[0])

	# for n in arange(1, len(m1),2):
	# 	m2 = np.full(len(np.where(ma == m1[n])[0]), m1[n])
	# 	tt2 = teff[np.where(ma == m1[n])]
	# 	tt2 = [10 ** t for t in tt2]

	# 	ax.plot(tt2, lll[np.where(ma == m1[n])], color = cm.plasma(m1[n]/6))


	a = np.array(page)
	b = np.array(sage)

	#put the input *primary* teff and luminosity onto the isochrone plot 
	inp = ax.scatter(ptemp, pluminosity, s = 25, color = 'xkcd:orange', label = 'Input')
	#this line is if you want to show the secondary input population also
	# ax.scatter(stemp, sluminosity, s = 25, color = 'xkcd:orange')

	#initialize an extra extinction recording list
	avv = []

	td = np.zeros(len(csqs))
	#now go through each system - recall that csqs is the glob-queried list of all files containing a fitting result
	for k, file in enumerate(csqs):
		#get the system number of the file assuming the filename structure is parvals_number.txt
		number = int(file.split('.')[0].split('_')[1])
		#and record it in the number recording array above so I can check what order files are being read in, as needed
		num[k] = number
		#read in the 6 columns from the fitting results file
		numb, cs, temp, lg, norm, ext = np.genfromtxt(file, unpack = True, autostrip = True)
		#if for some reason a file exists where it timed out before hitting the chi square < 1.5 benchmark, show the final chi square value, along with the system number
		# if cs[-1] > 1.5:
		# 	print(k, cs[-1])
		
		#find the best fit temperature, log(g), extinction, and normalization
		ts =temp[-1]#[np.where(cs == min(cs))][0]
		l = lg[np.where(cs == min(cs))][0]
		extinct = ext[-1]#[np.where(cs == min(cs))][0]
		norm = norm[-1]#[np.where(cs == min(cs))][0]

		#find the initial assigned temp, log(g), Av, and age for the system and put them in the input arrays we defined before
		inp_t[k] = ptemp[np.where(pnum == number)[0]]
		inp_l[k] = plogg[np.where(pnum == number)[0]]
		inp_e[k] = av[np.where(pnum == number)[0]]
		inp_age[k] = page[np.where(pnum == number)[0]]

		#and put the best fit values into the output arrays 
		out_t[k] = ts
		out_l[k] = l
		out_e[k] = extinct

		if cs[-1] > 1:
			print(number, cs[-1])

		if number in snum:
			td[k] = inp_t[k] - stemp[np.where(snum == number)[0]]
		else:
			td[k] = -1

		#get the cardelli extinction factors for VRIJHK again
		factor = [1, 0.751, 0.479, 0.282, 0.190, 0.114] #cardelli et al. 1989
		#find the correct set of system mags (which we calculated earlier using the apparent magnitudes from the initial system creation)
		m = sys_mag[number,:]
		#and correct back to absolute mag using the measured extinction and (for now) the correct assigned distance - this will be changed eventually
		mags = [m[n] - (extinct * factor[n]) + 5 - (5 * np.log10(distance[np.where(pnum == number)[0]]/10)) for n in range(len(factor))]
		
		#use the magnitudes to find the best-fit luminosity
		luminosity = fit_lum(mags, ts) 
		#save the extinction again to that extra array we defined before the loop started
		avv.append(extinct)

		#this allows for other models to potentially be used in the future
		if model == 'parsec':

			#use griddata to bilinearly interpolate the temperature and luminosity arrays to get an age and a mass from the system measured temperature and luminosity
			a = griddata((teff, lll), aage, (np.log10(ts), luminosity))
			m = griddata((teff, lll), ma, (np.log10(ts), luminosity))

			#record the measured age, mass, and luminosity of the system
			masses[k] = m
			lums[k] = luminosity
			ages[k] = a

	#now finish plotting the isochrone
	#first, identify the multiples in the order that the temperature array is in
	multiples = np.array([multiplicity[np.where(pnum == num[k])][0] for k in range(len(num))])

	#then sort out hte binaries vs. singles
	out_t_bin, out_lum_bin = out_t[np.where(multiples > 0)], lums[np.where(multiples > 0)]
	out_t_single, out_lum_single = out_t[np.where(multiples == 0)], lums[np.where(multiples == 0)]

	#plot the single stars and label them as such
	out = ax.scatter(out_t_single, out_lum_single, s = 20, color = 'xkcd:blue purple', label = 'Single star output')
	#if there are binaries
	if len(out_t_bin) > 0:
		#plot those and add a colorbar showing the temperature differences between primaries and secondaries
		a = ax.scatter(out_t_bin, out_lum_bin, label = 'Binary star output', marker = 'x', c = td[np.where(td > 0)], cmap = plt.cm.plasma)
		cbar = plt.colorbar(a)
		cbar.set_label(r'T$_{eff}$ difference (K)')

	#set the fitting parameters, labels, and limits
	ax.set_xlim(8500, 2500)
	ax.set_ylim(-2, 2)
	ax.set_xlabel(r'T$_{eff}$', fontsize = 13)
	ax.set_ylabel(r'log(L)', fontsize = 13)
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	try:
		ax.legend(fontsize = 13, loc = 'best', handles = [inp, out, a])
	except:
		ax.legend(fontsize = 13, loc = 'best', handles = [inp, out])
	fig.tight_layout()
	#save the figure in the run directory
	plt.savefig(os.getcwd() + '/' + runfolder + '/isochrone.pdf')
	plt.close()

	#select the input and output temps where the temperature difference is > 0 - this corresponds to binaries
	it, ot = inp_t[np.where(td >= 0)], out_t[np.where(td >= 0)]
	#and find the in and output temps where temp difference is < 0 - this corresponds to singles
	it_, ot_ = inp_t[np.where(td < 0)], out_t[np.where(td < 0)]

	#we see different fitting behavior above and below 4300 K, so we also want to sort those to do some stats
	it1, it2, ot1, ot2 = it_[np.where(it_ < 4300)], it_[np.where(it_ >= 4300)], ot_[np.where(it_ < 4300)], ot_[np.where(it_ >= 4300)]

	#input T vs T residual plot
	fig, ax = plt.subplots()#figsize = (8, 5))
	#if there are binaries it (the input binaries list) will be longer than 2
	if len(it) > 2:
		#in that case plot the binaries using an x marker and use a colorbar to show the temperature difference between the two components
		a = ax.scatter(it, it-ot, marker = 'x', s = 25, label = 'Binary stars', c = td[np.where(td >= 0)], cmap = plt.cm.plasma)
		cbar = plt.colorbar(a)
		cbar.set_label(r'T$_{eff}$ difference (K)')
	#if there is nonzero extinction
	if inp_e[0] > 0:
		#plot the single stars as points and use a colorbar to show the different extinction values
		b = ax.scatter(it_, it_ - ot_, marker = '.', s = 25, label = 'Single stars', c = np.array(inp_e)[np.where(td <= 0)])
		cbar = plt.colorbar(b)
		cbar.set_label(r'Extinction (A$_{V}$)')
	else:
		#if there's no extinction just plot the singles as dots
		ax.scatter(it_, it_ - ot_, marker = '.', s = 25, label = 'Single stars', color = 'navy')
	#plot the zero error line and the average single star error line
	ax.plot([min(min(inp_t), min(out_t)), max(max(inp_t), max(out_t))], [0, 0], linestyle = ':', label = 'Zero error')
	ax.axhline(np.mean(it_ - ot_), color = 'orange', zorder= 0, label = r'Single Star Error:\\{:.0f} $\pm$ {:.0f} K'.format(np.mean(it_-ot_), np.std(it_-ot_)))#(T $<$ 4300)\\{:.0f} $\pm$ {:.0f} K (T $\geq$ 4300)'\
		#.format(np.mean(it1 -ot1), np.std(it1 - ot1), np.mean(it2 -ot2), np.std(it2- ot2)))
	#format and label everything
	plt.minorticks_on()
	ax.set_xlim(2200, max(max(ot_), max(it_)) + 50)
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input Temp (K)', fontsize = 13)
	ax.set_ylabel('Input Temp - Output Temp (K)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/temp_plot.pdf')
	plt.close()	


	#make input and output extinction values for binaries (1) and singles (2)
	ie1, ie2, oe1, oe2 = inp_e[np.where(multiplicity == 1)], inp_e[np.where(multiplicity == 0)], out_e[np.where(multiplicity == 1)], out_e[np.where(multiplicity == 0)]

	#plot the extinction residual for the single stars
	fig, ax = plt.subplots()
	a = ax.scatter(ie2, ie2-oe2, marker = '.', s = 20, label = 'Single stars', c = inp_t[np.where(td < 0)], cmap = plt.cm.plasma)
	cbar = plt.colorbar(a)
	#if there are binaries, plot them with a different marker and color
	if len(ie1) >2:
		ax.scatter(ie1, ie1 - oe1, marker = 'x', s = 20, label = 'Binary stars', color = 'xkcd:sky blue')
	#plot the zero error and average error lines
	ax.plot([min(min(inp_e), min(out_e)), max(max(inp_e), max(out_e))], [0, 0], linestyle = ':', label = 'Zero error')
	ax.axhline(np.mean(ie2 - oe2), label = r'Average Error: {:.2f} $\pm$ {:.2f}'.format(np.mean(oe2 -ie2), np.std(oe2 -ie2)))
	#format and label everything
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input Extinction (mags)', fontsize = 13)
	ax.set_ylabel('Input Extinction - Output Extinction (mags)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/av_plot.pdf')
	plt.close()	

	#plot input and output temperature as two histograms 
	fig, ax = plt.subplots()
	ax.hist(inp_t, color = 'navy', label = 'Input T', bins = np.arange(min(min(inp_t), min(out_t)), max(max(inp_t), max(out_t)), 250))
	ax.hist(out_t, color='xkcd:sky blue', alpha = 0.7, label = 'Output T', bins = np.arange(min(min(inp_t), min(out_t)), max(max(inp_t), max(out_t)), 250))
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	#ax.set_yscale('symlog')
	ax.set_xlabel('Temp (K)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/temp_hist.pdf')
	plt.close()	

	#plot input and output spectral type histograms using the spt to teff breakdown
	#from Herczeg and Hillenbrand 2014 (Table 5)
	divs = [2570, 2670, 2770, 2860, 2980, 3190, 3410, 3560, 3720, 3900, 4020, 4210, 4710, 4870, 5180, 5430, 5690, 5930, 6130, 6600]
	labels = ['M9', 'M8', 'M7', 'M6', 'M5', 'M4', 'M3', 'M2', 'M1', 'M0', 'K7', 'K5', 'K2', 'K0', 'G8', 'G5', 'G2', 'G0', 'F8', 'F5']
	fig, ax = plt.subplots()
	ax.hist(inp_t, bins = divs, label = "Input SpT", color = 'navy')
	ax.hist(out_t, bins = divs, label = "Fitted SpT", color = 'xkcd:sky blue', alpha = 0.7)
	ax.set_xticks(divs[::2])
	ax.set_xticklabels(labels[::2], fontsize=14)
	ax.set_yscale('log')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Stellar Spectral Type', fontsize = 13)
	ax.set_ylabel('Number', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/spt_hist.pdf')
	plt.close()

	#if the age of a system is undefined set it to a negative
	for n, a in enumerate(ages): 
		if np.isnan(a):
			ages[n] = -99

	#make input and output age arrays for binaries and singles
	iab, oab = inp_age[np.where(td >= 0)], ages[np.where(td >= 0)]
	ias, oas = inp_age[np.where(td < 0)], ages[np.where(td < 0)]

	#then remove anything that was undefined (that has now gotten a value of -99)
	iab, oab, ias, oas = iab[np.where(oab > -99)], oab[np.where(oab > -99)], ias[np.where(oas > -99)], oas[np.where(oas > -99)]

	fig, ax = plt.subplots()
	#plot the single star input age vs. age residual
	ax.scatter(ias, ias - oas, marker = '.', s = 20, color = 'navy', label = 'Primary stars')
	#if there are binaries plot those too with a different marker
	if len(iab) > 2:
		a = ax.scatter(iab, iab-oab, marker = 'x', s = 25, label = 'Binary stars')
		#including a line showing the average binary age error
		ax.axhline(np.mean(iab-oab), color = 'xkcd:red orange', label = r'Average binary star error: {:.2f} $\pm$ {:.2f} Myr'.format(np.mean(iab-oab), np.std(iab-oab)))
	#plot the zero error and average error lines
	ax.plot([min(min(inp_age), min(ages)), max(max(inp_age), max(ages))], [0, 0], linestyle=':', label = 'Zero error')
	ax.axhline(np.mean(ias - oas), label = r'Average single star error: {:.2f} $\pm$ {:.2f} Myr'.format(np.mean(ias - oas), np.std(ias - oas)), color = 'orange')
	#format and label everything
	plt.minorticks_on()
	ax.set_xlim(min(inp_age) - 0.25, max(inp_age) + 0.25)
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input age (Myr)', fontsize = 13)
	ax.set_ylabel('Input age - Output age (Myr)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/age_diff_plot.pdf')
	plt.close()

	#plot input age vs output age, along with the 1:1 line that would show perfect agreement
	fig, ax = plt.subplots()
	ax.scatter(inp_age, ages, marker = '.', s = 20, color = 'navy', label = 'Primary stars')
	ax.plot([min(min(inp_age), min(ages)), max(max(inp_age), max(ages))], [min(min(inp_age), min(ages)), max(max(inp_age), max(ages))], linestyle=':', label = '1:1')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input age (Myr)', fontsize = 13)
	ax.set_ylabel('Output age (Myr)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/age_plot2.pdf')
	plt.close()

	#sort out any nans 
	agesn = ages[~np.isnan(ages)]

	#cutoffs decided from Pecaut and Mamajek 2013
	fage = agesn[np.where((out_t > 6000) & (out_t < 7000))]
	gage = agesn[np.where((out_t > 5300) & (out_t < 6000))]
	kage = agesn[np.where((out_t < 5300) & (out_t > 3900))]
	ma = agesn[np.where(out_t < 3900)]
	mage = ma #np.tile(ma, 24)
	# agesn = np.concatenate((agesn, np.tile(ma, 23)), axis = None)

	fig, ax = plt.subplots()
	#then make histograms of the input and fitted temperatures using 0.25 myr bins
	try:
		ax.hist(inp_age, color = 'navy', label = 'Input ages', bins = np.arange(min(min(inp_age), min(agesn)), max(max(agesn), max(inp_age)), 0.25))
		ax.hist(agesn, color = 'xkcd:sky blue', alpha = 0.6, label = \
			r'Output ages\\Avg. M age: {:.1f} $\pm$ {:.1f} (N = {})\\Avg. F age: {:.1f} $\pm$ {:.1f} (N = {})\\Avg. age: {:.1f} $\pm$ {:.1f} (N = {})'\
			.format(np.mean(mage), np.std(mage), len(mage), np.mean(fage), np.std(fage), len(fage), np.mean(agesn), np.std(agesn), len(agesn)),\
			 bins = np.arange(min(min(inp_age), min(agesn)), max(max(agesn), max(inp_age)), 0.25))
	except:
		ax.hist(inp_age, color = 'navy', label = 'Input ages', bins = np.arange(inp_age, max(agesn), 0.25))
		ax.hist(agesn, color = 'xkcd:sky blue', alpha = 0.6, label = 'Output ages', bins = np.arange(inp_age, max(agesn), 0.25))
	#format and label everything
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.set_xlim(min(min(agesn), min(inp_age))-0.25, max(max(inp_age), max(agesn))+0.25)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input and Output Age (Myr)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/age_hist.pdf')
	plt.close()

	fig, ax = plt.subplots()
	ax.errorbar(['F', 'G', 'K', 'M'], [np.mean(fage), np.mean(gage), np.mean(kage), np.mean(mage)], yerr = [np.std(fage), np.std(gage), np.std(kage), np.std(mage)])
	ax.set_xlabel('Spectral type', fontsize = 13)
	ax.set_ylabel('Measured age (Myr)', fontsize = 13)
	plt.minorticks_on()
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/age_SpT.pdf')
	plt.close()


	#save the results of the interpolations: age, masses, luminosities, fitted temp and Av
	np.savetxt(runfolder + '/results/mass_fit_results.txt', np.column_stack((num, ages, masses, lums, out_t, av)),\
		 header= '#number, age (myr), Mass, Luminosity, fitted temperature, extinction')
	#return the derived masses and luminosities 
	return masses, lums

def plot_specs(num, run, res = 3000, plot_reg = [6000, 9000]):
	'''Uses fit output files to plot the data and best fit spectrum for a given system.

	Args:
		num (int): system number to analyze
		run (string): name of run directory (anticipated structure is run/specs/[spectra di])
		res (int): spectral resolution. Default is 3000.
		plot_reg (list): Beginning and end points of spectral region to plot, in Angstroms. Default is 6000-9000 A.

	Returns:
		Writes a figure with the best fit spectrum and the data spectrum to disk

	'''
	#get the working directory
	cwd = os.getcwd()
	#read in the data spectrum - this is already extincted, broadened, and with pixellation imposed
	wl, spec = np.genfromtxt(run + '/specs/spec_{}.txt'.format(num), unpack = True)

	wl, spec = mft.broaden(wl, spec, res)
	res_element = np.mean(wl)/res
	spec_spacing = wl[1] - wl[0]
	if 3 * spec_spacing < res_element:
		factor = (res_element/spec_spacing)/3
		wl, spec = mft.redres(wl, spec, factor)

	#make an evenly spaced wavelength vector and interpolate the data spectrum onto it 
	ewl = np.linspace(wl[0], wl[-1], len(wl))
	inte = interp1d(wl, spec)
	espec = inte(ewl)

	#read in all the initial values for all systems
	p_num, p_mass, mul, p_age, p_av, p_temp, p_logg, p_luminosity, distance, pvmag, prmag, pimag, pjmag, phmag, pkmag = np.genfromtxt(run + '/ppar.txt', unpack = True)
	#find the correct values for system number, extinction, multiplicity, temperature, and log(g) for the primary star in this specific system
	pav, pnum, multiplicity, p_temp, p_logg = p_av[np.where(p_num == num)[0]], \
		int(p_num[np.where(p_num == num)[0]]), mul[np.where(p_num == num)[0]], p_temp[np.where(p_num == num)[0]], p_logg[np.where(p_num == num)[0]]

	#read in the fitting values for the step number, chi square, temperature, log(g), normalization, and extinction
	numb, xs, temp, lg, norm, ext = np.genfromtxt(cwd + '/' + run+'/results/parvals_{}.txt'.format(num), unpack = True)
	#find the best fit value by getting the minimum chi square value
	idx = -1 #np.where(xs == min(xs))[0][0]
	#define variables for the best fit values
	t, l, n, extinct = temp[idx], lg[idx], norm[idx], ext[idx]

	#get the best fit spectrum using the best fit temperature and log(g)
	#use the input wavelength vector to define the region so it covers the same space as the data
	w_, s_ = mft.get_spec(t, l, [min(wl)/1e4, max(wl)/1e4], normalize = False, resolution = res)

	#extinct the best fit spectrum using the best fit extinction value
	s_ = mft.extinct(np.asarray(w_), np.asarray(s_), extinct)

	#make sure everything is an array 
	wl, w_, spec, s_ = np.array(wl), np.array(w_), np.array(spec), np.array(s_)

	#multiply the best fit spectrum by the best fit normalization so scaling is correct
	s_ = np.array([ss * n for ss in s_])

	#initialize the figure
	fig1, ax1 = plt.subplots()#figsize = (8, 4))#(nrows = 2, sharex = True)

	#if it's a single star 
	if int(multiplicity) == 0:
		#if something weird happened with formatting, fix it
		if not type(p_temp) is float:
			p_temp = p_temp[0]
			p_logg = p_logg[0]
		#plot the input data spectrum, and include the temperature, log(g), and extinction values in the legend
		ax1.plot(wl, spec, color = 'navy', label = 'Input: T = {:.0f}, extinction = {:.1f}'.format(p_temp, float(pav)))
	#if it's a multiple
	else:
		#read in the secondary star attributes for all systems
		s_num, s_mass, s_sep, s_age, eccentricity, period, s_temp, s_logg, s_luminosity, svmag, srmag, simag, sjmag, shmag, skmag = np.genfromtxt(run + '/spar.txt', unpack = True)
		#as long as there's more than one
		if np.size(s_num) > 1:
			#find the correct temperature and log(g) for the secondary in this system
			sn = np.where(s_num == pnum)[0]
			s_temp, s_logg = s_temp[sn], s_logg[sn]

		#if it's weirdly formatted for some reason, fix it
		if not type(s_temp) is float:
			s_temp = s_temp[0]
			s_logg = s_logg[0]

		#plot the data spectrum (the SAME THING as above in the single part of the loop) but include both the primary and secondary temperature and log(g)
		#as well as the extinction, in the legend
		ax1.plot(wl, spec, color = 'navy', \
			label = 'Input: T1 = {:.0f}, T2 = {:.0f}, \nextinction = {:.1f}'\
			.format(float(p_temp), float(s_temp), float(pav)))

	#plot the best fit spectrum and include its temperature, log(g), chi square, and extinction in the legend
	ax1.plot(w_, s_, color='xkcd:sky blue', \
		label = 'Best fit model: \nT = {:.0f}, \nChi sq = {:.2f}, extinction = {:.1f}'.format(float(t), xs[idx], float(extinct)), linestyle= '-')
	#set plot parameters and make labels, and save the figure
	plt.minorticks_on()
	ax1.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax1.tick_params(bottom=True, top =True, left=True, right=True)
	ax1.tick_params(which='both', labelsize = "large", direction='in')
	ax1.tick_params('both', length=8, width=1.5, which='major')
	ax1.tick_params('both', length=4, width=1, which='minor')
	ax1.set_xlabel(r'$\lambda$ (\AA)', fontsize = 13)
	ax1.set_ylabel(r'$L_{\lambda}$', fontsize = 13)
	ax1.set_xlim(min(plot_reg), max(plot_reg))
	ax1.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(run + '/plot_spec_{}.pdf'.format(num))
	plt.close(fig1)

	#plot the same scheme, but zooming in on the TiO bandhead at ~6900A
	# fig2, ax2 = plt.subplots()
	# if multiplicity == 0:
	# 	ax2.plot(wl, spec, color = 'navy', label = 'Input: T = {:.0f}, log(g) = {:.1f}'.format(p_temp, p_logg))
	# else:
	# 	ax2.plot(wl, spec, color = 'navy', label = 'Input: T1 = {:.0f}, T2 = {:.0f}, \nlog(g)1 = {:.1f}, log(g)2 = {:.1f}'\
	# 		.format(float(p_temp), float(s_temp), float(p_logg), float(s_logg)))
	# ax2.plot(w_, s_, color='xkcd:sky blue', label = 'Best fit model: \nT = {:.0f}, log(g) = {:.1f}, \nChi sq = {:.2f}, extinction = {:.1f}'\
	# 	.format(float(t), float(l), xs[idx], float(extinct)), linestyle= '-')
	# plt.minorticks_on()
	# ax2.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	# ax2.tick_params(bottom=True, top =True, left=True, right=True)
	# ax2.tick_params(which='both', labelsize = "large", direction='in')
	# ax2.tick_params('both', length=8, width=1.5, which='major')
	# ax2.tick_params('both', length=4, width=1, which='minor')
	# ax2.set_xlabel(r'$\lambda$ (\AA)', fontsize = 13)
	# ax2.set_ylabel(r'$L_{\lambda}$', fontsize = 13)
	# ax2.set_xlim(6850, 7250)
	# ax2.legend(fontsize = 13, loc = 'best')
	# plt.tight_layout()
	# plt.savefig(run + '/plot_spec_TiO_{}.pdf'.format(num))
	# plt.close(fig2)

	#same plots as above, but focusing on the 8700A Ca triplet
	# fig3, ax3 = plt.subplots()
	# if multiplicity == 0:
	# 	ax3.plot(wl, spec, color = 'navy', label = 'Input: T = {:.0f}, log(g) = {:.1f}'.format(p_temp, p_logg))
	# else:
	# 	ax2.plot(wl, spec, color = 'navy', label = 'Input')#: T1 = {:.0f}, T2 = {:.0f}, \nlog(g)1 = {:.1f}, log(g)2 = {:.1f}'\
	#		.format(float(p_temp), float(s_temp), float(p_logg), float(s_logg)))
	# ax3.plot(w_, s_, color='xkcd:sky blue', label = 'Best fit model')#: \nT = {:.0f}, log(g) = {:.1f}, \nChi sq = {}'\
	#	.format(float(t[0]), float(l[0]), float((str(xs[idx]).split('.')[0].split('[')[-1])[0])), linestyle= ':')
	# plt.minorticks_on()
	# ax3.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	# ax3.tick_params(bottom=True, top =True, left=True, right=True)
	# ax3.tick_params(which='both', labelsize = "large", direction='in')
	# ax3.tick_params('both', length=8, width=1.5, which='major')
	# ax3.tick_params('both', length=4, width=1, which='minor')
	# ax3.set_xlabel(r'$\lambda$ (\AA)', fontsize = 13)
	# ax3.set_ylabel(r'$L_{\lambda}$', fontsize = 13)
	# ax3.set_xlim(8400, 8950)
	# ax3.legend(fontsize = 13, loc = 'best')
	# plt.tight_layout()
	# plt.savefig(run + '/plot_spec_CaIR_{}.pdf'.format(num))
	# plt.close(fig3)

	return 

def run_pop(nsys, run, new_pop = False, region1 = [0.6, 0.9], region2 = [0.51, 0.535], bf = [0.26, 0.4, 0.45, 0.5], md = [0.5, 0.7, 1.5], \
	age_range = [1.01, 3.01], extinction = 0, res1 = 3000, res2 = 34000, nsteps = 100, reduce_ms = False, m_reduce = 0.8):

	cwd = os.getcwd()

	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

	if new_pop == True:

		if rank == 0:
			if not os.path.exists(cwd + '/' + run):
				os.mkdir(cwd + '/' + run)
			else:
				pass
			if not os.path.exists(cwd + '/' + run + '/specs'):
				os.mkdir(cwd + '/' + run + '/specs')
			else:
				pass
			if not os.path.exists(cwd + '/' + run + '/results'):
				os.mkdir(cwd + '/' + run + '/results')
			else:
				pass

		else:
			pass

		print('Making systems')

		ns = np.arange(0, nsys)

		s = comm.scatter(ns, root = 0)
		aa = extinction * np.random.rand(nsys)

		p = make_binary_sys(s, 1, bf, md, age_range, aa[rank], run, reg = [min(min(region1), min(region2)) - 0.005, max(max(region1), max(region2)) + 0.005],\
			 res = max(res1, res2), reduce_ms = reduce_ms, m_reduce = m_reduce)

		out = comm.gather(p, root = 0)

		if len(p) == 2:
			pp, sp = p
		else:
			pp = p

		ps = ''
		for n in range(len(pp)):
			ps = ps + str(pp[n]) + ' '

		try:
			ss = ''
			for n in range(len(sp)):
				ss = ss + str(sp[n]) + ' '
		except:
			pass

		ppar = open(cwd + '/' + run + '/ppar.txt', 'a')
		spar = open(cwd + '/' + run +'/spar.txt', 'a')

		ppar.write(ps + '\n')

		try:
			spar.write(ss + '\n')
		except:
			pass
		ppar.close()
		spar.close()


	print('Fitting now')
	n, ex_g, t_g, lg_g = np.genfromtxt(cwd +'/' + run + '/ppar.txt', usecols = (0,4,5,6), unpack = True, autostrip = True)
	
	files = np.array([run + '/specs/spec_{}.txt'.format(n) for n in range(nsys)])

	if type(lg_g) is not np.ndarray:
		tg, lg, norm, ex, fi = t_g, lg_g, 5e23, ex_g, files[rank]
	else:
		tg, lg, norm, ex, fi = t_g[np.where(n == rank)[0][0]], lg_g[np.where(n == rank)[0][0]], 5e23, ex_g[np.where(n == rank)[0][0]], files[rank]
	t1 = time.clock()
	outcs = fit_test(tg, lg, [norm], ex, fi, res = [res1], reg = [region1], nsteps = 20, cutoff = False)
	t3 = time.clock()
	print('Time for initial fit: ', t3 - t1)
	step, cs, temp, logg, normalize, extinct = np.genfromtxt(run + '/results/parvals_{}.txt'.format(rank), unpack = True)
	t2 = time.clock()
	print('file read time:', t2 - t3)
	if temp[np.where(cs == min(cs))[0][0]] > 5500:
	#order should be lowres, highres, lowres_region, highres_region
		oc = fit_test(temp[np.where(cs == min(cs))[0][0]], logg[np.where(cs == min(cs))[0][0]], [normalize[np.where(cs == min(cs))[0][0]], normalize[np.where(cs == min(cs))[0][0]]],\
				 extinct[np.where(cs == min(cs))[0][0]], fi, res = [res1, res2], nsteps = nsteps, reg = [region1, region2], perturb_init = False, cutoff = True)
	else:
		oc = fit_test(temp[np.where(cs == min(cs))[0][0]], logg[np.where(cs == min(cs))[0][0]], [normalize[np.where(cs == min(cs))[0][0]]],\
				 extinct[np.where(cs == min(cs))[0][0]], fi, res = [res1], nsteps = nsteps, reg = [region1], perturb_init = False, cutoff = True)

	print('Time for second fit:', time.clock() - t2)
	ga = comm.gather(oc, root = 0)
	
	print('done!')

	return

def plot_init_pars(run, pri_num, sec_num, pri_mass, sec_mass, sep, av, distance):
	mr = []
	sys_mass = []
	for n in pri_num:
		if n in sec_num:
			mr.append(float(sec_mass[np.where(sec_num == n)]/pri_mass[np.where(pri_num == n)][0]))
			sys_mass.append((pri_mass[np.where(pri_num == n)][0] + sec_mass[np.where(sec_num == n)[0]])[0])
		else:
			sys_mass.append(pri_mass[np.where(pri_num == n)][0])

	fig, [ax1, ax2] = plt.subplots(nrows = 2)
	ax1.hist(sep, color = 'navy')
	ax1.set_xlabel('Separation (AU)')
	ax1.set_ylabel('Number')

	ax2.hist(mr, bins = 20, color='xkcd:sky blue')
	ax2.set_xlabel(r'Mass ratio (secondary/primary), M$_{\odot}$')
	ax2.set_ylabel('Number')

	plt.minorticks_on()
	ax1.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax1.tick_params(bottom=True, top =True, left=True, right=True)
	ax1.tick_params(which='both', labelsize = "large", direction='in')
	ax1.tick_params('both', length=8, width=1.5, which='major')
	ax1.tick_params('both', length=4, width=1, which='minor')

	ax2.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax2.tick_params(bottom=True, top =True, left=True, right=True)
	ax2.tick_params(which='both', labelsize = "large", direction='in')
	ax2.tick_params('both', length=8, width=1.5, which='major')
	ax2.tick_params('both', length=4, width=1, which='minor')

	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + run + '/mr_and_sep.pdf')
	plt.close(fig)

	fig = plt.figure(figsize=(6, 6))
	grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
	main_ax = fig.add_subplot(grid[:-1, 1:], xticklabels=[], yticklabels=[])
	y_hist = fig.add_subplot(grid[:-1, 0], xticklabels = [])#, sharey=main_ax)
	x_hist = fig.add_subplot(grid[-1, 1:], yticklabels = [])#, sharex=main_ax)

	main_ax.plot(distance, av, 'ok', alpha = 0.5)
	#main_ax.set_xlabel('Distance (pc)')
	#main_ax.set_ylabel(r'A$_{V}$, (mag)')

	x_hist.hist(distance, orientation = 'vertical')
	x_hist.set_xlabel('Distance (pc)')
	x_hist.invert_yaxis()

	y_hist.hist(av, orientation = 'horizontal')
	y_hist.set_ylabel('Extinction (AV, mag)')
	y_hist.invert_xaxis()

	plt.minorticks_on()
	main_ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	main_ax.tick_params(bottom=True, top =True, left=True, right=True)
	main_ax.tick_params(which='both', labelsize = "large", direction='in')
	main_ax.tick_params('both', length=8, width=1.5, which='major')
	main_ax.tick_params('both', length=4, width=1, which='minor')

	x_hist.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	x_hist.tick_params(bottom=True, top =True, left=True, right=True)
	x_hist.tick_params(which='both', labelsize = "large", direction='in')
	x_hist.tick_params('both', length=8, width=1.5, which='major')
	x_hist.tick_params('both', length=4, width=1, which='minor')

	y_hist.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	y_hist.tick_params(bottom=True, top =True, left=True, right=True)
	y_hist.tick_params(which='both', labelsize = "large", direction='in')
	y_hist.tick_params('both', length=8, width=1.5, which='major')
	y_hist.tick_params('both', length=4, width=1, which='minor')

	# plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + run + '/dist_av.pdf')
	plt.close(fig)

	nmr = np.linspace(min(sys_mass), max(sys_mass), len(sys_mass) * 100)
	cm = make_chabrier_imf(nmr)
	primary_numbers = [int(np.rint(len(pri_num) * p)) * 2 for p in cm]

	fig, ax = plt.subplots()
	ax.hist(sys_mass, bins = np.logspace(np.log10(min(sys_mass)), np.log10(max(sys_mass)), 15), label = 'Single + binary stars', color = 'b')
	ax.hist(pri_mass, bins = np.logspace(np.log10(min(pri_mass)), np.log10(max(pri_mass)), 15), label = 'Single stars', facecolor = 'cyan', alpha = 0.7)
	#ax.hist(smass, bins = np.logspace(np.log10(min(smass)), np.log10(max(smass)), 20), label = 'Secondary', alpha = 0.5, color = 'teal')
	ax.plot(nmr, primary_numbers, label = 'Theoretical IMF', linestyle = '--', color = 'tab:orange')
	plt.axvline(x = np.median(pri_mass), label = 'Median stellar mass', color='red')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel(r'Mass ($M_{\odot}$)', fontsize = 13)
	ax.set_ylabel('Number of stars', fontsize = 13)
	ax.set_xlim(0.08, 2.1)
	#ax.set_ylim(1e2, 1e3)
	#ax.set_yscale('log')
	ax.set_xscale('log')
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + run + '/masshist_init.pdf')
	plt.close(fig)
	return

def final_analysis(nsys, run, plots = False, res = 3000, reg = [6000, 9000]):
	print('Plotting!')

	if plots == True:
		[plot_specs(n, run, res = res, plot_reg = reg) for n in np.random.randint(low = 0, high = nsys, size = 10)]

	print('Running mass analysis')

	analyze_sys(run)

	num, age, fit_mass, logg, fit_t, extinct = np.genfromtxt(run + '/results/mass_fit_results.txt', unpack = True)

	n, mass, multiplicity, page, av, ptemp, plogg, pluminosity, distance, pvmag, prmag, pimag, pjmag, phmag, pkmag = np.genfromtxt(run + '/ppar.txt', unpack = True)

	n, mass, multiplicity, page, av, ptemp, plogg, pluminosity, distance, pvmag, prmag, pimag, pjmag, phmag, pkmag = n[np.argsort(n)], mass[np.argsort(n)], \
		multiplicity[np.argsort(n)], page[np.argsort(n)], av[np.argsort(n)], ptemp[np.argsort(n)], plogg[np.argsort(n)], pluminosity[np.argsort(n)], \
		distance[np.argsort(n)], pvmag[np.argsort(n)], prmag[np.argsort(n)], pimag[np.argsort(n)], pjmag[np.argsort(n)], phmag[np.argsort(n)], pkmag[np.argsort(n)]

	try:
		sn, smass, sep, sage, seccentricity, period, stemp, slogg, sluminosity, svmag, srmag, simag, sjmag, shmag, skmag = np.genfromtxt(run + '/spar.txt', unpack = True)

		sn, smass, sep, sage, seccentricity, period, stemp, slogg, sluminosity, svmag, srmag, simag, sjmag, shmag, skmag = sn[np.argsort(sn)], smass[np.argsort(sn)], \
			sep[np.argsort(sn)], sage[np.argsort(sn)], seccentricity[np.argsort(sn)], period[np.argsort(sn)], stemp[np.argsort(sn)], slogg[np.argsort(sn)], \
			sluminosity[np.argsort(sn)], svmag[np.argsort(sn)], srmag[np.argsort(sn)], simag[np.argsort(sn)], sjmag[np.argsort(sn)], shmag[np.argsort(sn)], skmag[np.argsort(sn)]

		sluminosity = np.asarray([np.log10(10**(-0.4 * s)) for s in sluminosity])
	except:
		stemp, sluminosity = [], []

	fig, ax = plt.subplots()
	ax.plot(ptemp, pluminosity, 'v', color = 'navy', label = 'Primary stars')
	ax.plot(stemp, sluminosity, 'o', color='xkcd:sky blue', label = 'Secondary stars')
	ax.set_xlabel('Temperature (K)', fontsize = 13)
	ax.set_ylabel('log10(luminosity) (solar lum.)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	ax.invert_xaxis()
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + run + '/hrd_input.pdf')
	plt.close()

	teff, ext = [], []
	for k in n:
		try:
			idx = np.where(num == int(k))[0][0]
			teff.append(fit_t[idx])
			ext.append(float(extinct[idx]))
		except:
			pass

	# plot_init_pars(run, n, sn, mass, smass, sep, extinct, distance)

	sys_lum = []

	mass_resid = [] 
	fm = []
	mm = []
	for k in n: 
		try:
			idx = np.where(num == int(k))
			m = mass[int(k)]
			l = pluminosity[int(k)]
			# if int(k) in sn:
			# 	m += smass[np.where(sn == int(k))][0]
			# 	l += sluminosity[np.where(sn == int(k))][0]
			test = m - fit_mass[idx] 
			fm.append(fit_mass[idx][0])
			mm.append(m)
			sys_lum.append(l)
			mass_resid.append(test[0]) 
		except:
			pass

	sys_flux = [((10**sys_lum[n]) * 3.9e33) /(4 * np.pi * (distance[n] * 3.086e18)**2) for n in range(len(num))]
	sys_mag_app = [-2.5 * np.log10(sf/17180) for sf in sys_flux]

	mr = [mass_resid[n]/mm[n] for n in range(len(mass_resid))]


	plt.figure(1)
	for k in n:
		k = int(k)
		try:
			# plt.plot([ptemp[k], teff[k]], [av[k], ext[k]], color = 'k', alpha = 0.5)
			if k == 1:
				# plt.scatter(ptemp[k], av[k], color = 'navy', label = 'input')
				plt.scatter(teff[k], ext[k], color = 'xkcd:sky blue', label = 'output')
			else:
				# plt.scatter(ptemp[k], av[k], color = 'navy')
				plt.scatter(teff[k], ext[k], color = 'xkcd:sky blue')
		except:
			pass
	plt.minorticks_on()
	plt.legend(fontsize = 13, loc = 'best')
	plt.xlabel(r'T$_{eff}$ (K)', fontsize = 13)
	plt.ylabel(r'A$_{V}$ (mag)', fontsize = 13)
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/temp_av.pdf')
	plt.close()


	chab = make_chabrier_imf(np.linspace(min(mm), max(mm), 300))
	plt.figure(1)
	plt.hist(np.log10(mm), label = 'input', color = 'navy', bins = np.linspace(min(np.log10(mm)), max(np.log10(mm)), 7), log=True)
	plt.hist(np.log10(fit_mass), color = 'xkcd:sky blue', label = 'output', alpha = 0.6, bins = np.linspace(min(np.log10(mm)), max(np.log10(mm)), 7), log=True) 
	# plt.plot(np.linspace(min(mm), max(mm), 300), chab * 1200, linestyle = '-', color = 'xkcd:light red')
	plt.legend(fontsize = 13)
	# plt.gca().invert_xaxis()
	plt.minorticks_on()
	plt.xlabel(r'Mass (M$_{\odot}$)', fontsize = 13)
	plt.ylabel('Number', fontsize = 13)
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/masshist.pdf')
	plt.close()

	plt.figure(2)
	plt.hist(mass_resid, color='navy')	
	plt.xlabel(r'Mass residual (M$_{\odot}$)', fontsize = 13)
	plt.ylabel('Number', fontsize = 13)
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/error_fit.pdf')
	plt.close()

	plt.figure(3)
	plt.hist(mr, color = 'navy', label = 'Avg. frac. error = {:.2f}\n 1 stdev = {:.2f}'.format(float(np.mean(mr)), float(np.std(mr))))
	plt.legend(fontsize = 13) 
	plt.xlabel('Fractional mass residual', fontsize = 13)
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/error_fit_fractional.pdf')
	plt.close()

	one_one = np.arange(0, 4)
	plt.figure(4)
	plt.scatter(mm, fm)
	plt.plot(one_one, one_one, ':', label = 'One-to-one correspondence line')
	plt.xlabel(r'Expected mass (M$_{\odot}$)', fontsize =14)
	plt.ylabel(r'Fitted mass (M$_{\odot}$)', fontsize =14)
	plt.legend(fontsize = 13, loc = 'best')
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/mass_scatter.pdf')
	plt.close()

	plt.figure(5)
	plt.scatter(fit_t, sys_lum)
	plt.xlabel('Fitted Temperature (K)', fontsize =14)
	plt.gca().invert_xaxis()
	plt.ylabel(r'log(System Luminosities) (L$_{\odot}$)', fontsize = 13)
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/HRD_output.pdf')
	plt.close()

	plt.figure(6)
	plt.scatter(fit_t, sys_mag_app)
	plt.xlabel('Fitted Temperature', fontsize = 13)
	plt.ylabel(r'System apparent bolometric magnitude', fontsize = 13)
	plt.gca().invert_yaxis()
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/HRD_output_flux.pdf')
	plt.close()

	md = [mm[n] - fm[n] for n in range(len(mm))]
	mm, md = np.array(mm), np.array(md)
	md1 = md[np.where(multiplicity == 0)]
	mm1 = mm[np.where(multiplicity == 0)]
	md2 = md[np.where(multiplicity == 1)]
	mm2 = mm[np.where(multiplicity == 1)]
	mdiff = []
	try:
		mass, smass = np.array(mass), np.array(smass)
		for number in sn:
			pri = mass[np.where(n == number)[0][0]]
			mdiff.append(float(smass[np.where(sn == number)[0][0]])/pri)
	except:
		mass, smass = np.array(mass), []

	d1, d2 = md1[np.where(mm1 < 0.95)], md1[np.where(mm1 >= 0.95)]

	plt.figure(7)
	plt.scatter(mm1, md1, color='navy', s = 20, marker = '.', label = r'Single Star Error:\\{:.2f} $\pm$ {:.2f} (T $<$ 4300 K)\\{:.2f} $\pm$ {:.2f} (T $\geq$ 4300 K)'\
		.format(np.nanmean(d1), np.nanstd(d1), np.nanmean(d2), np.nanstd(d2)))
	if len(mm2) > 2:
		a = plt.scatter(mm2, md2, c = np.array(mdiff), cmap = plt.cm.plasma, marker = 'x', label = 'Binary stars')
		cbar = plt.colorbar(a)
		cbar.set_label(r'Mass ratio')
	plt.plot((min(mm), max(mm)), (0,0), ':', label = 'Zero error line')
	plt.xlabel(r'Input mass (M$_{\odot}$)', fontsize = 13)
	plt.ylabel('Input mass - fitted mass', fontsize = 13)
	plt.legend(fontsize = 13, loc = 'best')
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/mass_diff.pdf')
	plt.close()

	return

def example_sys(teff, logg, norm, extinct, filename):

	logg, logg_try = 4, 4

	wl, spec = np.genfromtxt(filename, unpack = True)

	teff += np.random.normal(scale = 100)
	
	ww_init, ss_init = mft.get_spec(teff, logg, [0.45, 0.9], normalize = False)

	second_wl = np.linspace(max(wl[0], ww_init[0]), min(wl[-1], ww_init[-1]), len(spec))

	dd2 = interp1d(wl, spec)
	md2_init = interp1d(ww_init, ss_init)
	spec2 = dd2(second_wl)
	ss_init = md2_init(second_wl)

	extinct += np.random.normal(scale = 0.1)
	while extinct < 0:
		extinct += np.random.rand()/2

	ss_init = mft.extinct(second_wl, ss_init, extinct)
	normalization = np.mean(spec2)/np.mean(ss_init)

	normalization += np.random.normal(scale = np.abs(normalization * 0.02))

	ss_init = [s * normalization for s in ss_init]
	vary = [sp * 0.01 for sp in spec2]
	cs_init = mft.chisq(ss_init, spec2, vary)

	niter = 0
	total_iter = 0

	plt.figure()
	plt.plot(second_wl, ss_init, label = 'data')
	plt.plot(second_wl, spec2, label = 'test')
	plt.savefig('test/fit_test.png')
	plt.close()

	f = open('test/parvals.txt', 'a')
	f.write('{} {} {} {} {} {}\n'.format(niter, np.sum(cs_init)/len(cs_init), teff, logg, normalization, extinct))
	f.close()

	#everything block
	while niter < 50:
		rn = np.random.randint(0, 3)
		rn2 = np.random.rand()

		parvar = np.asarray([teff, normalization, extinct])

		if niter < 10:
			var = np.asarray([np.random.normal(scale = 100), np.random.normal(scale=np.abs(normalization*0.05)), np.random.normal(scale = 0.1)])[rn]
			while rn == 2 and parvar[rn] + var < 0:
				var = np.random.normal(scale = 0.5)

		else:
			var = np.asarray([np.random.normal(scale = 10), np.random.normal(scale=np.abs(normalization*0.02)), np.random.normal(scale = 0.05)])[rn]
			while rn == 2 and parvar[rn] + var < 0:
				var = np.random.normal(scale = 0.5)

		parvar[rn] = parvar[rn] + var

		teff_try, normalization_try, extinct_try = parvar

		if teff_try < 7000 and teff_try > 2000:

			ww_i, ss_i = mft.get_spec(teff_try, logg, [0.45, 0.9], normalize = False)

			mi2_init = interp1d(ww_i, ss_i)
			sse_i = mi2_init(second_wl)

			sse_i = mft.extinct(second_wl, sse_i, extinct_try)

			sse_i = [s * normalization_try for s in sse_i]
			vv = [0.01 * s for s in spec2]

			plt.figure()
			plt.plot(second_wl, sse_i, label = 'test, {}'.format(teff_try))
			plt.plot(second_wl, spec2, label = 'data')
			plt.plot(second_wl, vv, label = 'error')
			plt.legend()
			plt.savefig('test/fit_test_{}.png'.format(total_iter))
			plt.close()

			cs_try = [((sse_i[n] - spec2[n])**2)/vv[n]**2 for n in range(len(sse_i))]#returns a chi^2 vector

			print('try, init', np.sum(cs_try)/len(cs_try), np.sum(cs_init)/len(cs_init))

			if np.sum(cs_try)/len(cs_try) < np.sum(cs_init)/len(cs_init): #uses reduced chi^2 as my statistic

				teff = teff_try
				cs_init = cs_try
				if niter >= 20:
					niter = 20
				else:
					niter = 0
			else:
				niter += 1

			f = open('test/parvals.txt', 'a')
			f.write('{} {} {} {} {} {}\n'.format(niter, np.sum(cs_init)/len(cs_init), teff, logg, normalization, extinct))
			f.close()

			f = open('test/testpars.txt', 'a')
			f.write('{} {} {} {} {} {}\n'.format(rn, np.sum(cs_try)/len(cs_try), teff_try, logg_try, normalization_try, extinct_try))
			f.close()
			total_iter += 1

		else:
			teff_try, normalization_try, extinct_try = teff_try + np.random.normal(scale = 200),\
				normalization_try + np.random.normal(scale = np.abs(normalization_try * 0.1)), extinct_try + np.random.normal(scale = 0.1)
	return np.sum(cs_try)/len(cs_try)

def compare_two_runs(run1, run2):
	number1, age1, mass1, lum1, temp1, extinct1 = np.genfromtxt(run1 + '/results/mass_fit_results.txt', unpack = True) 
	number2, age2, mass2, lum2, temp2, extinct2 = np.genfromtxt(run2 + '/results/mass_fit_results.txt', unpack = True)
	print('Median mass, run 1: ', np.median(mass1), '+/-', np.std(mass1))
	print('Median mass, run 2: ', np.median(mass2), '+/-', np.std(mass2))
	fig, ax = plt.subplots()
	ax.hist(mass1, color = 'navy', label = 'Single mass fit', bins = np.logspace(np.log10(min(min(mass1), min(mass2))), np.log10(max(max(mass1), max(mass2))), 15), log = True)
	ax.hist(mass2, color = 'xkcd:sky blue', alpha = 0.6, label = 'Binary mass fit', bins = np.logspace(np.log10(min(min(mass1), min(mass2))), np.log10(max(max(mass1), max(mass2))), 15), log = True)
	ax.axvline(np.median(mass1), label = 'Single median mass', color='navy')
	ax.axvline(np.median(mass2), label = 'Binary median mass', color = 'xkcd:sky blue')
	ax.set_xscale('log')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel(r'Single and Binary Masses (M$_{\odot}$)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/comp_mass_{}_{}.pdf'.format(run1, run2))
	plt.close()

	print('average age, run 1: ', np.mean(age1), '+/-', np.std(age1))
	print('average age, run 2: ', np.mean(age2), '+/-', np.std(age2))
	fig, ax = plt.subplots()
	ax.hist(age1, color = 'navy', label = 'Single age fit', bins = np.linspace(min(min(age1), min(age2)), max(max(age1), max(age2)), 10))
	ax.hist(age2, color = 'xkcd:sky blue', alpha = 0.6, label = 'Binary age fit', bins = np.linspace(min(min(age1), min(age2)), max(max(age1), max(age2)), 10))
	ax.axvline(np.mean(age1), label = 'Single mean age', color='navy')
	ax.axvline(np.mean(age2), label = 'Binary mean age', color = 'xkcd:sky blue')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Single and Binary Ages (Myr)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/comp_age_{}_{}.pdf'.format(run1, run2))
	plt.close()

	#divisions from
	divs = [2570, 2670, 2770, 2860, 2980, 3190, 3410, 3560, 3720, 3900, 4020, 4210, 4710, 4870, 5180, 5430, 5690, 5930, 6130, 6600]
	labels = ['M9', 'M8', 'M7', 'M6', 'M5', 'M4', 'M3', 'M2', 'M1', 'M0', 'K7', 'K5', 'K2', 'K0', 'G8', 'G5', 'G2', 'G0', 'F8', 'F5']
	fig, ax = plt.subplots()
	ax.hist(temp1, bins = divs, label = "Single SpTs", color = 'navy')
	ax.hist(temp2, bins = divs, label = "Binary SpTs", color = 'xkcd:sky blue', alpha = 0.7)
	ax.set_xticks(divs[::2])
	ax.set_xticklabels(labels[::2], fontsize=14)
	#ax.set_yscale('log')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Single and Binary spectral type dist.', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/comp_spt_{}_{}.pdf'.format(run1, run2))
	plt.close()

# bf = [0.6, 0.8, 1]

run_pop(480, 'run27', new_pop = True, region1 = [0.56, 0.69], region2 = [0.385, 0.54], bf = [0.6, 0.6, 0.8], md = [0.8, 1.2], \
	age_range = [9, 11], extinction = 2, res1 = 2100, res2 = 1650, nsteps = 50, reduce_ms = True, m_reduce = 0.85)
# final_analysis(144, 'run26', res = 2100, plots = False, reg = [5600, 6900])
# final_analysis(96, 'run20', res = 2100, plots = False, reg = [5600, 6900])
# final_analysis(480, 'run21', res = 2100, plots = False, reg = [5600, 6900])
# final_analysis(240, 'run17', res = 34000, plots = False, reg = [5100, 5350])
# final_analysis(96, 'run18', res = 34000, plots = True, reg = [5650, 5950])

