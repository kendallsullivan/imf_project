from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import operator

#M1: Luhman + 2003 APJ 593 1093

iso = glob('baraffe_*')

age = [int(iso[n].split('_')[1]) * 1e-1 for n in range(len(iso))]

fig, ax = plt.subplots()
for n, file in enumerate(iso):
   if age[n] < 10:
      fi = np.loadtxt(file, comments = '!')
      jmag = fi[:, 6]
      teff = fi[:, 1]
      ax.plot(teff, jmag, label = age[n])
ax.set_ylabel('J magnitude (abs)', fontsize = 14)
ax.set_xlabel('Effective temperature (K)', fontsize = 14)
ax.set_title("d = 130 pc")
ax.legend(loc = "best")

ax.plot([4000], [3.33], marker = 'o', color = 'k', label = 'S A')
ax.plot([3700], [4.1], marker = "v", color = 'k', label = "S B")
ax.plot([3850], [4.4], marker = "o", color = 'r', label = 'VV A')
ax.plot([3700], [4.55], marker = "v", color = 'r', label = 'VV B')
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])
plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = "large", direction='in')
handles, labels = ax.get_legend_handles_labels()
hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
handles2, labels2 = zip(*hl)

ax.legend(handles2, labels2)

plt.show()

fig, ax = plt.subplots()
for n, file in enumerate(iso):
   if age[n] < 10:
      fi = np.loadtxt(file, comments = '!')
      hmag = fi[:, 7]
      teff = fi[:, 1]
      ax.plot(teff, hmag, label = age[n])
ax.set_ylabel('H magnitude (abs)', fontsize = 14)
ax.set_xlabel('Effective temperature (K)', fontsize = 14)
ax.set_title("d = 130 pc")
ax.legend(loc = "best")

ax.plot([4000], [1.22], marker = 'o', color = 'k', label = 'S A')
ax.plot([3700], [2.5], marker = "v", color = 'k', label = "S B")
ax.plot([3850], [2.8], marker = "o", color = 'r', label = 'VV A')
ax.plot([3700], [2.2], marker = "v", color = 'r', label = 'VV B')
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])
plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = "large", direction='in')
handles, labels = ax.get_legend_handles_labels()
hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
handles2, labels2 = zip(*hl)

ax.legend(handles2, labels2)

plt.show()


#---------------------------- USING D = 150 PC FROM GAIA ---------------------------- 
fig, ax = plt.subplots()
for n, file in enumerate(iso):
   if age[n] < 10:
      fi = np.loadtxt(file, comments = '!')
      jmag = fi[:, 6]
      teff = fi[:, 1]
      ax.plot(teff, jmag, label = age[n])
ax.set_ylabel('J magnitude (abs)', fontsize = 14)
ax.set_xlabel('Effective temperature (K)', fontsize = 14)
ax.set_title("d = 150 pc (Gaia)")
ax.legend(loc = "best")

ax.plot([4000], [3.01], marker = 'o', color = 'k', label = 'S A')
ax.plot([3700], [3.7], marker = "v", color = 'k', label = "S B")
ax.plot([3850], [4.09], marker = "o", color = 'r', label = 'VV A')
ax.plot([3700], [4.23], marker = "v", color = 'r', label = 'VV B')
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])
plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = "large", direction='in')
handles, labels = ax.get_legend_handles_labels()
hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
handles2, labels2 = zip(*hl)

ax.legend(handles2, labels2)

plt.show()

fig, ax = plt.subplots()
for n, file in enumerate(iso):
   if age[n] < 10:
      fi = np.loadtxt(file, comments = '!')
      hmag = fi[:, 7]
      teff = fi[:, 1]
      ax.plot(teff, hmag, label = age[n])
ax.set_ylabel('H magnitude (abs)', fontsize = 14)
ax.set_xlabel('Effective temperature (K)', fontsize = 14)
ax.set_title("d = 150 pc (Gaia)")
ax.legend(loc = "best")

ax.plot([4000], [0.9], marker = 'o', color = 'k', label = 'S A')
ax.plot([3700], [2.2], marker = "v", color = 'k', label = "S B")
ax.plot([3850], [2.5], marker = "o", color = 'r', label = 'VV A')
ax.plot([3700], [2.0], marker = "v", color = 'r', label = 'VV B')
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])
plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = "large", direction='in')
handles, labels = ax.get_legend_handles_labels()
hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
handles2, labels2 = zip(*hl)

ax.legend(handles2, labels2)

plt.show()