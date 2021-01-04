# %%
#_
#___________________________________________________________________________
#|                                                                         |
#| Playground para pruebas:                                                |
#|      De ser necesario probar código, este script puede ser usaado       |
#|      para ese propósito. El archivo se eliminará del repositorio        |
#|      una vez concluido el proyecto.                                     |
#|_________________________________________________________________________|

from TT_utilities import Case
from pathlib import Path

from wfdb.processing.qrs import gqrs_detect
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
#from memory_profiler import profile
from matplotlib import gridspec
from scipy.stats import stats
from pprint import pprint 
import seaborn as sns
import entropy as tpy
import pandas as pd
import numpy as np
import biosppy
import decimal
import json
import wfdb
import ast
import os
import re


# %%
from main import MainDF, MainDummy
# %%
""" ANÁLISIS ESPECTRAL USANDO WAVELETS"""

from __future__ import division
import numpy
from matplotlib import pyplot
import pycwt as wavelet
from pycwt.helpers import find

# Definir parámetros de la señal

dat = numpy.array(MainDummy.iloc[2]['rr'])
record = str(MainDummy.iloc[2]['record'])
#url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
#dat = numpy.genfromtxt(url, skip_header=19)
title = 'Señal de HRV_record ['+ record +']'
label = 'HRV Control '
units = 'mV'
t0 = 1871.0
dt = 0.25  # In years

# Time array
N = dat.size
t = numpy.arange(0, N) * dt + t0

# detrend and normalize the input data
p = numpy.polyfit(t - t0, dat, 1)
dat_notrend = dat - numpy.polyval(p, t - t0)
std = dat_notrend.std()  # Standard deviation
var = std ** 2  # Variance
dat_norm = dat_notrend / std  # Normalized dataset

# Parameters of wavelet analysis andMother Wavelet selection with w=6

mother = wavelet.Morlet(6)
s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
dj = 1 / 12  # Twelve sub-octaves per octaves
J = 7 / dj  # Seven powers of two with dj sub-octaves
alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

# Wavelet transform and inverse wavelet transform
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

# Normalized wavelet and Fourier power spectra

power = (numpy.abs(wave)) ** 2
fft_power = numpy.abs(fft) ** 2
period = 1 / freqs
power /= scales[:, None]

# Power spectra significance test
signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                         significance_level=0.95,
                                         wavelet=mother)
sig95 = numpy.ones([1, N]) * signif[:, None]
sig95 = power / sig95

# Global wavelet spectrum
glbl_power = power.mean(axis=1)
dof = N - scales  # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                        significance_level=0.95, dof=dof,
                                        wavelet=mother)
sel = find((period >= 2) & (period < 8))
Cdelta = mother.cdelta
scale_avg = (scales * numpy.ones((N, 1))).transpose()
scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                             significance_level=0.95,
                                             dof=[scales[sel[0]],
                                                  scales[sel[-1]]],
                                             wavelet=mother)

# Prepare the figure
pyplot.close('all')
pyplot.ioff()
figprops = dict(figsize=(11, 8), dpi=72)
fig = pyplot.figure(**figprops)

# First sub-plot, the original time series anomaly and inverse wavelet
# transform.
ax = pyplot.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(t, dat, 'k', linewidth=1.5)
ax.set_title('a) {}'.format(title))
ax.set_ylabel(r'{} [{}]'.format(label, units))

# Second sub-plot, the normalized wavelet power spectrum and significance
# level contour lines and cone of influece hatched area. Note that period
# scale is logarithmic.
bx = pyplot.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
bx.contourf(t, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
            extend='both', cmap=pyplot.cm.viridis)
extent = [t.min(), t.max(), 0, max(period)]
bx.contour(t, numpy.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
           extent=extent)
bx.fill(numpy.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                           t[:1] - dt, t[:1] - dt]),
        numpy.concatenate([numpy.log2(coi), [1e-9], numpy.log2(period[-1:]),
                           numpy.log2(period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
bx.set_title('b) {} Espectro de Potencia Wavelet- WPS ({})'.format(label, mother.name))
bx.set_ylabel('Periodo (s)')
#
Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
                           numpy.ceil(numpy.log2(period.max())))
bx.set_yticks(numpy.log2(Yticks))
bx.set_yticklabels(Yticks)

# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra. Note that period scale is logarithmic.
cx = pyplot.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
cx.plot(glbl_signif, numpy.log2(period), 'k--')
cx.plot(var * fft_theor, numpy.log2(period), '--', color='#cccccc')
cx.plot(var * fft_power, numpy.log2(1./fftfreqs), '-', color='#cccccc',
        linewidth=1.)
cx.plot(var * glbl_power, numpy.log2(period), 'k-', linewidth=1.5)
cx.set_title('c) Espectro Global de Wavelet')
cx.set_xlabel(r'Power [({})^2]'.format(units))
cx.set_xlim([0, glbl_power.max() + var])
cx.set_ylim(numpy.log2([period.min(), period.max()]))
cx.set_yticks(numpy.log2(Yticks))
cx.set_yticklabels(Yticks)
pyplot.setp(cx.get_yticklabels(), visible=False)

# Fourth sub-plot, the scale averaged wavelet spectrum.
dx = pyplot.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
dx.plot(t, scale_avg, 'k-', linewidth=1.5)
dx.set_title('d) {}--{} year scale-averaged power'.format(2, 8))
dx.set_xlabel('Time (year)')
dx.set_ylabel(r'Average variance [{}]'.format(units))
ax.set_xlim([t.min(), t.max()])

pyplot.show()

# %%
data = MainDF.sample(frac=1.0)
#Clean dataset
pathology = data[["cond","ae_m","ae_v","se_m","se_v","hfd_m","hfd_v","dfa_m","dfa_v","psd_m","psd_v"]]
#pathology
tar_labels =['AF','CHF','IM']
# Define train set and targets
#Group by pathology
a_f=pathology[pathology["cond"] ==0]
c_c=pathology[pathology["cond"] ==1][0:1190]
m_i=pathology[pathology["cond"] ==2][0:1190]

#Extract important metrics
atrial_f = a_f[["ae_m","ae_v","se_m","se_v","hfd_m","hfd_v","dfa_m","dfa_v","psd_m","psd_v"]]
congestive_h = c_c[["ae_m","ae_v","se_m","se_v","hfd_m","hfd_v","dfa_m","dfa_v","psd_m","psd_v"]]
myocardial_i = m_i[["ae_m","ae_v","se_m","se_v","hfd_m","hfd_v","dfa_m","dfa_v","psd_m","psd_v"]]

#Create target array for training
targets=a_f['cond'].tolist()+m_i['cond'].tolist()

#Create input array for training
X=pd.concat([atrial_f,myocardial_i],ignore_index=True)
X
#MainDF.shape

#Datmat = MainDF.copy()
#del Datmat['case']
#Datmat.dropna()
#Datmat.info()
# Transformations
#Datmat['cond'].astype(np.int32)
#Datmat['ae_m'].astype('float32')
#Data =np.float32(Datmat.to_numpy()).reshape(-13,15117)
#np.shape(Data)

# %%
import umap
import umap.plot
#from sklearn.datasets import load_digits

#digits = load_digits()

mapper = umap.UMAP().fit(X)
umap.plot.points(mapper, labels=tar_labels)
# %%
digits_df = pd.DataFrame(digits.data[:,1:11])
digits_df['digit'] = pd.Series(digits.target).map(lambda x: 'Digit {}'.format(x))
sns.pairplot(digits_df, hue='digit', palette='Spectral')
# %%
import tensorflow as tf
from tensorflow import keras
#convert the pandas object to a tensor

MainDummy.info()
MainDummy.shape
data = MainDummy.to_numpy()
#data=tf.convert_to_tensor(MainDummy)
#type(data)
# %%
mapper = umap.UMAP().fit(data)
#umap.plot.points(mapper, labels=digits.target)
# %%
# AREA UNDER THE CURVE
from sklearn import metrics
y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)

# %%
hist = np.histogram(MainDummy.iloc[0]['SampEn'])


# %% 
"""Holt-Winters (Triple Exponential Smoothing)"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing

fit = ExponentialSmoothing(data, seasonal_periods=periodicity, trend='add', seasonal='add').fit(use_boxcox=True)
fit.fittedvalues.plot(color='blue')
fit.forecast(5).plot(color='green')
plt.show()
# %% 
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

p = 5  # lag
d = 1  # difference order
q = 0  # size of moving average window

Y = np.array(hist[1]).astype('float32')

train, test = train_test_split(Y, test_size=0.20, shuffle=False)
history = train.tolist()
predictions = []

for t in range(len(test)):
	model = ARIMA(history, order=(p,d,q))
	fit = model.fit(disp=False)
	pred = fit.forecast()[0]
  
	predictions.append(pred)
	history.append(test[t])
  
print('MSE: %.3f' % mean_squared_error(test, predictions))

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
# %%
!