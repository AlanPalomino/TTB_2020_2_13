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
from main import MainDF
# %%
""" ANÁLISIS ESPECTRAL USANDO WAVELETS"""
def WaveletPowerSpectrum():
        """
        docstring
        """
        
        from __future__ import division
        import numpy
        from matplotlib import pyplot
        import pycwt as wavelet
        from pycwt.helpers import find

        # Definir parámetros de la señal

        dat = numpy.array(MainDummy.iloc[40]['rr'])
        record = str(MainDummy.iloc[40]['record'])
        cond = str(MainDummy.iloc[40]['conditon'])
        #url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
        #dat = numpy.genfromtxt(url, skip_header=19)
        title = 'Señal de HRV_record ['+ record +'] Condition: ['+cond+ ']'
        label = 'HRV'
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
        dx.set_title('d) {}--{} seconds scale-averaged power'.format(2, 8))
        dx.set_xlabel('Time (s)')
        dx.set_ylabel(r'Average variance [{}]'.format(units))
        ax.set_xlim([t.min(), t.max()])

        pyplot.show()

# %%
data = MainDF.sample(frac=1.0)
#Clean dataset
pathology = data[['record','cond_id', 'ae_mean',
       'ae_variance', 'ae_skewness', 'ae_spectral_entropy', 'se_mean',
       'se_variance', 'se_skewness', 'se_spectral_entropy', 'hfd_mean',
       'hfd_variance', 'hfd_skewness', 'hfd_spectral_entropy', 'dfa_mean',
       'dfa_variance', 'dfa_skewness', 'dfa_spectral_entropy', 'psd_mean',
       'psd_variance', 'psd_skewness', 'psd_spectral_entropy']]
#pathology
pathology = pathology.dropna()
tar_labels =  data['condition']
# Define train set and targets
#Group by pathology
a_f = pathology[pathology["cond_id"] ==0]
c_c = pathology[pathology["cond_id"] ==1][0:1190]
m_i = pathology[pathology["cond_id"] ==2][0:1190]

#Extract important metrics
atrial_f = a_f[['record', 'cond_id', 'ae_mean', 'ae_variance', 'ae_skewness',
       'ae_spectral_entropy', 'se_mean', 'se_variance', 'se_skewness',
       'se_spectral_entropy', 'hfd_mean', 'hfd_variance', 'hfd_skewness',
       'hfd_spectral_entropy', 'dfa_mean', 'dfa_variance', 'dfa_skewness',
       'dfa_spectral_entropy', 'psd_mean', 'psd_variance', 'psd_skewness',
       'psd_spectral_entropy']]
congestive_h = c_c[['record', 'cond_id',  'ae_mean', 'ae_variance', 'ae_skewness',
       'ae_spectral_entropy', 'se_mean', 'se_variance', 'se_skewness',
       'se_spectral_entropy', 'hfd_mean', 'hfd_variance', 'hfd_skewness',
       'hfd_spectral_entropy', 'dfa_mean', 'dfa_variance', 'dfa_skewness',
       'dfa_spectral_entropy', 'psd_mean', 'psd_variance', 'psd_skewness',
       'psd_spectral_entropy']]
myocardial_i = m_i[['record', 'cond_id',  'ae_mean', 'ae_variance', 'ae_skewness',
       'ae_spectral_entropy', 'se_mean', 'se_variance', 'se_skewness',
       'se_spectral_entropy', 'hfd_mean', 'hfd_variance', 'hfd_skewness',
       'hfd_spectral_entropy', 'dfa_mean', 'dfa_variance', 'dfa_skewness',
       'dfa_spectral_entropy', 'psd_mean', 'psd_variance', 'psd_skewness',
       'psd_spectral_entropy']]

#Create target array for training
targets=a_f['cond_id'].tolist()+m_i['cond_id'].tolist()+ c_c['cond_id'].tolist()

#Create input array for training0
X=pd.concat([atrial_f,myocardial_i, congestive_h ],ignore_index=True)
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
#================= UMAP =====================================#
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import umap
import umap.plot

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

reducer = umap.UMAP()
# Scale Data
get_data = X.values
scaled_data = StandardScaler().fit_transform(get_data)
print('Forma de Datos escalados: {}'.format(scaled_data.shape))
# Reduce data
embedding = reducer.fit_transform(scaled_data)
print('Forma de Datos reducidos: {}'.format(embedding.shape))

colors = np.array().map(["atrial_fibrilation", "myocardial_infarction", "congestive_heartfailure"])
# %% 
# =============== Run only in server!!!!  ============================

sns.pairplot(pathology, hue='cond_id')
# %%
#colors = pd.Series(tar_labels).map({"atrial_fibrilation":0, "myocardial_infarction":2, "congestive_heartfailure":1})

plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(5)-0.5).set_ticks(np.arange(3))
#plt.title('UMAP projection of the Digits dataset', fontsize=24);
plt.title('Proyección de UMAP de Patologías', fontsize=24)
#embedding.shape
# %%
#================ AREA UNDER THE CURVE =======================================
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
#================= ARIMA =====================================# 
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
#================= PCA =====================================#
# %%
#================= LDA =====================================#
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


