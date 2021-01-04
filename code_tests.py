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
umap.plot.points(mapper, labels=targets)
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