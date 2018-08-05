"Extended isolated forest functions"
__author__ = 'Matias Carrasco Kind '
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import random as rn
import pandas as pd
from pandas import Series, DataFrame
import csv
import eif_vnif as iso
import eif as iso_eif
import iso_forest as iso_if
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D

#training data
X0 = np.array(pd.read_csv('idiqvd.csv'))  
X0_norm0 = (X0.T[0]-X0.T[0].min())/(X0.T[0].max()-X0.T[0].min()) #normalization
X0_norm1 = (X0.T[1]-X0.T[1].min())/(X0.T[1].max()-X0.T[1].min()) #normalization
X0_norm2 = (X0.T[2]-X0.T[2].min())/(X0.T[2].max()-X0.T[2].min()) #normalization
X0_Norm = np.c_[X0_norm0,X0_norm1,X0_norm2]
#training data finished

#predicting data
X2 = np.array(pd.read_csv('idiqvdyqs.csv'))  
X2_norm0 = (X2.T[0]-X0.T[0].min())/(X0.T[0].max()-X0.T[0].min()) #normalization
X2_norm1 = (X2.T[1]-X0.T[1].min())/(X0.T[1].max()-X0.T[1].min()) #normalization
X2_norm2 = (X2.T[2]-X0.T[2].min())/(X0.T[2].max()-X0.T[2].min()) #normalization
X2_Norm = np.c_[X2_norm0,X2_norm1,X2_norm2]
#predicting data finished

#setting
Nobjs = X0.shape[0]
Nobjs1 = X2.shape[0]
ntrees=50
sample = 256
CT=[]
CT_if = []
S = np.zeros(Nobjs1)
S_if = np.zeros(Nobjs1)
score_outlier = np.zeros(50)
score_normal = np.zeros(50)
c = iso.c_factor(sample)
#setting finished

# VNIF: 
for i in range(ntrees): #used for comparison between normal points and outliers
    ix = rn.sample(range(Nobjs),sample)
    X_p = X0_Norm[ix]  
    limit = 10
    C=iso.iTree(X_p,0,limit,exlevel=2)
    CT.append(C)
# VNIF scoring
F1 = iso.iForest(X0_Norm,ntrees=50, sample_size=sample, ExtensionLevel=2)
S1 = F1.compute_paths(X_in=X2_Norm)
# VNIF finished

#eif
F1_eif = iso_eif.iForest(X0_Norm,ntrees=50, sample_size=sample, ExtensionLevel=2)
S1_eif = F1_eif.compute_paths(X_in=X2_Norm)
#eif finished

#if     
for i in range(ntrees):
    ix = rn.sample(range(Nobjs),sample)
    X_p = X0_Norm[ix]
    
    limit = 10
    C_if=iso_if.iTree(X_p,0,limit)
    CT_if.append(C_if)
for i in range(Nobjs):
    h_temp = 0
    for j in range(ntrees):
        h_temp += iso_if.PathFactor(X2_Norm[i],CT_if[j]).path*1.0
    Eh = h_temp/ntrees
    S_if[i] = 2.0**(-Eh/c)
#if finised

#used for comparison between normal points and outliers
f = plt.figure(figsize=(12,6))
for j in range(ntrees):
    score_outlier[j] = iso.PathFactor(X2_Norm[1],CT[j]).path*1.0
plt.plot(score_outlier, marker = 'o',markerfacecolor= 'red')
plt.ylim([-30,30])
plt.title('Path length of an abnormal point: Mean={0:.3f}, Var={1:.3f}'.format(np.mean(score_outlier),np.var(score_outlier),))
plt.show
f = plt.figure(figsize=(12,6))
for j in range(ntrees):
    score_normal[j] = iso.PathFactor(X2_Norm[20],CT[j]).path*1.0
plt.plot(score_normal, marker = 'o',markerfacecolor= 'red')
plt.ylim([-30,30])
plt.title('Path length of a normal point: Mean={0:.3f}, Var={1:.3f}'.format(np.mean(score_normal),np.var(score_normal),))
plt.show()
#used for comparison between normal points and outliers: finished

#used for VNIF plotting
new_data1 = pd.concat([DataFrame(X2),DataFrame(S1)],axis =1)
new_data1.columns = ['id','iq','vd','score']
new_data1.to_csv('data_and_S1_eif_vnif.csv')
new_data_sort = new_data1.sort_values(by = 'score')
xx = new_data_sort['id']
yy = new_data_sort['iq']
zz = new_data_sort['vd']
#used for VNIF plotting finished

#used for eif plotting 
new_data1_eif = pd.concat([DataFrame(X2),DataFrame(S1_eif)],axis =1)
new_data1_eif.columns = ['id','iq','vd','scoreeif']
new_data1_eif.to_csv('data_and_S1_eif.csv')
new_data_sort_eif = new_data1_eif.sort_values(by = 'scoreeif')
xx_eif = new_data_sort_eif['id']
yy_eif = new_data_sort_eif['iq']
zz_eif = new_data_sort_eif['vd']
#used for eif plotting finished

#used for if plotting 
new_data1_if = pd.concat([DataFrame(X2),DataFrame(S_if)],axis =1)
new_data1_if.columns = ['id','iq','vd','score_if']
new_data1_if.to_csv('data_and_S1_if.csv')
new_data_sort_if = new_data1_if.sort_values(by = 'score_if')
xx_if = new_data_sort_if['id']
yy_if = new_data_sort_if['iq']
zz_if = new_data_sort_if['vd']
#used for if plotting finished

# plotting VNIF
fig=plt.figure(figsize=(15,5))
ax0 = fig.add_subplot(131, projection='3d')
ax0.scatter(xx[:-13],yy[:-13],zz[:-13], s = 15, color='green')
plt.title('Normal points')
ax1 = fig.add_subplot(132,projection = '3d')
ax1.scatter(xx,yy,zz , s = 15, color='#000000')
plt.title('With Outliers')
ax2 = fig.add_subplot(133,projection = '3d')
ax2.scatter(xx[-13:],yy[-13:],zz[-13:], s = 55, color='#000000')
ax2.scatter(xx[:-13],yy[:-13],zz[:-13], s= 15, color='#FF8000')
plt.title('VNIF(black:outliers && yellow:normal points)')
plt.show()

# plotting eif
fig=plt.figure(figsize=(15,5))
ax0_1 = fig.add_subplot(131, projection='3d')
ax0_1.scatter(xx[:-13],yy[:-13],zz[:-13], s = 15, color='green')
plt.title('Normal points')
ax1_1 = fig.add_subplot(132,projection = '3d')
ax1_1.scatter(xx_eif,yy_eif,zz_eif , s = 15, color='#000000')
plt.title('With Outliers')
ax2_1 = fig.add_subplot(133,projection = '3d')
ax2_1.scatter(xx_eif[-13:],yy_eif[-13:],zz_eif[-13:], s = 55, color='#000000')
ax2_1.scatter(xx_eif[:-13],yy_eif[:-13],zz_eif[:-13], s= 15, color='#FF8000')
plt.title('eif (black:outliers && yellow:normal points)')
plt.show()

# plotting if
fig=plt.figure(figsize=(15,5))
ax0_2 = fig.add_subplot(131, projection='3d')
ax0_2.scatter(xx[:-13],yy[:-13],zz[:-13], s = 15, color='green')
plt.title('Normal points')
ax1_2 = fig.add_subplot(132,projection = '3d')
ax1_2.scatter(xx_if,yy_if,zz_if , s = 15, color='#000000')
plt.title('With Outliers')
ax2_2 = fig.add_subplot(133,projection = '3d')
ax2_2.scatter(xx_if[-13:],yy_if[-13:],zz_if[-13:], s = 55, color='#000000')
ax2_2.scatter(xx_if[:-13],yy_if[:-13],zz_if[:-13], s= 15, color='#FF8000')
plt.title('if (black:outliers && yellow:normal points)')
plt.show()