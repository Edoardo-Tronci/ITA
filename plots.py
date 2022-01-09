import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

which = 1

if which == 1:
    DATA = np.load('adult_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'Adult Dataset'
if which == 2:
    DATA = np.load('ailerons_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'Ailerons Dataset'
if which == 3:
    DATA = np.load('appliances_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'Appliances Energy Prediction Dataset'
if which == 4:
    DATA = np.load('arcene_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'Arcene Dataset'
if which == 5:
    DATA = np.load('blogfeed_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'BlogFeedback Dataset'
if which == 6:
    DATA = np.load('boston_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'Boston House Prices Dataset'
if which == 7:
    DATA = np.load('breast_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'Breast Cancer Wisconsin (Diagnostic) Dataset'
if which == 8:
    DATA = np.load('cifar_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'CIFAR 10 Dataset'
if which == 9:
    DATA = np.load('gisette_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'Gisette Dataset'
if which == 10:
    DATA = np.load('iris_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'Iris Dataset'
if which == 11:
    DATA = np.load('mnist_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'MNIST Handwritten Digit Dataset'
if which == 12:
    DATA = np.load('mv_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'Mv Dataset'
if which == 13:
    DATA = np.load('power_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'Power Consumption Dataset'
if which == 14:
    DATA = np.load('qsar_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'QSAR Oral Toxicity'
if which == 15:
    DATA = np.load('yearpred_10_100_res.npy',allow_pickle='TRUE').item()
    datasetname = 'YearPred Dataset'


TABF = DATA['TABF']
MAXIT = DATA['MAXIT']
nrnd = DATA['nrnd']

nit, nc = TABF.shape
for i in range(nrnd):
    mit = round(MAXIT[i])
    TABF[mit:,i] = TABF[mit-1,i]
    j = nrnd+i
    mit = round(MAXIT[j])
    TABF[mit:,j] = TABF[mit-1,j]

F1 = []
F2 = []

for i in range(nrnd):
    mit = round(MAXIT[i])
    #plt.plot(TABF[:mit,i],linestyle='dashed')
    F1.append(TABF[mit,i])
    mit = round(MAXIT[nrnd+i])
    #plt.plot(TABF[:mit,(nrnd+i)])
    F2.append(TABF[mit,nrnd+i])

MINV = np.zeros((nit,2))
MAXV = np.zeros((nit,2))
for i in range(nit):
    a = np.min(TABF[i,:nrnd])
    b = np.max(TABF[i,:nrnd])
    MINV[i,0] = a
    MAXV[i,0] = b
    a = np.min(TABF[i,nrnd:])
    b = np.max(TABF[i,nrnd:])
    MINV[i,1] = a
    MAXV[i,1] = b

plt.ion()

fig,ax = plt.subplots()
ax.fill_between(range(nit),MINV[:,0],MAXV[:,0],alpha=0.5,color='blue',label='standard')    
ax.fill_between(range(nit),MINV[:,1],MAXV[:,1],alpha=0.6,color='green',label='incremental')  
ax.legend()  
plt.yscale("log")
fig.suptitle(datasetname)

fig1,ax1 = plt.subplots()
plt.yscale("log")
bp = ax1.boxplot([F1,F2])
ax1.set_xticklabels(['stand.', 'incre.'],rotation=45, fontsize=8)
fig1.suptitle(datasetname)

fig.savefig("Plots/Loss_Adult.png", dpi = 300)
fig1.savefig("Plots/BoxPlot_Adult.png", dpi = 300)

pp = PdfPages("Results Adult.pdf")
pp.savefig(fig, dpi = 300, transparent = True)
pp.savefig(fig1, dpi = 300, transparent = True)
pp.close()