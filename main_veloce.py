from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

from tqdm import tqdm
import datetime
import time
import math
import numpy as np
import sys
sys.path.append('../Datasets')
import troncato_veloce as ott
import torch
torch.set_default_dtype(torch.double)

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

############################################################
# import train data del dataset di Terna
############################################################
from adult import X_train, y_train

############################################################
# import train data from the MNIST 784 (zalando) dataset
############################################################
# from ailerons import X_train, y_train

############################################################
# import train data from the MNIST dataset
############################################################
# from appliances import X_train, y_train

############################################################
# import train data from the YearPredictionMSD dataset
############################################################
# from arcene import X_train, y_train

############################################################
# import train data from the teacher dataset
############################################################
#from blogfeed import X_train, y_train, teach

############################################################
# import train data from the iris dataset
############################################################
# from boston_house import X_train, y_train

############################################################
# import train data from the breast cancer dataset
############################################################
# from breast_cancer import X_train, y_train

############################################################
# import train data for the SEATTLE HOUSE SALES dataset
############################################################
# from cifar10 import X_train, y_train

############################################################
# import train data for the MV dataset
############################################################
# from gisette import X_train, y_train

############################################################
# import train data for the ailerons dataset
############################################################
# from iris import X_train, y_train

############################################################
# import train data for the blogdata dataset
############################################################
# from mnist import X_train, y_train

############################################################
# import train data for the arcene dataset
############################################################
# from mv import X_train, y_train

############################################################
# import train data for the gisette dataset
############################################################
# from qsar import X_train, y_train

############################################################
# import train data for the adult dataset
############################################################
# from yearpred import X_train, y_train

X_train, y_train = map(torch.tensor, (X_train, y_train))
X_train = X_train.double()
y_train = y_train.double()

X_train = X_train.to(device)
y_train = y_train.to(device)

ntrain, input_dim = X_train.shape 
ntrain, out_dim = y_train.shape
#out_dim = 1

print(X_train.shape)
print(y_train.shape)
print(out_dim)

dims = [input_dim, 100, out_dim]

nhid = len(dims)

c = 1

seed = 1598711

torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

    
class Net(nn.Module):

    def __init__(self, dims):
        super(Net, self).__init__()

        self.nhid = len(dims)
        self.dims = dims
        
        self.fc = nn.ModuleList().double()
        for i in range(self.nhid - 1):
            linlay = nn.Linear(dims[i], dims[i+1]).double().to(device)
            self.fc.append(linlay)
                
    def forward(self, x):
        b = torch.tensor([], device = device, dtype=torch.double)
        
        for i in range(self.nhid - 2):
            x = self.fc[i](x)
            #x = torch.relu(x)
            #x = torch.sigmoid(x)
            x = torch.tanh(x)
            
        x = self.fc[self.nhid - 2](x)
        #uncomment line below for multiclass
        #if out_dim > 1:
        #    x = torch.softmax(x, dim = 1)
        #x = torch.sigmoid(x)
        b = torch.cat((b,x))
        return b
                         
def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.normal_(m.weight.data).double()
        # torch.nn.init.zeros_(m.bias.data).double()
        # torch.nn.init.normal_(m.bias.data).double()
        torch.nn.init.uniform_(m.weight.data,a=-1.0,b=1.0).double()
        torch.nn.init.uniform_(m.bias.data,  a=-1.0,b=1.0).double()
        
def cross_entropy(y_hat, y):
    print(y)
    print(range(len(y_hat)))
    return torch.sum(- torch.log(y_hat[range(len(y_hat)), y]).double(),dim=0)
    #return 0
	
def mse(y_hat, y):
	return torch.pow(y_hat-y,2)

if out_dim >= 1:
    loss = torch.nn.MSELoss()
else:
    #uncomment line below for multiclass problem
    loss = torch.nn.functional.binary_cross_entropy

def my_loss(X, y):
    y_hat = net(X).double()
    ##print('my_loss: y_hat=',y_hat)
    ##l = cross_entropy(y_hat, y.view(-1,)).view(-1, 1).double()
    ##return torch.sum(l, dim = 0)
    #l = mse(y_hat, y).double()
    #return torch.sum(l)/len(y_hat)
    return loss(y_hat,y)

#net  = Net(dims).double().to(device)
#print(net)
#net.apply(init_weights).double()

#################################
# define the variable array for
# NWTNM optimizer
#################################
def set_param(net_in):
    net.fc[0].weight.data[0:nneu,:] = net_in.fc[0].weight.data
    net.fc[0].bias.data[0:nneu] = net_in.fc[0].bias.data
    torch.nn.init.uniform_(net.fc[0].weight.data[nneu:newneu,:],a=-1.0,b=1.0).double()
    #net.fc[0].weight.data[nneu:newneu,:] = torch.zeros((deltan,input_dim)).double().to(device)
    net.fc[1].weight.data[:,nneu:newneu] = torch.zeros((out_dim,newneu-nneu)).double().to(device)
    net.fc[1].weight.data[:,0:nneu] = net_in.fc[1].weight.data
    net.fc[1].bias.data = net_in.fc[1].bias.data

def dim():
	n = 0
	for k,v in net.state_dict().items():
		n += v.numel()
	return n

def startp(n1):
    x = torch.zeros(n1,dtype=torch.double,requires_grad=True)
    #seed = 4726393
    #torch.cuda.manual_seed(seed)
    #torch.manual_seed(seed)
    torch.nn.init.normal_(x).double()

    #i = 0
    #for k,v in net.state_dict().items():
    #	lpart = v.numel()
    #	d = v.reshape(lpart).double()
    #	#print(i,lpart,d.shape)
    #	with torch.no_grad():
    #		x[i:i+lpart] = d
    #	i += lpart
    return x.detach().to(device)

def set_x(x):
	state_dict = net.state_dict()
	i = 0
	for k,v in state_dict.items():
		#print('set_x: ',k[-4:])
		lpart = v.numel()
		x[i:i+lpart] = state_dict[k].reshape(lpart).double()
		i += lpart
        
def save_point(n1):
    state_dict = net.state_dict()
    x = torch.zeros(n1,dtype=torch.double,requires_grad=False)
    i = 0
    #first save the biases
    for k,v in state_dict.items():
        if k[-4:] == 'bias':
            lpart=v.numel()
            x[i:i+lpart] = -state_dict[k].reshape(lpart).double()
            i += lpart
    #then save the weights
    for k,v in state_dict.items():
        if k[-4:] == 'ight':
            lpart=v.numel()
            x[i:i+lpart] = state_dict[k].T.reshape(lpart).double()
            i += lpart    
    x0 = x.detach().numpy()
    np.savetxt("x0.txt",x0,delimiter="\t")

def funct(x):
    state_dict = net.state_dict()
    i = 0
    for k,v in state_dict.items():
        lpart = v.numel()
        state_dict[k] = x[i:i+lpart].reshape(v.shape).double()
        i += lpart
    net.load_state_dict(state_dict)
    l_train = my_loss(X_train, y_train)
    return l_train

def grad(x):
    for param in net.parameters():
        if param.requires_grad:
            if not type(param.grad) is type(None):
                param.grad.zero_()
            param.requires_grad_()
    #print('grad: x.grad=',x.grad)
    #print('grad: x=',x)
    f = funct(x)
    f.backward()

    if False:
        g = x.clone().detach()
        i = 0
        #grad_norm = 0.0
        for v in net.parameters():
            if v.requires_grad:
                lpart = v.numel()
                d = v.grad.reshape(lpart)
                #print(i,lpart,d.shape)
                #print(d)
                g[i:i+lpart] = d
                #grad_norm += d.norm().item()
                i += lpart
        #print(grad_norm)
    
    views = []
    for p in net.parameters():
        #if p.grad is None:
        #    view = p.new(p.numel()).zero_()
        #elif p.grad.is_sparse:
        #    view = p.grad.to_dense().view(-1)
        #else:
        #    view = p.grad.view(-1)
        if p.requires_grad:
            view = p.grad.view(-1)
        views.append(view)

    g1 = torch.cat(views, 0).to(device)
    #print('grad: ',g1)
    #print(g1.norm().item(),g.norm().item())
    
    return g1

def hessdir2(x,d):
    state_dict = net.state_dict()
    i = 0
    for k,v in state_dict.items():
        lpart = v.numel()
        state_dict[k] = x[i:i+lpart].reshape(v.shape).double()
        i += lpart
    net.load_state_dict(state_dict)
    for param in net.parameters():
        if param.requires_grad:
            if not type(param.grad) is type(None):
                param.grad.zero_()
            param.requires_grad_()
    grads = torch.autograd.grad(outputs=funct(x), inputs=net.parameters(), create_graph=True)
    dot   = nn.utils.parameters_to_vector(grads).mul(d).sum()
    grads = [g.contiguous() for g in torch.autograd.grad(dot, net.parameters(), retain_graph = True)]
    return nn.utils.parameters_to_vector(grads)

print()
print("----------------------------------------------")
print(" define a neural net to be minimized ")
print("----------------------------------------------")
print()
which_algo    = 'sgd'
which_algo    = 'lbfgs'
which_algo    = 'troncato'
nneu_tot      = 100 
maxiter_tot   = 1000
nrnd          = 10
TABF          = np.zeros((maxiter_tot+1,2*nrnd))
MAXIT         = np.zeros(2*nrnd)

# DATA = np.load('blogdata_100_20_res.npy',allow_pickle='TRUE').item()
# TABF = DATA['TABF']
# MAXIT = DATA['MAXIT']

for imeth in [1, 2]:
    tolmax    = 1.e-6
    tolchmax  = 1.e-9
    outlev    = 0

    for irnd in range(nrnd):
        niter_tot = 0
        time_tot  = 0
        if imeth == 1:
            maxiter = maxiter_tot
            nneu    = nneu_tot
        else:
            maxiter   = 200
            nneu      = 20 
            maxiter   = nneu*10
        satura    = False
        dims      = [input_dim, nneu, out_dim]
        net       = Net(dims).double().to(device)
        net.apply(init_weights)
        #net.load_state_dict(teach.state_dict())

        y_hat = net(X_train).double()

        print('mio: ',torch.pow(y_hat-y_train,2).sum()/ntrain)
        y1 = y_train.to(torch.device("cpu")).detach().numpy()
        y2 = y_hat.to(torch.device("cpu")).detach().numpy()
        print('mse: ',mean_squared_error(y1,y2))
        print(len(y_hat))

        print(net)
        print(net.parameters())
        if False:
            for p in net.parameters():
                print('------------------------------')
                print(p.data)

            print('hit return to continue ...')
            input()

        for i in range(10):
            n = dim()
            x = startp(n)
            set_x(x)
            #save_point(n)
            #print("x.shape = ",x.shape)
            #print("x: ",x)

            #print('hit return to continue...')
            #input()
            
            if outlev > 0:
                print('Try torch.optim.LBFGS first ....')   

            l_train = funct(x)
            nabla_l_train = grad(x)
            gnorm = nabla_l_train.norm().item()
            print("numero di parametri totali: n=",n," neuroni: ",nneu," loss:",l_train.item()," gloss:",gnorm)

            if i == 0:
                tol = 1.e-1*gnorm 
            else:
                tol = 1.e-1
            #tolch = tolchmax
            #tol = tolmax
            tolch = 1.e-3*tol
            #tol = 1.e-1*(np.minimum(gnorm,1.0))**2
            if nneu >= nneu_tot:
                satura = True
                maxiter= maxiter_tot-niter_tot
                tol    = tolmax
                tolch  = 1.e-1*tol
            else:
                maxiter= nneu*10
                
            with tqdm(total=maxiter) as pbar:
                ng = 0
                ni = 0
                def fun_closure(x):
                    global ni
                    deltai = ott.n_iter - ni
                    pbar.update(deltai)
                    ni = ott.n_iter
                    l_train = funct(x)
                    if ni < maxiter_tot+1:
                        if ni > 0:
                            if l_train.item() < TABF[ni-1,(imeth-1)*nrnd+irnd]:
                                TABF[ni,(imeth-1)*nrnd+irnd] = l_train.item()
                        else:
                            TABF[ni,(imeth-1)*nrnd+irnd] = l_train.item()
                    return l_train
                    
                def closure():
                    global ng
                    global ni
                    optimizer.zero_grad()
                    loss1 = my_loss(X_train, y_train)
                    ng += 1
                    #print(loss1,' ',ng)
                    deltai = optimizer.state_dict()['state'][0]['n_iter'] - ni
                    pbar.update(deltai)
                    ni = optimizer.state_dict()['state'][0]['n_iter']
                    if niter_tot+ni < maxiter_tot+1:
                        TABF[niter_tot+ni,(imeth-1)*nrnd+irnd] = loss1.item()
                    loss1.backward()
                    return loss1

                def closure_sgd(ni):
                    optimizer.zero_grad()
                    loss1 = my_loss(X_train, y_train)
                    pbar.update(1)
                    if niter_tot+ni < maxiter_tot+1:
                        TABF[niter_tot+ni,(imeth-1)*nrnd+irnd] = loss1.item()
                    loss1.backward()
                    return loss1

                if which_algo == 'lbfgs':
                    timelbfgs = time.time()
                    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
                    optimizer = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=maxiter, max_eval=None, tolerance_grad=tol, 
                                                  tolerance_change=tolch, history_size=10, line_search_fn="strong_wolfe")

                    optimizer.step(closure)
                    niter = optimizer.state_dict()['state'][0]['n_iter']
                    timelbfgs_tot = time.time() - timelbfgs
                    timeparz = timelbfgs_tot
                elif which_algo == 'troncato': 
                    ott.n_iter = 0
                    #fstar,xstar,niter,timeparz = ott.NWTNM(fun_closure,grad,hessdir2,x,tol,maxiter,-1,True)
                    fstar,xstar,niter,timeparz = ott.NWTNM(funct,grad,hessdir2,x,tol,maxiter,0,True)
            
                elif which_algo == 'sgd':
                    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

                    timesgd = time.time()
                    niter = maxiter
                    for it in range(0, niter):
                        closure_sgd(it)
                        optimizer.step()
                    timeparz = time.time() - timesgd
                    
            pbar.close()
            if outlev > 0:
                print('.... done', timeparz)
                
            time_tot += timeparz
            niter_tot += niter
            if (niter_tot > maxiter_tot) or satura:
                break
            if outlev > 0:
                print()
                print("----------------------------------------------")
                print(" define a bigger neural net to be minimized ")
                print("----------------------------------------------")
                print()
                print("old loss=",my_loss(X_train, y_train))
            
            net_copy = Net(dims).double().to(device)
            net_copy.load_state_dict(net.state_dict())
            
            deltan = 200
            newneu = nneu + deltan
            newneu = min(2*nneu,nneu_tot)
            
            dims = [input_dim, newneu, out_dim]
            net  = Net(dims).double().to(device)
            #print(net)
            
            set_param(net_copy)
            
            if outlev > 0:
                print("new loss=",my_loss(X_train, y_train))

            nneu = newneu

        print(niter_tot,time_tot,my_loss(X_train, y_train).item())
        MAXIT[(imeth-1)*nrnd+irnd] = niter_tot

DATA = {'TABF': TABF, 'MAXIT': MAXIT, 'nrnd': nrnd}
# np.save('yearpred_10_100_res.npy', DATA) 

F1 = []
F2 = []

fig_loss = plt.figure()
for i in range(nrnd):
    mit = round(MAXIT[i])
    plt.plot(TABF[:mit,i],linestyle='dashed')
    F1.append(TABF[mit,i])
    mit = round(MAXIT[nrnd+i])
    plt.plot(TABF[:mit,(nrnd+i)])
    F2.append(TABF[mit,nrnd+i])
#plt.plot(TABF[:maxiter_tot,:4],linestyle='dashed')
#plt.plot(TABF[:maxiter_tot,4:])
plt.yscale("log")

fig_boxplot = plt.figure()
plt.boxplot([F1,F2])

# pp = PdfPages("Risultati YEARPRED/Results YearPred 10-%i neurons.pdf" %nneu_tot)
# pp.savefig(fig_loss, dpi = 300, transparent = True)
# pp.savefig(fig_boxplot, dpi = 300, transparent = True)
# pp.close()