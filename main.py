import time
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.append('../Datasets')
sys.path.append('../ALA')

# import ALA_mon as ott
# import ALA_nonmon as ott
import ALA_nonmon_mon as ott

torch.set_default_dtype(torch.double)
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import torch.nn as nn

#########################################################################
# import train data from the Adult dataset
#########################################################################
from adult import X_train, y_train

#########################################################################
# import train data from the Ailerons dataset
#########################################################################
# from ailerons import X_train, y_train

#########################################################################
# import train data from the Appliances Energy Prediction dataset
#########################################################################
# from appliances import X_train, y_train

#########################################################################
# import train data from the Arcene dataset
#########################################################################
# from arcene import X_train, y_train

#########################################################################
# import train data from the BlogFeedback dataset
#########################################################################
#from blogfeed import X_train, y_train, teach

#########################################################################
# import train data from the Boston House Prices dataset
#########################################################################
# from boston_house import X_train, y_train

#########################################################################
# import train data from the Breast Cancer Wisconsin (Diagnostic) dataset
#########################################################################
# from breast_cancer import X_train, y_train

#########################################################################
# import train data for the CIFAR 10 dataset
#########################################################################
# from cifar10 import X_train, y_train

#########################################################################
# import train data for the Gisette dataset
#########################################################################
# from gisette import X_train, y_train

#########################################################################
# import train data for the Iris dataset
#########################################################################
# from iris import X_train, y_train

#########################################################################
# import train data for the MNIST Handwritten Digit dataset
#########################################################################
# from mnist import X_train, y_train

#########################################################################
# import train data for the Mv dataset
#########################################################################
# from mv import X_train, y_train

#########################################################################
# import train data for the QSAR dataset
#########################################################################
# from qsar import X_train, y_train

#########################################################################
# import train data for the Power Consumption dataset
#########################################################################
# from power import X_train, y_train

#########################################################################
# import train data for the YearPred dataset
#########################################################################
# from yearpred import X_train, y_train

X_train, y_train = map(torch.tensor, (X_train, y_train))

X_train = X_train.double()
y_train = y_train.double()
# y_train = y_train.view(-1, 1).long()

X_train = X_train.to(device)
y_train = y_train.to(device)

ntrain, input_dim = X_train.shape 
ntrain, out_dim = y_train.shape

# print(X_train.shape)
# print(y_train.shape)
# print(out_dim)

#Define NWTNM parameters

t = "Nash"  #Dembo
c = "Curv"  #NoCurv
name = "adult" #dataset

r = 1598711 #seed

torch.cuda.manual_seed(r)
torch.manual_seed(r)

    
class Net(nn.Module):

    def __init__(self, dims):
        super(Net, self).__init__()

        self.nhid = len(dims)
        self.dims = dims
        
        self.fc = nn.ModuleList().double()
        for i in range(self.nhid - 1):
            linlay = nn.Linear(dims[i], dims[i+1]).double().to(device)
            # linlay = nn.Linear(dims[i], dims[i+1], bias = False).double().to(device)
            self.fc.append(linlay)
                
    def forward(self, x):
        b = torch.tensor([], device = device, dtype=torch.double)
        
        for i in range(self.nhid - 2):
            x = self.fc[i](x)
            #x = torch.relu(x)
            # x = torch.sigmoid(x)
            x = torch.tanh(x)
            # x = swish(x)
            
        x = self.fc[self.nhid - 2](x)
        # x = torch.softmax(x, dim = 1)
        b = torch.cat((b,x))
        return b
                         
def swish(x):
    return torch.mul(x, sigmoid(x))

def sigmoid(x):
    return torch.where(x >= 0, _positive_sigm(x), _negative_sigm(x))

def _negative_sigm(x):
    expon = torch.exp(-x)
    return 1 / (1 + expon)

def _positive_sigm(x):
    expon = torch.exp(x)
    return expon / (1 + expon)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight.data,a=-1.0,b=1.0).double()
        torch.nn.init.uniform_(m.bias.data,a=-1.0,b=1.0).double()

def cross_entropy(y_hat, y):
    return torch.mean(- torch.log(y_hat[range(len(y_hat)), y.view(-1,)])).double()

MSELoss = torch.nn.MSELoss()

def my_loss(X, y):
    y_hat = net(X).double()
    loss = MSELoss(y_hat,y)
    # loss = cross_entropy(y_hat, y)
    return loss

def my_loss_reg(X, y, ro):
    y_hat = net(X).double()
    loss = MSELoss(y_hat,y)
    # loss = cross_entropy(y_hat, y)
    l2_reg = torch.tensor(0.0, device = device, dtype=torch.double)
    for param in net.parameters():
         l2_reg += torch.norm(param)**2
    loss += ro * l2_reg
    return loss

#################################
# Define the variable array for
# NWTNM optimizer
#################################
def set_param(net_in):
    net.fc[0].weight.data[0:nneu,:] = net_in.fc[0].weight.data
    net.fc[0].bias.data[0:nneu] = net_in.fc[0].bias.data
    torch.nn.init.uniform_(net.fc[0].weight.data[nneu:newneu,:],a=-1.0,b=1.0).double()
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
    torch.nn.init.normal_(x).double()

    return x.detach().to(device)

def set_x(x):
 	state_dict = net.state_dict()
 	i = 0
 	for k,v in state_dict.items():
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
    # l_train = my_loss_reg(X_train, y_train, l2_lambda)
    return l_train

def grad(x):
    for param in net.parameters():
        if param.requires_grad:
            if not type(param.grad) is type(None):
                param.grad.zero_()
            param.requires_grad_()
    f = funct(x)
    f.backward()

    if False:
        g = x.clone().detach()
        i = 0
        for v in net.parameters():
            if v.requires_grad:
                lpart = v.numel()
                d = v.grad.reshape(lpart)
                g[i:i+lpart] = d
                i += lpart

    views = []
    for p in net.parameters():
        if p.requires_grad:
            view = p.grad.view(-1)
        views.append(view)

    g1 = torch.cat(views, 0).to(device)

    return g1

def hessdir2(x,d):
 	if False:
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

'''
in hessdir3 a seconda del valore di goth:
 FALSE -> si calcola gradstore e lo si memorizza
 TRUE  -> si usa gradstore salvato senza ricalcolarlo
'''
def hessdir3(x,d,goth):
	for param in net.parameters():
		if param.requires_grad:
			if not type(param.grad) is type(None):
				param.grad.zero_()
			param.requires_grad_()
	if not goth:
		hessdir3.gradstore = torch.autograd.grad(outputs=funct(x), inputs=net.parameters(), create_graph=True)
	dot   = nn.utils.parameters_to_vector(hessdir3.gradstore).mul(d).sum()
	grads = [g.contiguous() for g in torch.autograd.grad(dot, net.parameters(), retain_graph = True)]
	return nn.utils.parameters_to_vector(grads)

# which_algo    = 'sgd'
# which_algo    = 'lbfgs'
which_algo    = 'troncato'

nneu_tot      = 100 
maxiter_tot   = 1000
maxtim        = 1800
l2_lambda     = 1e-05
nrnd          = 10
iprint        = 0  # -1
satura        = True
hd_exact      = True
TABF          = np.zeros((maxiter_tot+1,2*nrnd))
MAXIT         = np.zeros(2*nrnd)

# print()
# print("----------------------------------------------")
# print(" define a neural net to be minimized ")
# print("----------------------------------------------")
# print()

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
        # n_class = 10
        # dims      = [input_dim, hidden_1, n_class]
        dims      = [input_dim, nneu, out_dim]
        net       = Net(dims).double().to(device)
        net.apply(init_weights)

        for i in range(10):
            n = dim()
            x = startp(n)
            set_x(x)  

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
                    # loss1 = my_loss_reg(X_train, y_train, l2_lambda)
                    ng += 1
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
                    # loss1 = my_loss_reg(X_train, y_train, l2_lambda)
                    pbar.update(1)
                    if niter_tot+ni < maxiter_tot+1:
                        TABF[niter_tot+ni,(imeth-1)*nrnd+irnd] = loss1.item()
                    loss1.backward()
                    return loss1

                if which_algo == 'lbfgs':
                    timelbfgs = time.time()
                    optimizer = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=maxiter, max_eval=None, tolerance_grad=tol, 
                                                  tolerance_change=tolch, history_size=10, line_search_fn="strong_wolfe")

                    optimizer.step(closure)
                    niter = optimizer.state_dict()['state'][0]['n_iter']
                    timelbfgs_tot = time.time() - timelbfgs
                    timeparz = timelbfgs_tot
                elif which_algo == 'troncato': 
                    ott.n_iter = 0
                    f_0 = funct(x)
                    #fstar,xstar,niter,nf,ng,nneg,timeparz = ott.NWTNM(fun_closure,grad,hessdir3,x,tol,maxiter,maxtim,iprint,satura,hd_exact)
                    fstar,xstar,niter,nf,ng,nneg,timeparz = ott.NWTNM(funct,grad,hessdir3,x,tol,maxiter,maxtim,iprint,satura,hd_exact,name,r,nneu,c,t,f_0)
            
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
np.save('adult_10_100_res.npy', DATA) 