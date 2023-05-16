#%%
from tt.core.vector import vector
import tt

#%%
from __future__ import print_function, absolute_import, division
import sys
sys.path.append('../')
import numpy as np
import tt
import random
import scipy.stats as si
#import sympy as sy
#from sympy.stats import Normal, cdf
#import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal
import time

import matplotlib.pyplot as plt
from tt.amen import amen_solve
#import matplotlib.pyplot as plt


import tt.cross.rectcross as rect_cross
import scipy as sci

from numpy.linalg import inv
#import matplotlib.animation as animation
#import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import tt.optimize.tt_min  as tt_min

#%%
#  !pip install git+https://github.com/oseledets/ttpy

#%%

d = 5
n = 10
b = 1E3
#x = np.arange(n)
#x = np.reshape(x, [2] * d, order = 'F')
#x = tt.tensor(x, 1e-12)
x = tt.xfun(n, d)
#x = tt.ones(n, d)

#sf = lambda x : x[0]+x[1]+x[2]+x[3]+x[4] #Should be rank 2

#y = tt.multifuncrs([x,x,x,x,x], sf, 1e-6, y0=tt.ones(n, d))
#y1 = tt.tensor(sf(x.full()), 1e-8)

#print("pi / 2 ~ ", tt.dot(y, tt.ones(2, d)) * h)
#print (y - y1).norm() / y.norm()
# %%
x.full().shape
# %%
xfull=x = tt.xfun(5, 2).full()
# %%
xfull
# %%
sf = lambda x : x[0]+x[1]+x[2]+x[3]+x[4] #Should be rank 2
# %%
sf([1,2,3,4,5])
# %%
y = tt.multifuncrs(x, sf, 1e-6, y0=tt.ones(n, d))
# %%
a = tt.tensor(np.random.rand (3, 3),1e-4)
b = tt.tensor(np.random.rand (3, 3),1e-4)
c = tt.multifuncrs([a, b], lambda x: np.sum(x, axis=1), eps=1E-6)

# %%
a
# %%
a
# %%
a.full()
# %%
b
# %%
c.full()
# %%
a.full()+b.full()-c.full()
# %%
#from __future__ import print_function, absolute_import, division
#import numpy as np
#import tt


#%%
#Tested function
def myfun(x):
    return np.sin((x.sum(axis=1))) #** 2
    #return 1.0 / ((x.sum(axis=1)) + 1e-3)

    #return (x + 1).prod(axis=1)
    #return np.ones(x.shape[0])

def sumFun(x):
    thesum=x.sum(axis=1)
    print("x=",x.shape,x,thesum)
    return thesum

d = 3
n = 5
r = 2

#sz = [n] * d
#ind_all = np.unravel_index(np.arange(n ** d), sz)
#ind_all = np.array(ind_all).T
#ft = reshape(myfun(ind_all), sz)
#xall = tt.tensor(ft, 1e-8)
#x0 = tt.tensor(ft, 1e-8)


x0 = tt.rand(n, d, r)

x1 = rect_cross.cross(sumFun, x0, nswp=5, kickrank=1, rf=2)

# %%
x1
#%%
def unit(n, d=None, j=None, tt_instance=True):
    ''' Generates e_j _vector in tt.vector format
    ---------
    Parameters:
        n - modes (either integer or array)
        d - dimensionality (integer)
        j - position of 1 in full-format e_j (integer)
        tt_instance - if True, returns tt.vector;
                      if False, returns tt cores as a list
    '''
    if isinstance(n, int):
        if d is None:
            d = 1
        n = n * np.ones(d, dtype=np.int32)
    else:
        d = len(n)
    if j is None:
        j = 0
    rv = []

    j = ind2sub(n, j)

    for k in range(d):
        rv.append(np.zeros((1, n[k], 1)))
        rv[-1][0, j[k], 0] = 1
    if tt_instance:
        rv = tt.vector.from_list(rv)
    return rv


def ind2sub(siz, idx):
    '''
    Translates full-format index into tt.vector one's.
    ----------
    Parameters:
        siz - tt.vector modes
        idx - full-vector index
    Note: not vectorized.
    '''
    n = len(siz)
    subs = np.empty((n))
    k = np.cumprod(siz[:-1])
    k = np.concatenate((np.ones(1), k))
    for i in range(n - 1, -1, -1):
        subs[i] = np.floor(idx / k[i])
        idx = idx % k[i]
    return subs.astype(np.int32)

def lin_ind(n0,ind):
    d=len(n0)
    ni = float(n0[0])
    ii=ind[0]
    for i in range(1, d):
        ii+=ind[i]*ni
        ni *= n0[i]
    return ii

def invInd(j,n):
    h1=ind2sub(n,j)
    #print("h1",h1)
    h1=h1[::-1]
    #print("h1r",h1)
    return lin_ind(n,h1)


t=(1,0,2)
print (lin_ind([2,3,5],t))
print(tt.xfun([2,3,5]).full()[t])
print(tt.xfun([2,3,5]).full())

print(ind2sub(d*[2],5))


#%%
##
def bs1(a):
    S,K,T,r,sigma = a
    return bs(S,K,T,r,0,sigma,'call')

def bs(S, K, T, r, q, sigma, option = 'call'):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: rate of continuous dividend paying asset 
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))
        
    return result


Strikes = np.arange(80,100,5)
Ts = np.arange(0.1,1,0.1)
Ss = np.arange(80,100,5)
sigmas = np.arange(0.2,0.7,0.1)
rs = np.arange(0.01,.05,0.01)

A = [Ss,Strikes,Ts,rs,sigmas]

shape = [len(x) for x in A]

def index_to_var(x):
    s_i,k_i,t_i,r_i,sig_i = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4]
    Aa=np.array([A[0][s_i],A[1][k_i],A[2][t_i],A[3][r_i],A[4][sig_i]]).T
    return Aa

ind1=np.array([[0,0,0,0,0],[1,0,0,1,0]])
index_test= index_to_var(ind1)
print()
def testf(a):
    print("a=",a)
    return a[3]

def bsFun(x):
    pv=np.apply_along_axis(bs1,1,x)
    return pv

print("bs=",bsFun(index_test))

def bs_ind(x):
    y = x.astype(int)
    return bsFun(index_to_var(y))

print("bs_ind=",bs_ind(ind1))
#sz = [n] * d
#ind_all = np.unravel_index(np.arange(n ** d), sz)
#ind_all = np.array(ind_all).T
#ft = reshape(myfun(ind_all), sz)
#xall = tt.tensor(ft, 1e-8)
#x0 = tt.tensor(ft, 1e-8)
#%%

class TrainBSData:
    def __init__(self,strikes,ts,Ss,sigmas,rs):
        self.Strikes = strikes
        self.Ts = ts
        self.Ss = Ss
        self.sigmas = sigmas
        self.rs = rs
        self.A = [self.Ss,self.Strikes,self.Ts,self.rs,self.sigmas]
        self.shape = [len(x) for x in self.A]
    


class TrainBS:
    def __init__(self,bsData : TrainBSData, r : int):
        self.bsData=bsData
        self.r=r
    
    def index_to_var(self,x):
        A=self.bsData.A
        s_i,k_i,t_i,r_i,sig_i = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4]
        Aa=np.array([A[0][s_i],A[1][k_i],A[2][t_i],A[3][r_i],A[4][sig_i]]).T
        return Aa

    def bs_ind(self,x):
        y = x.astype(int)
        return bsFun(self.index_to_var(y))


    def train(self):
        d = 5
        n = self.bsData.shape
        r = self.r
        x0 = tt.rand(n, d, r)
        x1 = rect_cross.cross(self.bs_ind, x0, nswp=5, kickrank=1, rf=2)
        return x1

    def test(self,p,train_res):
        data = self.bsData
        print("shape=",data.shape)
        #p=[1,0,65,23,15]
        print("p    =",p)
        bs_data=[data.A[j][p[j]] for j,k in enumerate(p)]
        print("[Ss,Strikes,Ts,rs,sigmas]=\n",bs_data)
        u1=unit(data.shape,5,lin_ind(data.shape,p))
        print("tt-cross=",tt.dot(train_res,u1))
        print("bs      =",bs1(bs_data))

#%%
Strikes = np.arange(50,150,5)
Ts = np.arange(0.1,1,0.01)
Ss = np.arange(50,150,5)
sigmas = np.arange(0.1,0.7,0.01)
rs = np.arange(0.01,.25,0.01)

data=TrainBSData(Strikes,Ts,Ss,sigmas,rs)
#%%
train = TrainBS(data,10) 
train_res=train.train()
#%%
print("1. preparing .....")
Strikes = np.arange(50,250,1)
Ts = np.arange(0.1,5,0.01)
Ss = Strikes
sigmas = np.arange(0.1,0.7,0.01)
rs = np.arange(0.01,.45,0.01)

data1=TrainBSData(Strikes,Ts,Ss,sigmas,rs)
train1 = TrainBS(data1,10)
print("2. training ......") 
train_res1=train1.train()
print('3. all done, you can tes!')




#%%
np.prod(data1.shape)
#%% 
##############3333 Testing .... #################
p=[60,60,100,0,20]
print(train_res1.core.shape)
print(train_res1.core.shape/np.prod(data1.shape))
train1.test(p,train_res1)
print(train_res1.__getitem__(p))

#print(tt.dot(tt.ones(data1.shape,5),train_res1))
print(tt.sum(train_res1[:,0,0,0,0]))

#%%
############### Testing fit #########
print("shape=",data.shape)
p=[1,0,65,23,15]
print("p    =",p)
bs_data=[data.A[j][p[j]] for j,k in enumerate(p)]
print("[Ss,Strikes,Ts,rs,sigmas]=\n",bs_data)
u1=unit(data.shape,5,lin_ind(data.shape,p))
print("tt-cross=",tt.dot(train_res,u1))
print("bs      =",bs1(bs_data))
#%%

'''
train = TrainBS(data,10) ...
swp: 0/4 er_rel = 2.8e+00 er_abs = 6.7e+05 erank = 15.0 fun_eval: 44472
swp: 1/4 er_rel = 3.2e-03 er_abs = 7.7e+02 erank = 20.0 fun_eval: 129680
swp: 2/4 er_rel = 7.9e-04 er_abs = 1.9e+02 erank = 25.7 fun_eval: 273300
swp: 3/4 er_rel = 2.9e-04 er_abs = 6.9e+01 erank = 31.4 fun_eval: 494640
swp: 4/4 er_rel = 3.6e-06 er_abs = 8.6e-01 erank = 36.9 fun_eval: 809956
'''

fun_evals=\
[44472,
129680,
273300,
494640,
809956]

sum(fun_evals)

print("sum_evals    ",1752048)
print("total params ",np.prod(data.shape))
print("tt params    ",train_res.core.shape[0])

#%%
d = 5
n = shape
r = 3
x0 = tt.rand(n, d, r)
x1 = rect_cross.cross(bs_ind, x0, nswp=5, kickrank=1, rf=2)
#%%

#%%
# x1 contains approximation

print("shape=",shape)
p=[2,0,5,0,0]
print("p    =",p)
u1=unit(shape,5,lin_ind(shape,p))
print("tt-cross=",tt.dot(x1,u1))
#[A[0][0],A[1][0],A[2][0],A[3][0],A[4][0]]

print("bs      =",bs1([A[j][p[j]] for j,k in enumerate(p)]))

# %%
x1.full()
# %%
xf=tt.xfun([3,2,3],1)
# %%
xf.full()
# %%
u1=unit((3,2,3),2,[0,0,0])
print(u1.full())
print(tt.dot(xf,u1))
# %%
###########################33
# 
# try to approximate payoff*density and then calcaulate inegral by summing up
#%%
#%%
class BlackScholesMCInput:
    def __init__(self,s,k,r,t,sigma,N):
        self.S=s
        self.r=r
        self.K=k
        self.T=t
        self.sigma=sigma
        self.N = N

class BlackScholesMC:

    def price(self,inp : BlackScholesMCInput):
        nAssets=1
        w=np.random.normal(0.0,size=[nAssets,inp.N])
        S=inp.S*np.exp((inp.r-.5*inp.sigma**2)*inp.T + np.sqrt(inp.T)*inp.sigma*w)
        payoff = np.maximum(S-inp.K,0)
        pv = payoff*np.exp(-inp.r*inp.T)
        p = np.mean(pv)
        return p#S,payoff,pv,p

class BlackScholesIntegration:

    def price(self,inp : BlackScholesMCInput):
        nAssets=1
        density=norm.pdf
        x = np.linspace(norm.ppf(0.00001),
                norm.ppf(0.99999), 100000)
        dx = x[1:]-x[:-1]
        dP = density(x[1:])*dx
        payoff = lambda s : np.maximum(s-inp.K,0)
        S=inp.S*np.exp((inp.r-.5*inp.sigma**2)*inp.T + np.sqrt(inp.T)*inp.sigma*x[1:])
        integral = np.dot(payoff(S),dP)
        pv = integral*np.exp(-inp.r*inp.T)
        return pv

class BlackScholesMCInputMA(BlackScholesMCInput):
    
    def __init__(self,s,k,r,t,sigma,C,N):
        super().__init__(s,k,r,t,sigma,N)
        self.C=C
    
class BlackScholesMCMA:

    def price(self,inp : BlackScholesMCInputMA):
        nAssets=len(inp.S)
        rr = inp.r*np.ones(nAssets)
        mean=np.zeros(nAssets)
        mean = mean+ (rr - .5*np.array(inp.sigma)**2)*inp.T
        cov = inp.C
        sigs=np.concatenate([sig*np.ones([nAssets,1]) for sig in inp.sigma],axis=1)
        cov=cov*sigs*sigs.T*inp.T
        #print(mean)
        #print(cov)
        w=np.random.multivariate_normal(mean,cov,size=inp.N) 
        S = np.array(inp.S)*np.exp(w)
        Karr=np.array(inp.K)
        SminusK=S-Karr
        payoff = np.amax(np.maximum(SminusK,0,),axis=1)
        pv = np.mean(payoff)*np.exp(-inp.r*inp.T)
        return pv

class BlackScholesIntegralMA:

    def price(self,inp : BlackScholesMCInputMA):
        nAssets=len(inp.S)
        rr = inp.r*np.ones(nAssets)
        mean=np.zeros(nAssets)
        mean = mean+ (rr - .5*np.array(inp.sigma)**2)*inp.T
        cov = inp.C
        sigs=np.concatenate([sig*np.ones([nAssets,1]) for sig in inp.sigma],axis=1)
        cov=cov*sigs*sigs.T*inp.T

        dist = multivariate_normal(mean,cov)
        
        u=np.random.uniform(-10,10,size=(nAssets,inp.N)).T
        #print(u)
        S = np.array(inp.S)*np.exp(u)
        mu=dist.pdf(u)
        print("mu=",mu)
        #print(S)
        Karr=np.array(inp.K)
        SminusK=S-Karr
        payoff = np.amax(np.maximum(SminusK,0,),axis=1)*mu
        pv = np.mean(payoff)*np.exp(-inp.r*inp.T)*((20)**nAssets)
        return pv


#%%
class PayoffInp:
    def __init__(self,strikes):
        self.strikes = strikes


class VariableRange:
    def __init__(self,low,high,dx):
        self.low = low
        self.high = high
        self.dx = dx

class BSVariableRanges:
    def __init__(self,\
        s0 : VariableRange,\
        sig : VariableRange,\
        r : VariableRange,\
        t : VariableRange):
        
        self.s0=s0
        self.sig=sig
        self.r=r
        self.t=t


class TTDiscretization:
    def __init__(self,num_assets, bsRanges : BSVariableRanges):
        self.num_assets = num_assets
        self.ranges = bsRanges
    
    def get_discretization(self,x : VariableRange):
        factor = self.get_discretization1(x)
        return np.array(self.num_assets*[factor])

    def get_discretization1(self,x : VariableRange):
        factor = np.arange(x.low,x.high,x.dx)
        return factor


    def get_arrays(self):
        factor = np.arange(-7,7,self.get_dx())
        s0=self.get_discretization(self.ranges.s0)
        return {\
            "spot_base":np.array(self.num_assets*[factor]),\
            "s0":s0,\
            "r":self.get_discretization1(self.ranges.r),\
            "t":self.get_discretization1(self.ranges.t),\
            "sigma":self.get_discretization(self.ranges.sig)}

    def get_dx(self):
        return 0.1
    
    def get_dvol(self):
        return self.get_dx()**self.num_assets

    
class TTDiscretizationIndex:
    def __init__(self,s : np.array,dvol :float):
        self.disc_arrays=s
        self.dvol1=dvol
        
    
    def dvol(self):
        return self.dvol1

    def get_n(self):
        s=self.disc_arrays['spot_base'].shape
        s0=self.disc_arrays['s0'].shape
        sig=self.disc_arrays['sigma'].shape
        r=self.disc_arrays['r'].shape
        t=self.disc_arrays['t'].shape
        gn=sig[0]*[sig[1]]+s0[0]*[s0[1]]+s[0]*[s[1]]+[r[0]]+[t[0]]

        
        return gn

    def get_d(self):
        return len(self.get_n())
        
    def index_to_var(self,index) -> dict:
        # index structure: 
        
        s = self.disc_arrays['spot_base']
        d = len(s)
        spot_base = [s[j][ind] for j,ind in enumerate(index[2*d:3*d])]
        
        s = self.disc_arrays['s0']
        s0 = [s[j][ind] for j,ind in enumerate(index[d:2*d])]
        
        s = self.disc_arrays['sigma']
        sigma = [s[j][ind] for j,ind in enumerate(index[0:1*d])]
        
        s = self.disc_arrays['r']
        r = [s[ind] for j,ind in enumerate(index[3*d:3*d+1])]

        s = self.disc_arrays['t']
        t = [s[ind] for j,ind in enumerate(index[3*d+1:3*d+2])]

        return {'spot_base':spot_base,'s0':s0,'sigma':sigma,'r':r,'t':t}

class TTDiscretizationIndexOnlySpot(TTDiscretizationIndex):
    def __init__(self,s : np.array,dvol :float):
        TTDiscretizationIndex.__init__(self,s,dvol)
        

    def get_n(self):
        s=self.disc_arrays['spot_base'].shape
        
        gn=s[0]*[s[1]]

        return gn

        
    def index_to_var(self,index) -> dict:
        # index structure: 
        #ind1 = index.reshape()
        #s = self.disc_arrays['spot']
        #d = len(s)
        #spot = [s[j][ind] for j,ind in enumerate(index[2*d:3*d])]
        
        #return {'spot':spot}
        pass



def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))

class BlackScholesTTMA:

    def __init__(self,r,payoff_inp : PayoffInp, indexing : TTDiscretizationIndex):
        self.r = r
        self.payoff_inp = payoff_inp
        self.indexing = indexing
        

    def index_to_var(self,x):
        return self.indexing.index_to_var(x)
    
    def get_payoff_times_density_ind(self,inp):
        Karr=np.array(self.payoff_inp.strikes)
        Sarr=np.array(inp.S)
        print("Sarr",Sarr,"Karr",Karr)
        nAssets=len(inp.S)
        dist = multivariate_normal()#(mean,cov)
        def payoff_times_density_ind1(ind):
            ivar = self.index_to_var(ind)
            inpsigma = np.array(ivar['sigma'])#np.array(inp.sigma)#
            inpr = inp.r#np.array(ivar['r'])
            inpT = inp.T#np.array(ivar['t'])
            
            rr = inpr*np.ones(nAssets)
            mean=np.zeros(nAssets)
            mean = mean + (rr - .5*inpsigma**2)*inpT
            cov = inp.C.copy()
            sigs=np.concatenate([sig*np.ones([nAssets,1]) for sig in inpsigma],axis=1)
            cov=cov*sigs*sigs.T*inpT

            


            u = np.array(ivar['spot_base'])
            s0 = np.array(ivar['s0'])
            S = s0*np.exp(u)
            SminusK=S-Karr
            mu=pdf_multivariate_gauss(u,mean,cov)
            payoff = np.amax(np.maximum(SminusK,0,),axis=0)*mu
            return payoff

        
        def payoff_times_density_ind(x):
            y = x.astype(int)
            return np.apply_along_axis(payoff_times_density_ind1,1,y)

        return payoff_times_density_ind

    def fit(self,inp : BlackScholesMCInputMA):
        start = time.time()
        print("fitting TT.....")
        nAssets=len(inp.S)
        d = self.indexing.get_d()
        n = self.indexing.get_n()
        r = self.r

        '''
        nAssets=len(inp.S)
        rr = inp.r*np.ones(nAssets)
        mean=np.zeros(nAssets)
        mean = mean+ (rr - .5*np.array(inp.sigma)**2)*inp.T
        cov = inp.C
        sigs=np.concatenate([sig*np.ones([nAssets,1]) for sig in inp.sigma],axis=1)
        cov=cov*sigs*sigs.T*inp.T

        dist = multivariate_normal(mean,cov)
        '''

        print("ndr=",n,d,r)
        x0 = tt.rand(n, d, 1)
        x0 = 100.0*x0
        x1 = rect_cross.cross(self.get_payoff_times_density_ind(inp), x0,nswp=15)#, nswp=8, kickrank=1, rf=2)     

        
        endt = time.time()
        print("done fitting T={}! ".format(endt-start))
        return x1

    
    def get_tensor(self,inp : BlackScholesMCInputMA):
        # fit
        x1=self.fit(inp)
        # calculate pv
        return x1#,tt.sum(x1)*np.exp(-inp.r*inp.T)*self.indexing.dvol()
    
    def get_pv(self,inp,tensor):

        a1 = self.indexing.disc_arrays
        s0_i  = [np.abs(a1['s0'] - s).argmin() for s in inp.S]
        sig_i = [np.abs(a1['sigma'] - s).argmin() for s in inp.sigma]
        r_i,t_i = \
                np.abs(a1['r'] - r).argmin(),\
                np.abs(a1['t'] - t).argmin()

        n =self.indexing.get_n()
        self.tensor = tensor
        num_assets=len(inp.S)
        to_sum_index=[range(n[2*num_assets + l]) for l in range(num_assets)]
        to_sum_index=sig_i+s0_i+to_sum_index+[r_i]+[t_i]
        pv=tt.sum(tensor[to_sum_index])*np.exp(-inp.r*inp.T)*self.indexing.dvol()
        return pv
    
    def price(self,inp : BlackScholesMCInputMA):
        x1=self.get_tensor(inp)
        pv=self.get_pv(inp,x1)
        return x1,pv



# %%

s,k,r,t,sigma,C,N = (12,12),(11.5,10.5),0.01,1.2,(.13,.2),np.array([[1,.7],[.7,1]]),1000000
inpMA=BlackScholesMCInputMA(s,k,r,t,sigma,C,N)
s_range,sig_range,r_range,t_range = \
    VariableRange(0.0,20,1),\
    VariableRange(0.1,.4,.01),\
    VariableRange(0.01,.1,.01),\
    VariableRange(0.0,2,.1)        
bs_ranges = BSVariableRanges(s_range,sig_range,r_range,t_range)
disc=TTDiscretization(len(s),bs_ranges)
pricers = [ ]

#plt.scatter(w[0][:,0],w[0][:,1])
#print("w=",w)

#s,k,r,t,sigma,C,N = (120,100),(95,115),0.01,1.2,(.3,.2),np.array([[1,.7],[.7,1]]),1000000
#%%
inpMA=BlackScholesMCInputMA(s,k,r,t,sigma,C,N)
pricers = [BlackScholesMCMA(),\
    BlackScholesIntegralMA(),\
        BlackScholesTTMA(10,PayoffInp(k),TTDiscretizationIndex(disc.get_arrays(),disc.get_dvol()))]
#%%
w=[pr.price(inpMA) for pr in pricers]
#plt.scatter(w[0][:,0],w[0][:,1])
print("w=",w)
#%%
disc_ind=TTDiscretizationIndex(disc.get_arrays(),disc.get_dvol())
print(np.array(disc_ind.index_to_var((100,100,10,10,25,26,10,200))['sigma'] )  )
#print(disc_ind.disc_arrays['sigma'])

#%%
s,k,r,t,sigma,C,N = (12,9),(11.5,10.5),0.01,1.2,(.3,.2),np.array([[1,.7],[.7,1]]),1000000

inp2 = BlackScholesMCInputMA(s,k,r,t,sigma,C,N)
print("tt=",pricers[2].get_pv(inp2,w[2][0]))

w1=[pr.price(inp2) for pr in pricers[0:2]]
print("w=",w1)

#%%
#tens=w[2][0]
#tens[]
#tens = np.load('blackscholes_8d.npy',allow_pickle=True)

ind1=TTDiscretizationIndex(disc.get_arrays(),disc.get_dvol())

a1=disc.get_arrays()
def graph_stuff(a1,tenz):
    show_by_s=True
    if show_by_s:
        inpps = [BlackScholesMCInputMA([s1,5],k,r,t,[.33,.33],C,N) for s1 in a1['s0'][0]]
        xs=a1['s0'][0]
        ys_tt=[pricers[2].get_pv(inpp,tenz) for inpp in inpps]
        ys_mc=[pricers[0].price(inpp) for inpp in inpps]
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks(xs)
        ax.set_yticks(np.arange(0,10,.1))
        plt.scatter(xs,ys_tt,label="tt")
        plt.scatter(xs,ys_mc,label="mc")
        plt.legend()
        plt.grid()
        fig.autofmt_xdate()
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_yticks(ax.get_yticks()[::4])
        fig.tight_layout()
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.show()



    xs=a1['sigma'][0]
    inpps = [BlackScholesMCInputMA([9,10],k,r,t,[sig,.2],C,N) for sig in xs]
    ys_tt=[pricers[2].get_pv(inpp,tenz) for inpp in inpps]
    ys_mc=[pricers[0].price(inpp) for inpp in inpps]

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(xs)
    ax.set_yticks(np.arange(0,10,.1))
    plt.scatter(xs,ys_tt,label="tt")
    plt.scatter(xs,ys_mc,label="mc")
    plt.legend()
    plt.grid()
    fig.autofmt_xdate()
    ax.set_xticks(ax.get_xticks()[::2])
    fig.tight_layout()
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()

tenz=tt.tensor.from_list(np.load('blackscholes_8d.npy',allow_pickle=True))
graph_stuff(a1,tenz)

#%%

#a=a1['s0']
#a0=np.array([12,13])
#agmin=np.abs(a - a0).argmin()
#print(agmin)
#print(a.flat[agmin])
#print(a[0][12])

s01_i,s02_i,sig1_i,sig2_i,r_i,t_i = \
    np.abs(a1['s0'] - s[0]).argmin(),\
    np.abs(a1['s0'] - s[1]).argmin(),\
    np.abs(a1['sigma'] - sigma[0]).argmin(),\
    np.abs(a1['sigma'] - sigma[1]).argmin(),\
    np.abs(a1['r'] - r).argmin(),\
    np.abs(a1['t'] - t).argmin()


print("s01_i,s02_i,sig1_i,sig2_i,r_i,t_i=",s01_i,s02_i,sig1_i,sig2_i,r_i,t_i)
print("s01_i,s02_i,sig1_i,sig2_i,r_i,t_i=",a1['s0'][0][s01_i],a1['s0'][1][s02_i],a1['sigma'][0][sig1_i],a1['sigma'][1][sig2_i],a1['r'][r_i],a1['t'][t_i])
print("sum=",tt.sum(tens[:,:,s01_i,s02_i,sig1_i,sig2_i,r_i,t_i])*ind1.dvol()*np.exp(-r*t))
#%%
s,k,r,t,sigma,N = 100,115,0.01,1.2,.2,1000000
inp=BlackScholesMCInput(s,k,r,t,sigma,N)
pricer = BlackScholesMC()
prcer_integral = BlackScholesIntegration()
o_integral = prcer_integral.price(inp)
bs_price = bs(s,k,t,r,0,sigma)
for _ in range(10):
    o=pricer.price(inp)
    print("mc={} integral={} bs={}".format(o,o_integral,bs_price))


#print(o[3])
#print(np.concatenate(o[0:3],axis=0).T)

#%%

fig, ax = plt.subplots(1, 1)
x = np.linspace(norm.ppf(0.01),
                norm.ppf(0.99), 100)
ax.plot(x, norm.pdf(x),
       'r-', lw=5, alpha=0.6, label='norm pdf')
# %%
x
# %%
[x[j]-x[j+1] for j in [1,2,3,4,5,6]]
# %%

#%%%%%%%%%%%%%%%%%%55

x=np.arange(16).reshape((4,2,2))
z=np.arange(10).reshape(2,5)
np.save("tempnp.bin",[x,z])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = np.load('tempnp.bin.npy',allow_pickle=True)
y
# %%
x
# %%
tens_l = tt.tensor.to_list(tens)
np.save('blackscholes_8d',tens_l)
#%%
tens1 = np.load('blackscholes_8d.npy',allow_pickle=True)
# %%
d=7
n=2**d
f = tt.vector(np.array([x**2 for x in np.arange(0,1,1/n)]).reshape(d*[2]) )

# %%
f.full().shape
# %%
f.core.shape
# %%
cores=tt.vector.to_list(f)
# %%
cores
# %%
pr=0
for x in cores:
    pr+=np.prod(x.shape)
print(pr)
# %%
#M = tt.matrix()
class Kk:
    def __init__(self):
        self.iik=0
    def finite_diff(self,i2):
        #print(i,"\n")
        #print(k,"\n")
        i1 = i2.astype(int)
        y=np.zeros((i1.shape[0],1),np.float)
        #print(self.iik)
        self.iik+=1
        for u in range(i1.shape[0]):
            i=i1[u][:d]
            k=i1[u][d:]

            
            l_i = int(sum([i[j]*2**j for j in range(d)]))
            k_i = int(sum([k[j]*2**j for j in range(d)]))
            y[u]=0.
            if k_i==l_i-1:
                y[u]=0.#-1.
            else:
                if k_i==l_i:
                    y[u]=1.
                    #print("!!!!!!!!!!!!!!!!!",k_i,)
        return y
    
def expand_index(i,b,d):
    j=i
    r=[]
    while d>0:
        rr=j % b
        r.append(int(rr))
        j-=rr
        j/=b
        d-=1
    return list(reversed(r))

#%%


#%%
k = Kk()
m = np.zeros(2*d*[2])
for i in range(n):
    for j in range(n):
        ii = expand_index(i,2,d)
        jj = expand_index(j,2,d)
        ind=[*ii,*jj]
        m[tuple(ind)] = k.finite_diff(np.array(ind).reshape(1,-1))[0,0]
#%%
pd.DataFrame(m.reshape(n,n))
#%%
ttmat=tt.matrix(m)

#%%
#print(finite_diff([0,0,0,0,0,0,0,0,1,1]))
x0  = tt.vector(1*np.ones(np.array(2*d*[2])))
x0=x0*10.0
x1 = rect_cross.cross(Kk().finite_diff, x0,nswp=20)
# %%
x1.full().reshape(n,n) -  ttmat.full().reshape(n,n)
#%%
ttmat.full().reshape(n,n)
# %%
np.array(2*d*[2])
# %%
x0.full().shape
# %%
a=x1.full().reshape((n,n))
# %%
#pd.DataFrame(a)
a
# %%
pd.DataFrame(a)
# %%
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

a = [-1, 0]; b = [1, 0, -1]; c = [0, 1]
A = tridiag(a, b, c)
A

# %%
d=6
nn=2**d
n=nn#nn-1

D=-.5*n*np.eye(nn,nn,k=-1) - np.eye(nn,nn)*0 - .5*n*np.eye(nn,nn,k=1)*(-1)
D[0,0]=-1*n
D[0,1]=1*n
D[n-1,n-1]=1*n
D[n-1,n-2]=-1*n

# boundary condition f(0) = c:
c=4.0
D[0,0]+=1
#D[1,0]=0
#D[2,0]=1
#D[nn-1]=np.zeros((nn))
#D[:,nn-1]=np.zeros((nn))
#D[nn-1,0]=1
#D[0,1]=0
D

#%%
def Bd(y):
    z = np.zeros_like(y)
    z[0]=c
    #z[1]=c
    return y + z

#%%
print("rank=",np.linalg.matrix_rank(D))
print('shape',D.shape)

# %%
start1,stop1=-2,2
step = (stop1-start1)/n
x=np.arange(start1,stop1,step)
print(x)
f=x**2
#f=np.ones(n)
print("shape x=",x.shape)
print(f)
print("\n\n",D.dot(f))
print("\n\n",Bd(f))
#%%
#sigma=.2
#r=0.05
#L = -.5*sigma**2*D.dot(D)-(r)


# %%

tt_m=tt.matrix(D.reshape(2*d*[2]))
#tt_bd=tt.matrix(Bd.reshape(2*d*[2]))

# %%
d*[2]
# %%
pp=1
for _x in tt.matrix.to_list(tt_m):
    pp+=np.prod(_x.shape)
print(pp)
print(np.prod(D.shape))
print(3*n)
# %%
x.shape
# %%
#plt.plot(x,f)
f_disp=f+c
dDotF= D.dot(f_disp)
dDotF[0]-=c
plt.plot(x,dDotF)
plt.show()
plt.plot(x,f_disp)
#%%
D.dot(f)
# %%
tt_m
# %%
tt_f = tt.vector(f.reshape(d*[2]))
#%%
tt_f.full().reshape(-1)
# %%
m_times_f=tt.matvec(tt_m,tt_f)
m_times_f_bd = tt.vector(Bd(m_times_f.full().reshape(-1)).reshape(d*[2]))
# %%
m_times_f_full=m_times_f.full().reshape(2**d)
# %%
np.sum(np.abs(m_times_f_full-D.dot(f)))
# %%
def _ind2sub(siz, idx):
    '''
    Translates full-format index into tt.vector one's.
    ----------
    Parameters:
        siz - tt.vector modes
        idx - full-vector index
    Note: not vectorized.
    '''
    _np=np
    n = len(siz)
    subs = np.empty((n))
    k = np.cumprod(siz[:-1])
    k = np.concatenate((_np.ones(1), k))
    for i in range(n - 1, -1, -1):
        subs[i] = _np.floor(idx / k[i])
        idx = idx % k[i]
    return subs.astype(_np.int32)


def _unit(n, d=None, j=None, tt_instance=True):
    ''' Generates e_j _vector in tt.vector format
    ---------
    Parameters:
        n - modes (either integer or array)
        d - dimensionality (integer)
        j - position of 1 in full-format e_j (integer)
        tt_instance - if True, returns tt.vector;
                      if False, returns tt cores as a list
    '''
    if isinstance(n, int):
        if d is None:
            d = 1
        n = n * np.ones(d, dtype=np.int32)
    else:
        d = len(n)
    if j is None:
        j = 0
    rv = []

    j = _ind2sub(n, j)

    for k in range(d):
        rv.append(np.zeros((1, n[k], 1)))
        rv[-1][0, j[k], 0] = 1
    if tt_instance:
        rv = tt.vector.from_list(rv)
    return rv
#%%
def tt_bd(v):
    mm = [2 for _ in range(d)] 
    mi=0
    print(mi)
    v+=c*unit(mm,d,0)
    return v



#%%
intgrate_back=amen_solve(tt_m,m_times_f_bd,tt.ones(d*[2]),1e-6)
plt.plot(x,intgrate_back.full().reshape(2**d))
plt.show()

plt.plot(x,intgrate_back.full().reshape(-1))
plt.show()

tt_bd_integrate1=tt_bd(intgrate_back)
tt_bd_integrate1.full().reshape(-1)
#plt.plot(x,tt_bd_integrate1.full().reshape(-1))
#plt.show()


intgrate_back2=amen_solve(tt_m,tt_bd_integrate1,tt.ones(d*[2]),1e-6)
plt.plot(x,intgrate_back2.full().reshape(2**d))
plt.show()

tt_bd_integrate2=tt_bd(intgrate_back2)
intgrate_back3=amen_solve(tt_m,tt_bd_integrate2,tt.ones(d*[2]),1e-6)
plt.plot(x,intgrate_back3.full().reshape(2**d))
plt.show()


# %%
np.sum(np.abs(intgrate_back.full().reshape(2**d)-f))
# %%
plt.plot(x,intgrate_back.full().reshape(2**d))
# %%
plt.plot(x,m_times_f.full().reshape(2**d))
# %%
# %%
400/6
# %%
m_times_f.full().reshape(-1)
# %%
m_times_f_bd.full().reshape(-1)
# %%

# %%
ttD = tt.matrix(D.reshape(2*d*[2]))
ttI = tt.matrix(np.eye(2**d).reshape(2*d*[2]))

tt2=ttD.__kron__(ttI)
# %%
ttD
# %%
tt2
# %%
ttD
# %%
x=np.arange(0,1,1./n)
y=np.arange(0,1,1./n)
f2 = np.zeros((len(x),len(y)))
for xx in range(len(x),len(y)):
    for yy in y:
        f2[xx,yy]=x[xx]**y[yy]*3
# %%
f2.shape
# %%
tt2_times_f2=tt.matvec(tt2,tt.vector(f2.reshape(2*d*[2])))
# %%
tt2_times_f2.full()
# %%
f=x**2+4
#print(inv(D).dot(intgrate_back.full().reshape(2**d))-f)
plt.plot(x,f)
plt.show()
Df=D.dot(f)
plt.plot(x,Df)
plt.show()
invDDf = inv(D).dot(Df)
plt.plot(x,Bd(invDDf))
print(invDDf-f)
# %%
inv(D).dot(Df)
# %%
f[0]
# %%
d=10
nn=2**d
n=nn#nn-1

D=-.5*n*np.eye(nn,nn,k=-1) - np.eye(nn,nn)*0 - .5*n*np.eye(nn,nn,k=1)*(-1)
D[0,0]=-1*n
D[0,1]=1*n
D[n-1,n-1]=1*n
D[n-1,n-2]=-1*n

B = np.zeros_like(D)
B[0,0]=1

start1,stop1 =-5,6
step = (stop1-start1)/n
x = np.arange(start1,stop1,step)

sigma = 0.2
r=0.0
k=np.log(100)
time_steps=100
T=1

I = np.eye(n)
BS = .5*(sigma**2)*D.dot(D)+ r*D - r*I
payoff = np.maximum(np.exp(x)-np.exp(k),0)

B = np.zeros_like(BS)
B[0,0]=1
B[n-1,n-1]=1

BB = np.eye(n)
BB = BB + B
#f=1./(x**2)

#plt.plot(x,BS.dot(f))
T=1
time_step = T/time_steps
Times = np.arange(0,T,time_step)
dt=Times[1]-Times[0]
op=inv(I-dt*(BS+B))


print(np.linalg.matrix_rank(B))
print(np.linalg.matrix_rank(BS))
print(np.linalg.matrix_rank(BS+B))
print(np.linalg.matrix_rank(I+dt*(BS+B)))
print(BS.shape)
BS
#%%


u=payoff.copy()
u[-1]-=(np.exp(x[-1])-np.exp(k))*dt
u=op.dot(u)
plt.plot(np.exp(x),payoff,np.exp(x),u)
plt.show()
for t in range(time_steps-1,0,-1)[:5]:
    u[-1]-=(np.exp(x[-1])-np.exp(k))*dt
    u=op.dot(u)
    plt.plot(np.exp(x),u)
    plt.show()
    print("t=",t,u[-1]- (np.exp(x[-1]) - np.exp(k)),u[0] )

plt.plot(np.exp(x),u)
plt.show()


print(np.linalg.matrix_rank(BS))
print(np.linalg.matrix_rank(D.dot(D)))
D
#
#np.matmul(D,D)
# %%
u
# %%
plt.plot(x,payoff)
# %%
D
# %%
np.linalg.matrix_rank(D)
# %%
n=2**10
DD=np.zeros((n,n))
start1,stop1 =-8,8
step = (stop1-start1)/n
x = np.arange(start1,stop1,step)
h=step
DD[0,1]=1/(2*h)
for jj in range(n-2):
    j=jj+1
    DD[j,j-2+1]=-1/(2*h)
    DD[j,j+1]=1/(2*h)
# bd condition 2nd derivative = 0
DD[n-1][n-1]=1/(h)
DD[n-1][n-2]=-1/h

DDD=np.zeros((n,n))
DDD[0,1]=1/(h*h)
DDD[0,0]=-2/(h*h)
for jj in range(n-2):
    j=jj+1
    DDD[j,j-1]=1/(h*h)
    DDD[j,j]=-2/(h*h)
    DDD[j,j+1]=1/(h*h)
# bd condition 2nd derivative = 0
DDD[n-1][n-1]=-2/(h*h)
DDD[n-1][n-2]=1/(h*h)

print(DD)
print("MyRakn=",np.linalg.matrix_rank(DDD))

#%%
print(DD)
print(np.linalg.matrix_rank(DD))
sigma=0.3
I = np.eye(n)
r=0.05
BS = .5*(sigma**2)*(DD.dot(DD) - DD)  + r*DD - r*I
BS1 = .5*(sigma**2)*(DD.dot(DD))
BS2 = .5*(sigma**2)*DDD

print(np.linalg.matrix_rank(BS))

# %%
T=1
time_steps=400
time_step = T/time_steps
Times = np.arange(0,T,time_step)
dt=Times[1]-Times[0]
op=inv(I-dt*(BS))
op1 = inv(I-dt*(BS1))
op2 = inv(I-dt*(BS2))

payoff = np.maximum(np.exp(x)-np.exp(k),0)
u=payoff.copy()
u1=payoff.copy()
u2=payoff.copy()
print(T)
u=op.dot(u)
plt.plot(np.exp(x),payoff,np.exp(x),u)
plt.show()
for t in range(time_steps-1,0,-1):
    #print("t",Times[t])
    u=op.dot(u)
    u1=op1.dot(u1)
    u2_ = u2.copy()
    u2_[-1]=0
    u2=op2.dot(u2_)
plt.plot(np.exp(x),u)
plt.plot(np.exp(x),u1)
plt.plot(np.exp(x),u2)
plt.show()
plt.plot(np.exp(x),u1)
plt.plot(np.exp(x),u2)
plt.show()

plt.plot(x,u1)
plt.plot(x,u2)
plt.show()
#%%
plt.plot(np.exp(x),DDD.dot(u2))
plt.show()

# %%
print(x.shape)
i=815
print(np.exp(x[i]))
print(u[i],u1[i],u2[i])
bs(np.exp(x[i]),np.exp(k),T,r,0,sigma)
# %%
plt.plot(x[-100:],u2[-100:])
# %%
x[-100:]
# %%
ttBS = tt.matrix(BS.reshape(2*d*[2]))
# %%
ttBS
# %%
bslinear=BS.reshape(-1)
# %%
sum([1 for x in bslinear if np.abs(x)>1e-10])
# %%
ttBS.tt.core.shape
# %%
ttD=tt.matrix(DD.reshape(2*d*[2]))


# %%
ttD1=tt.kron(ttD,tt.eye(ttD.n))


# %%
ttD1
# %%

# %%
ttu = tt.vector(payoff.reshape(d*[2]))
#%%
ttu
#%%
ttI=tt.eye(ttBS.n)
#%%
ttu = tt.vector(payoff.reshape(d*[2]))
for t in range(time_steps,0,-1):
    ttu=tt.amen.amen_solve(-dt*ttBS + ttI,ttu,ttu,1e-8)
    #plt.plot(np.exp(x),payoff,np.exp(x),ttu.full().reshape(-1))
    #plt.show()

#%%
#Q:  can we solve BS one mach (without time iteration)?
#Q:  what would be the rank for multidimesnial operator with correlations (can test it against MC simulation)
#%%
#BS = .5*(sigma**2)*(DD.dot(DD) - DD)  + r*DD - r*I

# u(t,x): u' = - BS u

def make_Dt(n,T):
    DD=np.zeros((n,n))
    start1,stop1 =0,T
    h = (stop1-start1)/n
   
    for j in range(n-1):
        DD[j,j]=-1/h
        DD[j,j+1]=1/h
        
    DD[n-1][n-1]=-1/h
    return DD,h

def make_tt_lhs(dt,g):
    j=0
    z=tt.vector(np.zeros(d*[2]))
    lhs=tt.kron(z,z)
    #print(lhs)
    jlastInv=invInd(2**d-1,d*[2])
    for gx in g:
        print("j,gx: ",j,gx)
        jinv=invInd(j,d*[2])
        e_j=unit(d*[2],j=jinv)
        e_last=unit(d*[2],j=jlastInv)
        e_last*=-gx/dt
        #print (j)
        toadd=tt.kron(e_last,e_j)
        #print("toadd\n",toadd)
        lhs=lhs+toadd
        j+=1
    return lhs

def make_tt_lhs_v2(dt,g):
    j=0
    z=tt.vector(np.zeros(d*[2]))
    lhs=tt.kron(z,z)
    #print(lhs)
    jlastInv=invInd(2**d-1,d*[2])
    e_last=unit(d*[2],j=jlastInv)
    lhs=tt.kron(e_last,tt.vector(-g.reshape(d*[2])/dt))
    return lhs



dt_for_global = 2**(-d)
lhs = make_tt_lhs_v2(dt_for_global,payoff)#make_tt_lhs(dt_for_global,payoff)
#make_Dt(4,1)    
#%%
rrr=lhs.round(1e-10)
#%%
def print_rrr():
    for x in range(2**d):
        y=rrr[ind2sub(2*d*[2],832*x)]
        if (np.abs(y)>1e-10):
            print(x,y)
print_rrr()
#%%

#%%
makeDt,dt=make_Dt(n,T)
Dop=makeDt.reshape(2*d*[2])
ttI = tt.eye(ttD.n)
ttII = tt.kron(ttI,ttI)
ttDx=tt.kron(ttI,ttD)
ttDt=tt.kron(tt.matrix(Dop),ttI) # should include final condition


ttBSFull=ttDt 
ttBSFull += .5*(sigma**2)*(ttDx.__matmul__(ttDx)-ttDx) 
ttBSFull += r*ttDx - r*ttII
#%%
ttBSFull

# lhs for the equation should be 0 + d/dt boundary condition left over 
#lhs = make_tt_lhs(dt,payoff)
#%%
# global solution
#bs_global_sol = tt.amen.amen_solve(ttBSFull,rrr,rrr,1e-8)
bs_global_sol2 = tt.amen.amen_solve(ttBSFull,lhs,lhs,1e-8)
bs_f = bs_global_sol2.full().reshape(2**d,2**d)
plt.plot(np.exp(x[:900]),bs_f[-1,:900])
plt.plot(np.exp(x[:900]),bs_f[0,:900])
plt.show()


#%%
bs_global_sol = tt.amen.amen_solve(ttBSFull,rrr,rrr,1e-10)
#%%
bs_f_apx = bs_global_sol.full().reshape(2**d,2**d)
#%%

#%%

bs_r = rrr.full().reshape(2**d,2**d)
#%%
lhs_full = lhs.full().reshape(2**d,2**d)
#%%
plt.plot(np.exp(x),lhs_full[:,0])
plt.show()
plt.plot(np.exp(x),lhs_full[:,-1])
plt.show()
plt.plot(np.exp(x),lhs_full[0,:])
plt.show()
plt.plot(np.exp(x),-dt*lhs_full[-1,:])
plt.show()
plt.plot(np.exp(x),lhs_full[-2,:])
plt.show()
#%%
plt.plot(np.exp(x),-dt*lhs_full[-1,:])
plt.plot(np.exp(x),-dt*bs_r[-1,:])
plt.show()

#%%
#bs_f[0,:].shape
bs_global_sol2 = tt.amen.amen_solve(ttBSFull,lhs,lhs,1e-11)
bs_f = bs_global_sol2.full().reshape(2**d,2**d)

bs_plot = [bs_f]#,bs_f_apx]
lim_ix = 820
for bs_pl in bs_plot:
    plt.plot(np.exp(x[:lim_ix]),-dt*lhs_full[-1,:lim_ix],label='final cond')
    plt.plot(np.exp(x[:lim_ix]),bs_pl[-1,:lim_ix],label = 'solution at final cond')
    plt.plot(np.exp(x[:lim_ix]),bs_pl[0,:lim_ix],label='at t=0')
    plt.plot(np.exp(x[:lim_ix]),[bs(np.exp(x),np.exp(k),T,r,0,sigma) for x in x[:lim_ix]],label='bs')
    plt.legend()
    plt.show()
#%%
lhs
#%%
ttuu = tt.vector(payoff.reshape(d*[2]))

# %%
plt.plot(np.exp(x),payoff,np.exp(x),ttu.full().reshape(-1))
# %%
ttu.full().reshape(-1)
# %%
print(x.shape)
i=805
for i in [805,815,850,900]:
    print(np.exp(x[i]))
    print("bm,tt,global:",u[i],ttu.full().reshape(-1)[i],bs_f_apx[0, i],bs_f[0, i])
    print("bs=",bs(np.exp(x[i]),np.exp(k),T,r,0,sigma))
    
#%%


def animate_heat_map(data):
    fig = plt.figure()

    
    ax = sns.heatmap(data[0], vmin=0, vmax=1)

    def init():
        plt.clf()
        ax = sns.heatmap(data[0], vmin=0, vmax=1)

    def animate(i):
        plt.clf()
        ax = sns.heatmap(data[i], vmin=0, vmax=1)

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1000)

    plt.show()
# %%
# multivariate Black-Scholes (start with 2 dim x1,x2, t)
# x_i = xi = ln(Si) − (r - .5\sigma_i^2)*tau
#
# du/dt + .5*sum_{i,j}\sigma_i\sigma_j\rho_ij*d^2u/dx_idx_j - r*u=0 
# u (t,x1,x2)

sigma_1 = .12
sigma_2 = .13
rho = .3
T = 1
def ttmkron(a : tt.matrix,b :tt.matrix,c:tt.matrix):
    return tt.kron(tt.kron(a,b),c)



Dt_np,dt=make_Dt(n,T)
ttDt = tt.matrix(Dt_np.reshape(2*d*[2]))
ttDx = tt.matrix(DD.reshape(2*d*[2]))
ttI = tt.eye(ttDx.n)

ttIII = ttmkron(ttI,ttI,ttI)
ttDx1=ttmkron(ttI,ttDx,ttI)
ttDx2=ttmkron(ttI,ttI,ttDx)

ttDx1Dx2 = ttmkron(ttI,ttDx,ttDx)
ttDx1Dx1 = ttmkron(ttI,ttDx.__matmul__(ttDx),ttI)
ttDx2Dx2 = ttmkron(ttI,ttI,ttDx.__matmul__(ttDx))

ttDt=ttmkron(ttDt,ttI,ttI) # should include final condition

_ttIII = tt.kron(ttI,ttI)
_ttDx1=tt.kron(ttDx,ttI)
_ttDx2=tt.kron(ttI,ttDx)

_ttDx1Dx2 = tt.kron(ttDx,ttDx)
_ttDx1Dx1 = tt.kron(ttDx.__matmul__(ttDx),ttI)
_ttDx2Dx2 = tt.kron(ttI,ttDx.__matmul__(ttDx))



'''
ttL=ttDt 
ttL += .5*(2*rho*sigma_1*sigma_2*ttDx1Dx2 +\
     sigma_1*sigma_1*ttDx1Dx1+ sigma_2*sigma_2*ttDx2Dx2) 
ttL -= r*ttIII
'''

ttL=ttDt 
ttL += .5*(2*rho*sigma_1*sigma_2*ttDx1Dx2 +\
     sigma_1*sigma_1*ttDx1Dx1+ sigma_2*sigma_2*ttDx2Dx2) 
ttL += (r-.5*sigma_1**2)*ttDx1 +  (r-.5*sigma_2**2)*ttDx2
ttL -= r*ttIII

ttLx = .5*(2*rho*sigma_1*sigma_2*_ttDx1Dx2 \
    +sigma_1*sigma_1*_ttDx1Dx1+ sigma_2*sigma_2*_ttDx2Dx2) \
    +(r-.5*sigma_1**2)*_ttDx1 +  (r-.5*sigma_2**2)*_ttDx2 \
    - r*_ttIII
#%%

# %%
# payoff as function of t,x1,x2

def bits(nn):
    # The number of bits we need to represent the number

    num_bits = d
    print(num_bits)
    # The bit representation of the number
    bits = [ (nn >> i) & 1 for i in range(num_bits) ]
    bits.reverse()
    # Do we need a leading zero?
    if nn < 0 or bits[0] == 0:
        return bits
    return bits

def get_x_from_spots(spots,r,T,sigma):
    x1 = np.log(spots) - 0*(r-.5*sigma**2)*T
    return x1

def get_spots_from_x(x_vect,r,T,sigma):
    S = np.exp(x_vect)# + 0*(r-.5*sigma**2)*T)
    return S 


def get_payoff_function_ind(strikes):
    Karr = np.array(strikes)
    sigma_a = np.array([sigma_1,sigma_2])
    def payoff_function(x_vect):
        # x_i = ln(Si) − (r - .5\sigma_i^2)*t
        # e^x_i  = Si*exp(-(r-.5sigma^2)t)
        # Si = exp( xi + (r-.5sigmai^2)t )
        S = np.exp(x_vect + 0*(r-.5*sigma_a**2)*T)
        SminusK=S-Karr
        payoff = np.amax(np.maximum(SminusK,0,),axis=0)
        return payoff

    def transform_ind_to_x(ind):
        #print(ind)
        return np.array([x[ind[0]],x[ind[1]]])

    
    def payoff_function_ind(ind):
        x_vect = transform_ind_to_x(ind)
        pf=payoff_function(x_vect)
        #print("from f:",ind,x_vect,pf)
        return pf
    
    def payoff_function_ind_batch(ind_batch):
        y = ind_batch.astype(int)
        return np.apply_along_axis(payoff_function_ind,1,y)

    def transform_qttind_to_x(qttind):
        #print(ind)
        # qttind is of shape [2,....,2] + [2,....,2] (concat of 2 lists d*[2] and d*[2])
        numAssets=2
        ind=np.zeros(2).astype(int)
        
        ind[0] = lin_ind(d*[2],qttind[0:d]) # first asset
        ind[1] = lin_ind(d*[2],qttind[d:2*d]) # second asset
        ind[0]=invInd(ind[0],d*[2])
        ind[1]=invInd(ind[1],d*[2])
        x_vect=np.array([x[ind[0]],x[ind[1]]])
        #print("0=",ind[0],"1=",ind[1],"x=",x_vect)
        return x_vect

    
    def payoff_function_qttind(qttind):
        x_vect = transform_qttind_to_x(qttind)
        pf=payoff_function(x_vect)
        #print("qtt=",qttind,"x=",x_vect,"pf=",pf)
        return pf
    
    def payoff_function_qttind_batch(ind_batch):
        y = ind_batch.astype(int)
        return np.apply_along_axis(payoff_function_qttind,1,y)

    return payoff_function_ind_batch,\
        payoff_function_qttind_batch,payoff_function


# %%
f_temp,f_qtt,_=get_payoff_function_ind([100,150])
def fit1():
    x0 = tt.rand((x.shape[0],x.shape[0]), 2, 1)
    x0 = 100.0*x0
    x1 = rect_cross.cross(f_temp, x0,nswp=15)     
    return x1

def fit1_qtt():
    x0 = tt.rand(2*d*[2], 2*d, 1)
    #x0 = 100.0*x0
    x1 = rect_cross.cross(f_qtt, x0,nswp=16)     
    return x1

def exact_payoff(Karr):
    s12 = get_spots_from_x(np.array([x,x]),r,T,sigma)
    s1=s12[0]
    s2=s12[1]
    payoff=np.zeros((2**d,2**d))
    for i,s in enumerate(s1):
        s1_fixed=s*np.ones_like(s1)
        s12f = np.array([s1_fixed,s2])
        SminusK=s12f.T - Karr*np.ones((2**d,1))
        payoff[i] = np.amax(np.maximum(SminusK,0,),axis=1)
    return payoff

exact_p=exact_payoff(np.array([100,150]))
#%%
qttPayoff = fit1_qtt()
x1_full = qttPayoff.full().reshape(2**d,2**d)
print(qttPayoff.core.shape[0])
print(np.prod(qttPayoff.full().shape))

#%%
# create exact payoff
qtt_exact_payoff = tt.vector(exact_p.reshape(2*d*[2]))

#%%

def make_qtt_2dim_rhs(dt,g):
    z=tt.vector(np.zeros(d*[2]))
    rhs=tt.kron(z,z)
    jlastInv=invInd(2**d-1,d*[2])#invInd(2**d-1,d*[2])
    e_last=unit(d*[2],j=jlastInv)
    rhs=tt.kron(e_last,(-1/dt*g))
    return rhs

def make_qtt_rhs(g):
    z=tt.vector(np.zeros(d*[2]))
    rhs=tt.kron(z,z)
    jlastInv=invInd(2**d-1,d*[2])#invInd(2**d-1,d*[2])
    e_last=unit(d*[2],j=jlastInv)
    rhs=tt.kron(e_last,(g))
    return rhs


dt_for_global = T*2**(-d)
#rhs = make_qtt_2dim_rhs(dt_for_global,qttPayoff)
rhs = make_qtt_2dim_rhs(dt_for_global,qtt_exact_payoff)
#%%


def grad_loss_f(x):
    [x2,z]=tt.amen.amen_mv(ttL,x,1e-4,verb=False)
    x2=x2-rhs
    [x2,z]=tt.amen.amen_mv(ttL,x2,1e-4,verb=False)
    return x2

x0_ = rhs
#%%
h0=1e-7
hh=h0
test_vals=[]
for i,iter in enumerate(range(1000)):

    
    print("------------------------------->iteration ",i,"step=",h)
    x0_=x0_ - hh*grad_loss_f(x0_)

    norm1=tt.amen.amen_mv(ttL,x0_,1e-4,verb=False)
    test_val=(norm1[0]-rhs).__dot__((norm1[0]-rhs))
    if len(test_vals)>0:
        print(hh,test_vals[-1],)
        if test_vals[-1]<test_val:
            hh*=0.1
            print("new h=",hh)
        print(hh,test_vals[-1],test_val)
    if (len(test_vals)>1):
        if np.abs((test_vals[-1]-test_vals[-2])/test_vals[-2])<1e-5:
            hh*=10
            print("new h=",hh)
    
    test_vals.append(test_val)

    print("test=",test_val)
    if test_val<1e-3:
        break

bs_2dim = x0_
#a=tt_min.min_func(lambda x : tt.sum((tt.matvec(ttL,x)-rhs)*(tt.matvec(ttL,x)-rhs)),-2,2,d,2)
#%%
bs_2dim,res=tt.GMRES(lambda x,eps:  tt.amen.amen_mv(ttL,x,eps,verb=False)[0],-rhs,-rhs,verbose=True,maxit=10)
#%%
bs_2dim,res=tt.GMRES( lambda x,eps:  tt.amen.amen_mv(ttL,x,eps,verb=False)[0],bs_2dim,-rhs,verbose=True,maxit=10)
#%%
x0_=bs_2dim
norm1=tt.amen.amen_mv(ttL,x0_,1e-4,verb=False)
test_val=(norm1[0]-rhs).__dot__((norm1[0]-rhs))
print(test_val)
#%%
bs_2dim=tt.ksl.ksl(ttLx,qtt_exact_payoff,T)
#%%
ttu = qtt_exact_payoff
t_steps=20
for t in range(t_steps,0,-1):
    dt=T/t_steps
    print(t)
    ttu=tt.amen.amen_solve(-dt*ttLx + _ttIII,ttu,ttu,1e-4)

#%%

class QTTSettings:
    def __init__(self,d):
        self.d=d
    
    def get_n(self):
        return 2**self.d
    
    def get_modes(self):
        return self.d*[2]

    def bits(self,nn):
        # The number of bits we need to represent the number
        num_bits = self.d
        print(num_bits)
        # The bit representation of the number
        bits1 = [ (nn >> i) & 1 for i in range(num_bits) ]
        bits1.reverse()
        # Do we need a leading zero?
        if nn < 0 or bits1[0] == 0:
            return bits1
        return bits1

    def unit(self, j=None, tt_instance=True):
        ''' Generates e_j _vector in tt.vector format
        ---------
        Parameters:
            n - modes (either integer or array)
            d - dimensionality (integer)
            j - position of 1 in full-format e_j (integer)
            tt_instance - if True, returns tt.vector;
                        if False, returns tt cores as a list
        '''
        n=self.get_modes()
        if isinstance(n, int):
            if d is None:
                d = 1
            n = n * np.ones(d, dtype=np.int32)
        else:
            d = len(n)
        if j is None:
            j = 0
        rv = []

        j = ind2sub(n, j)

        for k in range(d):
            rv.append(np.zeros((1, n[k], 1)))
            rv[-1][0, j[k], 0] = 1
        if tt_instance:
            rv = tt.vector.from_list(rv)
        return rv


    def ind2sub(self, idx):
        '''
        Translates full-format index into tt.vector one's.
        ----------
        Parameters:
            siz - tt.vector modes
            idx - full-vector index
        Note: not vectorized.
        '''
        siz = self.get_modes()
        n = len(siz)
        subs = np.empty((n))
        k = np.cumprod(siz[:-1])
        k = np.concatenate((np.ones(1), k))
        for i in range(n - 1, -1, -1):
            subs[i] = np.floor(idx / k[i])
            idx = idx % k[i]
        return subs.astype(np.int32)

    def lin_ind(self,ind):
        n0 = self.get_modes()
        d=len(n0)
        ni = float(n0[0])
        ii=ind[0]
        for i in range(1, d):
            ii+=ind[i]*ni
            ni *= n0[i]
        return ii

    def invInd(self,j,n):
        h1=ind2sub(n,j)
        #print("h1",h1)
        h1=h1[::-1]
        #print("h1r",h1)
        return lin_ind(n,h1)


class DiscretizationDimension:
    def __init__(self,x_range : list,qtt_settings : QTTSettings):
        self.x_range = x_range
        self.qtt_setting = qtt_settings
    
    def get_n(self):
        return self.qtt_setting.get_n()

    def get_modes(self):
        return self.qtt_setting.get_modes()

    def get_x_grid(self):
        a,b,c=self.get_x_step()
        return np.arange(a,b,c)

    def get_s_grid(self,s0):
        S = s0*np.exp(self.get_x_grid())
        return S 


    def get_x_step(self):
        n = self.qtt_setting.get_n()
        start1,stop1 = self.x_range
        step = (stop1-start1)/n
        return start1,stop1,step

class DiscretizationDimensionManager:
    def __init__(self,x_discretizations : list):
        self.x_dimensions = x_discretizations
    



class PDEUtils:
    def __init__(self, x_dimensions : list, t_dimension : DiscretizationDimension ):
        self.x_dimensions = x_dimensions
        self.t_dimension = t_dimension
        

    
    def get_1D_d_by_dx(self,x_dimension : DiscretizationDimension):
        n=x_dimension.get_n()
        DD=np.zeros((n,n))
        h=x_dimension.get_x_step()[2]
        DD[0,1]=1/(2*h)
        for jj in range(n-2):
            j=jj+1
            DD[j,j-2+1]=-1/(2*h)
            DD[j,j+1]=1/(2*h)
        # bd condition 2nd derivative = 0
        DD[n-1][n-1]=1/(h)
        DD[n-1][n-2]=-1/h
        modes=x_dimension.qtt_setting.get_modes()
        return tt.matrix(DD.reshape(2*modes))
    
    def get_1D_d2_by_dx2(self,x_dimension : DiscretizationDimension):
        ttDx = self.get_1D_d_by_dx(x_dimension)
        return ttDx.__matmul__(ttDx)
    
    def make_Dt(self,T):
        n = self.t_dimension.get_n()
        DD=np.zeros((n,n))
        start1,stop1 =0,T
        h = (stop1-start1)/n
    
        for j in range(n-1):
            DD[j,j]=-1/h
            DD[j,j+1]=1/h
            
        DD[n-1][n-1]=-1/h
        return DD,h

    def ttmkron(self, a : list):
        k=a[0]
        for aa in a[1:]:
            k=tt.kron(k,aa)
        return k
    
    def get_I(self,discretization : DiscretizationDimension):
        ttI = tt.eye(discretization.get_modes())
        return ttI

    def get_Z(self,discretization : DiscretizationDimension):
        z = tt.ones(2*discretization.qtt_setting.get_modes())
        z=z-z
        return z

    def get_zero_op(self):
        kron_list1 = [self.get_Z(i) for i in self.x_dimensions]
        return self.ttmkron(kron_list1)
    
    def get_I_op(self):
        kron_list1 = [self.get_I(i) for i in self.x_dimensions]
        return self.ttmkron(kron_list1)

    def get_1st_order_op(self,j,op):
        kron_list1 = [self.get_I(self.x_dimensions[i]) for i in range(0,j)]
        kron_list2 = [self.get_I(self.x_dimensions[i]) for i in range(j+1,len(self.x_dimensions))]
        op = [op]
        return self.ttmkron(kron_list1+op+kron_list2)

    def get_1st_order_d_dx(self,j):
        return self.get_1st_order_op(j,self.get_1D_d_by_dx(self.x_dimensions[j]))

    def get_2nd_order_d_dx(self,j,k):
        j1=np.minimum(j,k)
        j2=np.maximum(j,k)
        if j1==j2:
            j=j1
            return self.get_1st_order_op(j,self.get_1D_d2_by_dx2(self.x_dimensions[j]))
        
        kron_list1 = [self.get_I(self.x_dimensions[i]) for i in range(0,j1)]
        kron_list2 = [self.get_1D_d_by_dx(self.x_dimensions[j1])]
        kron_list3 = [self.get_I(self.x_dimensions[i]) for i in range(j1+1,j2)]
        kron_list4 = [self.get_1D_d_by_dx(self.x_dimensions[j2])]
        kron_list5 = [self.get_I(self.x_dimensions[i]) for i in range(j2+1,len(self.x_dimensions))]
        final_list = kron_list1+kron_list2+kron_list3+kron_list4+kron_list5   
        return self.ttmkron(final_list)


class TTBlackScholesPDEFactory:
    def __init__(self, pde : PDEUtils):
        self.pde = pde

    def new1(self,num_assets : int, sigma, corr, r):
        
        list_2nd_order = [(.5*sigma[L["j"]]*sigma[L["k"]])*L["pde"] \
            for L in [{"j":j,"k":k,"pde":self.pde.get_2nd_order_d_dx(j,k)} \
                for j in range(len(self.pde.x_dimensions)) \
                    for k in range(len(self.pde.x_dimensions))]]
        list_1st_order = [(r-.5*sigma[L['j']]**2)*L['pde'] \
            for L in [{'j':j,'pde':self.pde.get_1st_order_d_dx(j)} \
                for j in range(len(self.pde.x_dimensions))]]
        I_op  = self.pde.get_I_op()
        list_0_order = -r*I_op

        return list_0_order,list_1st_order,list_2nd_order

    
    def new_instance(self,num_assets : int, sigma, corr, r):
        
        list_2nd_order = [(.5*sigma[L["j"]]*sigma[L["k"]]*corr[L["j"],L["k"]])*L["pde"] \
            for L in [{"j":j,"k":k,"pde":self.pde.get_2nd_order_d_dx(j,k)} \
                for j in range(len(self.pde.x_dimensions)) \
                    for k in range(len(self.pde.x_dimensions))]]
        list_1st_order = [(r-.5*sigma[L['j']]**2)*L['pde'] \
            for L in [{'j':j,'pde':self.pde.get_1st_order_d_dx(j)} \
                for j in range(len(self.pde.x_dimensions))]]
        I_op  = self.pde.get_I_op()
        list_0_order = -r*I_op

        total_list = list_2nd_order + list_1st_order + [list_0_order]
        pde = total_list[0]      
        for j in total_list[1:]:
            pde+=j  
        return TTBlackScholesPDE(pde,I_op)


class TTBlackScholesPDE:
    def __init__(self, pde : tt.matrix, I_matrix : tt.matrix):
        self.pde_matrix = pde
        self.I_matrix = I_matrix

    def roll_back(self,payoff,T, t_steps):
        ttu = payoff
        ttLx = self.pde_matrix
        I_matrix = self.I_matrix
        for t in range(t_steps,0,-1):
            dt=T/t_steps
            ttu=tt.amen.amen_solve(-dt*ttLx + I_matrix,ttu,ttu,1e-4)
        return ttu

class call_payoff_2D:
    def __init__(self,x1_discretization : DiscretizationDimension,\
         x2_discretization : DiscretizationDimension):
        self.x1 = x1_discretization
        self.x2 = x2_discretization

    def get_spots_from_x(self, x_vect):
        S = np.exp(x_vect)
        return S 

    def get_payoff(self,Karr):
        x1,x2 = self.x1.get_x_grid(),self.x2.get_x_grid()
        s12 = self.get_spots_from_x(np.array([x1,x2]))
        s1=s12[0]
        s2=s12[1]
        d=self.x1.qtt_setting.d
        payoff=np.zeros((2**d,2**d))
        for i,s in enumerate(s1):
            s1_fixed=s*np.ones_like(s1)
            s12f = np.array([s1_fixed,s2])
            SminusK=s12f.T - Karr*np.ones((2**d,1))
            payoff[i] = np.amax(np.maximum(SminusK,0,),axis=1)
        return payoff
    
    def get_qtt(self,Karr):
        d = self.x1.qtt_setting.d
        exact_p = self.get_payoff(Karr)
        qtt_exact_payoff = tt.vector(exact_p.reshape(2*d*[2]))
        return qtt_exact_payoff

qtt_settings = QTTSettings(10)

x1_dim = DiscretizationDimension([-8,8],qtt_settings)
x2_dim = DiscretizationDimension([-8,8],qtt_settings)

pdeUtils = PDEUtils([x1_dim,x2_dim],None)
pdeFactory = TTBlackScholesPDEFactory(pdeUtils)
call_payoff1 = call_payoff_2D(x1_dim,x2_dim) 

Karr=np.array([100.0,150.0])
sigma = np.array([.2,.3])
corr=np.array([[1,.22],[.22,1]])
r=0.03
T=1
T_steps=20
pde = pdeFactory.new_instance(2,sigma,corr,r)
solution1=pde.roll_back(call_payoff1.get_qtt(Karr),T,T_steps)
#%%
sigma_1 ,sigma_2 = sigma
rho = corr[0,1]
ttu=solution1
x=x1_dim.get_x_grid()
spots = x1_dim.get_s_grid(1.0)
u0_full = ttu.full().reshape(2**d,2**d)
# %%

#%%
    Dt_np,dt=make_Dt(n,T)
    ttDt = tt.matrix(Dt_np.reshape(2*d*[2]))
    ttDx = tt.matrix(DD.reshape(2*d*[2]))
    ttI = tt.eye(ttDx.n)

    ttIII = ttmkron(ttI,ttI,ttI)
    ttDx1=ttmkron(ttI,ttDx,ttI)
    ttDx2=ttmkron(ttI,ttI,ttDx)

    ttDx1Dx2 = ttmkron(ttI,ttDx,ttDx)
    ttDx1Dx1 = ttmkron(ttI,ttDx.__matmul__(ttDx),ttI)
    ttDx2Dx2 = ttmkron(ttI,ttI,ttDx.__matmul__(ttDx))

    ttDt=ttmkron(ttDt,ttI,ttI) # should include final condition

    _ttIII = tt.kron(ttI,ttI)
    _ttDx1=tt.kron(ttDx,ttI)
    _ttDx2=tt.kron(ttI,ttDx)

    _ttDx1Dx2 = tt.kron(ttDx,ttDx)
    _ttDx1Dx1 = tt.kron(ttDx.__matmul__(ttDx),ttI)
    _ttDx2Dx2 = tt.kron(ttI,ttDx.__matmul__(ttDx))



    '''
    ttL=ttDt 
    ttL += .5*(2*rho*sigma_1*sigma_2*ttDx1Dx2 +\
        sigma_1*sigma_1*ttDx1Dx1+ sigma_2*sigma_2*ttDx2Dx2) 
    ttL -= r*ttIII
    '''

    ttL=ttDt 
    ttL += .5*(2*rho*sigma_1*sigma_2*ttDx1Dx2 +\
        sigma_1*sigma_1*ttDx1Dx1+ sigma_2*sigma_2*ttDx2Dx2) 
    ttL += (r-.5*sigma_1**2)*ttDx1 +  (r-.5*sigma_2**2)*ttDx2
    ttL -= r*ttIII

    ttLx = .5*(2*rho*sigma_1*sigma_2*_ttDx1Dx2 \
        +sigma_1*sigma_1*_ttDx1Dx1+ sigma_2*sigma_2*_ttDx2Dx2) \
        +(r-.5*sigma_1**2)*_ttDx1 +  (r-.5*sigma_2**2)*_ttDx2 \
        - r*_ttIII


#%%
#@@@@@@@@@@@@@@@@@@@2 Solve the pde !!! we have bc and pde.@@@@@@@@@@@@@@22222
bs_2dim = tt.amen.amen_solve(ttL,rhs,rhs,1e-16)
#%%
multislice0 =[0 for _ in range(d)] + [slice(0,2) for i in range(2*d)]  
multislice1 =[1 for _ in range(d)] + [slice(0,2) for i in range(2*d)]  
'''for j in range(0,0,-1):
    print(j)
    multislice1[d-j]=1
'''

u0=bs_2dim[multislice0]
u1=bs_2dim[multislice1]
u0_full = u0.full().reshape(2**d,2**d)
u1_full = u1.full().reshape(2**d,2**d)
#%%
p=[880,930]
#print (u0_full[p[0],p[1]],u1_full[p[0],p[1]],x1_full[p[0],p[1]])
#plt.imshow(x1_full)
#plt.show()
plt.imshow(u1_full)
plt.show()
plt.imshow(u0_full)
plt.show()
test_qmc_vs_mc(u0_full)
#%%
multislice0 =[0 for _ in range(d)] + [slice(0,2) for i in range(2*d)]  
multislice1 =[1 for _ in range(d)] + [slice(0,2) for i in range(2*d)]  
'''for j in range(0,0,-1):
    print(j)
    multislice1[d-j]=1
'''

u0=bs_2dim[multislice0]
u1=bs_2dim[multislice1]
u0_full = u0.full().reshape(2**d,2**d)
u1_full = u1.full().reshape(2**d,2**d)

p=[880,930]
#print (u0_full[p[0],p[1]],u1_full[p[0],p[1]],x1_full[p[0],p[1]])
#plt.imshow(x1_full)
#plt.show()
plt.imshow(u1_full)
plt.show()
plt.imshow(u0_full)
plt.show()

#%%
def rev_list(l):
    l.reverse()
to_show=[i for i in range(10)]+[i for i in range(900,1023,1)] #(0,1,2,3,4,5,6,7,8,9,10,100,150,200,1021,1022,1023)
indexes = [list(bits(k)) + [slice(0,2) for i in range(2*d)]  for k in to_show[::-1]]
indexes = [tuple(i) for i in indexes]
data = [bs_2dim[m].full().reshape(2**d,2**d) for m in indexes]
#plt.imshow(x1_full)
#plt.show()

for i in data:
    plt.imshow(i)
    plt.show()


#%%
to_show=(0,1,2,3,1021,1022,1023)
to_show=[i for i in range(1024,0,-50)]
to_show=to_show[::-1]
indexes = [list(bits(k)) + [slice(0,2) for i in range(2*d)]  for k in to_show]
indexes = [tuple(i) for i in indexes]
data = [bs_2dim[m].full().reshape(2**d,2**d) for m in indexes]
data_tind = list(zip(data,to_show))
fig = plt.figure()
ax = fig.gca()
#ax.set_xticks(x)
#ax.set_yticks(x)
ix1=400
x_range = [i for i,xx in enumerate(x) if xx>-2.5 and xx<9.5]
t_range = np.arange(0,T,2**(-d))

s12 = get_spots_from_x(np.array([x,x]),r,T,sigma)
s1=s12[0]
x_range = [i for i,xx in enumerate(s1) if xx>40 and xx<200]
for j in range(1,len(data),1):
    plt.scatter(s1[x_range],x1_full[ix1,x_range],label="payoff",s=4)
    for i in range(j):

        plt.scatter(s1[x_range],data[i][ix1,x_range],label="data[{},t={}]".format(i,t_range[data_tind[i][1]]),s=1)
        #plt.scatter(x,data[-1][ix1,:],label="data[-1]",s=1)
    #plt.legend()
    plt.grid()
    plt.show()
# %%
u0_full=ttu.full().reshape(2**d,2**d)
# %%
u0_full=ttu.full().reshape(2**d,2**d)
# %%
u0_full=ttu.full().reshape(2**d,2**d)
# %%
plt.imshow(u0_full)
plt.show()
#%%
####################################
# test fit against MC              #
####################################

bsma1 = BlackScholesMCMA()
strikes=[100,150]
x1i=[510,800]
sigma_a = [sigma_1,sigma_2]
spots = get_spots_from_x(x[x1i],r,T,np.array(sigma_a))
print("spots=",spots)
bsMA_input = BlackScholesMCInputMA(spots,strikes,r,T,sigma_a,np.array([[1,rho],[rho,1]]),10000)
print("MC price",bsma1.price(bsMA_input))
print("qtt price",u0_full[x1i[0],x1i[1]])
#%%

# plot QMC vs QTT
def test_qmc_vs_mc(u0_full,flip=False):
    test_index=850
    
    def get_inp_with_changing_x(i,TTT,flip=False):
        x1i=[i,test_index]
        if flip:
            x1i=[test_index,i]
        spots = get_spots_from_x(x[x1i],r,TTT,np.array(sigma_a))
        return BlackScholesMCInputMA(spots,strikes,r,TTT,sigma_a,np.array([[1,rho],[rho,1]]),10000)

    y_mc1 = [bsma1.price(get_inp_with_changing_x(i,0,flip)) for i in range(len(x))]
    y_mc2 = [bsma1.price(get_inp_with_changing_x(i,T,flip)) for i in range(len(x))]
    fig = plt.figure()
    ax = fig.gca()
    s12 = get_spots_from_x(np.array([x,x]),r,T,sigma)
    s1=s12[0]
    s2=s12[1]
    x_range = [i for i,xx in enumerate(s1) if xx>80 and xx<180]
    plt.scatter(s1[x_range],np.array(y_mc1)[x_range],label="MC1 "+str(np.exp(x[test_index])),s=5)
    plt.scatter(s1[x_range],np.array(y_mc2)[x_range],label="MC2"+str(np.exp(x[test_index])),s=5)
    yqtt=u0_full[x_range,test_index] if not flip else u0_full[test_index,x_range]
    #yqtt1=u1_full[x_range,test_index] if not flip else u1_full[test_index,x_range]
    
    k=0
    m=list(bits(k)) + [slice(0,2) for i in range(2*d)]
    #uu_full=bs_2dim[tuple(m)].full().reshape(2**d,2**d)
    #yqtt2=uu_full[x_range,test_index] if not flip else uu_full[test_index,x_range]
    
    plt.scatter(s1[x_range],yqtt,label="data[0] "+str(np.exp(x[test_index])),s=2)
    #plt.scatter(s1[x_range],yqtt1,label="data[1]",s=2)
    #plt.scatter(s1[x_range],yqtt2,label="data[k]",marker='*',s=12)
    plt.legend()
    plt.grid()
    plt.show()

test_qmc_vs_mc(u0_full,False)
test_qmc_vs_mc(u0_full,True)
# %%

#%%


# %%

#%%
data
#%%
z = np.zeros((2**d,2**d))
for i in range(2**d):
    for j in range(2**d):
        z[i,j]=f_temp(np.array([(i,j)]))
# %%
#np.argmax
agmax=np.argmax(np.abs(z-x1_full))
print(np.max(np.abs(z-x1_full)))
print(z.reshape(-1)[agmax],x1_full.reshape(-1)[agmax])
iii=ind2sub((1024,1024),invInd(agmax,(1024,1024)))
print(iii)
# %%
x1_full
# %%
z
# %%
z.shape
# %%
test_pt=(880,930)
test_pt=(iii[0],iii[1])
test_pt=(945,950)

a=get_payoff_function_ind([100,150])
x_ve=np.array([x[test_pt[0]],x[test_pt[1]]])
print(x_ve,a[2](x_ve),f_temp(np.array([test_pt])))
print(z[test_pt],x1_full[test_pt])

# %%
