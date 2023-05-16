#%%
import itertools
from numpy.core.shape_base import atleast_1d
from numpy.lib import index_tricks
from tt.core.vector import vector
import tt

import sys
sys.path.append('../')
import numpy as np
import tt
import random
import scipy.stats as si
from scipy import interpolate
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
from collections import namedtuple


def my_bunchify(name,d):
    d_named = namedtuple(name, d)(**d)
    return d_named



class TrainBSData:
    def __init__(self,strikes,ts,Ss,sigmas,rs):
        self.Strikes = strikes
        self.Ts = ts
        self.Ss = Ss
        self.sigmas = sigmas
        self.rs = rs
        self.A = [self.Ss,self.Strikes,self.Ts,self.rs,self.sigmas]
        self.shape = [len(x) for x in self.A]
    






def get_spots_from_x(x_vect,r,T,sigma):
    S = np.exp(x_vect)
    return S 

def get_bits_func(d):
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
    return bits


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


class BlackScholesMCInputMAFactory:
    def new_from_old(self,inp):
        return BlackScholesMCInputMA(
            inp.S,
            inp.K,
            inp.r,
            inp.T,
            inp.sigma,
            inp.C,
            inp.N
        )
    
    def new_update_S(self,inp,j,s):
        s1=inp.S.copy()
        s1[j]=s
        return BlackScholesMCInputMA(
            s1,
            inp.r,
            inp.K,
            inp.T,
            inp.sigma,
            inp.N,
            inp.C
        )
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
        #print(inp.N)
        w=np.random.multivariate_normal(mean,cov,size=inp.N) 
        S = np.array(inp.S)*np.exp(w)
        Karr=np.array(inp.K)
        SminusK=S-Karr
        payoff = np.amax(np.maximum(SminusK,0,),axis=1)
        pv = np.mean(payoff)*np.exp(-inp.r*inp.T)
        return pv

class BlackScholesMCMAGeneric:

    def __init__(self,payoff):
        self.payoff = payoff

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
        #print("S=",S)
        payoff = self.payoff(S)
        #print("payoff=",payoff)
        pv = np.mean(payoff)*np.exp(-inp.r*inp.T)
        return pv
    def apply_fs(self,fs,x):
        if len(x.shape)>=2:
            #print(fs)
            #print(x)
            ys = np.array([f(x[:,j]) for j,f in zip(range(x.shape[1]),fs)]).T    
            return ys
        else:
            ys = np.array([f(x[j]) for j,f in zip(range(x.shape[0]),fs)]).T    
            return ys

    
    def pricelv(self,inp : BlackScholesMCInputMA,Nt):
        lv1 = inp.sigma
        def lv(s):
           return self.apply_fs(lv1,s)

        nAssets=len(inp.S)
        rr = inp.r*np.ones(nAssets)
        meanz=np.zeros(nAssets)
        #mean = meanz+ (rr - .5*np.array(inp.sigma)**2)*inp.T
        #cov = inp.C
        #sigs=np.concatenate([sig*np.ones([nAssets,1]) for sig in inp.sigma],axis=1)
        #sigs2=np.concatenate([sig*np.ones([nAssets,1]) for sig in inp.sigma],axis=1)
        #cov=cov*sigs*sigs.T*inp.T
        #w=np.random.multivariate_normal(meanz,inp.C,size=inp.N) 
        #w2=np.random.multivariate_normal(mean,cov,size=inp.N) 
                
        S = np.array(inp.S)
        newSigma1 = lv(S)
        #newSigma1=inp.sigma
        dt =  inp.T/(Nt+0.0)
        #print(dt)
        for i in range(Nt):
            w=np.random.multivariate_normal(meanz,inp.C,size=inp.N) 
            nssig=np.array(newSigma1)
            Snext=S*np.exp((rr - .5*nssig**2)*dt +nssig*np.sqrt(dt)*w)
            S=Snext
            newSigma1 = lv(S)
                
        #ssig=np.array(inp.sigma)
        #S = np.array(inp.S)*np.exp((rr - .5*ssig**2)*inp.T +ssig*np.sqrt(inp.T)*w)
        #S2 = np.array(inp.S)*np.exp(w2)
        #Karr=np.array(inp.K)
        #SminusK=S-Karr
        #SminusK2=S2-Karr
        #payoff = np.amax(np.maximum(SminusK,0,),axis=1)
        #payoff2 = np.amax(np.maximum(SminusK2,0,),axis=1)
        payoff = self.payoff(S)
        pv = np.mean(payoff)*np.exp(-inp.r*inp.T)
        #pv2 = np.mean(payoff2)*np.exp(-inp.r*inp.T)
        return pv#,pv2#,vvv#,w,newSigma,S,Snext


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

    def ind2subi(self, idx):
        return self.ind2sub(idx)[::-1]

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
        h1=self.ind2sub(n,j)
        #print("h1",h1)
        h1=h1[::-1]
        #print("h1r",h1)
        return self.lin_ind(n,h1)


class DiscretizationDimension:
    def __init__(self,x_range : list,qtt_settings : QTTSettings):
        self.x_range = x_range
        self.qtt_setting = qtt_settings
    
    def get_n(self):
        return self.qtt_setting.get_n()

    def get_modes(self):
        return self.qtt_setting.get_modes()

    def get_x_grid(self) -> np.array:
        a,b,c=self.get_x_step()
        return np.arange(a,b,c)

    def get_s_grid(self,s0) -> np.array:
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

    def get_1st_order_op(self,j,op,opname="L"):
        kron_list1 = [self.get_I(self.x_dimensions[i]) for i in range(0,j)]
        kron_list2 = [self.get_I(self.x_dimensions[i]) for i in range(j+1,len(self.x_dimensions))]
        op = [op]

        kron_list1_ = ["I" for i in range(0,j)]
        kron_list2_ = ["I" for i in range(j+1,len(self.x_dimensions))]
        op_ = [opname]
        ll = kron_list1_+op_+kron_list2_
        print(ll)

        return self.ttmkron(kron_list1+op+kron_list2)

    def get_1st_order_d_dx(self,j):
        return self.get_1st_order_op(j,self.get_1D_d_by_dx(self.x_dimensions[j]),'d/dx')

    def get_2nd_order_d_dx(self,j,k):
        j1=np.minimum(j,k)
        j2=np.maximum(j,k)
        if j1==j2:
            j=j1
            return self.get_1st_order_op(j,self.get_1D_d2_by_dx2(self.x_dimensions[j]),"d2/dx2")
        
        kron_list1 = [self.get_I(self.x_dimensions[i]) for i in range(0,j1)]
        kron_list2 = [self.get_1D_d_by_dx(self.x_dimensions[j1])]
        kron_list3 = [self.get_I(self.x_dimensions[i]) for i in range(j1+1,j2)]
        kron_list4 = [self.get_1D_d_by_dx(self.x_dimensions[j2])]
        kron_list5 = [self.get_I(self.x_dimensions[i]) for i in range(j2+1,len(self.x_dimensions))]
        final_list = kron_list1+kron_list2+kron_list3+kron_list4+kron_list5   

        kron_list1_ = ["I" for i in range(0,j1)]
        kron_list2_ = ["d/dx"]
        kron_list3_ = ["I" for i in range(j1+1,j2)]
        kron_list4_ = ["d/dx"]
        kron_list5_ = ["I" for i in range(j2+1,len(self.x_dimensions))]
        final_list_ = kron_list1_+kron_list2_+kron_list3_+kron_list4_+kron_list5_   
        print(final_list_)
        return self.ttmkron(final_list)


class TTBlackScholesPDEFactory:
    def __init__(self, pde : PDEUtils):
        self.pde = pde

    def new_instance(self,num_assets : int, sigma1, corr, r,epsilon=1e-4):
        
        I_op  = self.pde.get_I_op()
        sigma = [self.pde.get_1st_order_op(j,s.get_qtt_matrix(),'sig') for j,s in enumerate(sigma1)]
        list_2nd_order = [(.5*sigma[L["j"]]*sigma[L["k"]]*corr[L["j"],L["k"]])*L["pde"] \
            for L in [{"j":j,"k":k,"pde":self.pde.get_2nd_order_d_dx(j,k)} \
                for j in range(len(self.pde.x_dimensions)) \
                    for k in range(len(self.pde.x_dimensions))]]

        list_1st_order = [(r*I_op-.5*sigma[L['j']]*sigma[L['j']])*L['pde'] \
            for L in [{'j':j,'pde':self.pde.get_1st_order_d_dx(j)} \
                for j in range(len(self.pde.x_dimensions))]]
        list_0_order = -r*I_op

        total_list = list_2nd_order + list_1st_order + [list_0_order]
        #random.shuffle(total_list)
        #pde = self.pde.ttmkron(len(self.pde.x_dimensions)*[self.pde.get_Z(self.pde.x_dimensions[0])]) #total_list[0]      
        pde = total_list[0]      
        for j in total_list[1:]:
            total_list[0]=total_list[0]+j
        #pde=pde-total_list[0]
        #pde=pde+total_list[0]
        #print(pde)
        print(total_list[0])
        print(total_list[0].erank)
        t = total_list[0].round(1e-12)
        print('----------------------- reduced rank ----------------------')
        print(t)
        print(t.erank)
        return TTBlackScholesPDE(t,I_op,epsilon)


class QTTWrapper:
    def __init__(self,qtt_vector,x_dim : list):
        self.qtt_vector=qtt_vector
        self.x_dim = x_dim
        
    def getv(self,i):
        k=np.concatenate([\
            self.x_dim[j].qtt_setting.ind2subi(i[j]) for j in range(len(self.x_dim))\
                ])
        print(k)
        return self.qtt_vector[k]

    def __getitem__(self,i):
        return self.qtt_vector[np.concatenate([\
            self.x_dim[j].qtt_setting.ind2subi(i[j]) for j in range(len(self.x_dim))\
                ])]
    

    
class TTBlackScholesPDE:
    def __init__(self, pde : tt.matrix, I_matrix : tt.matrix, epsilon : float):
        self.pde_matrix = pde
        self.I_matrix = I_matrix
        self.epsilon=epsilon

    def roll_back(self,payoff,T, t_steps):
        ttu = payoff
        ttLx = self.pde_matrix
        I_matrix = self.I_matrix
        for t in range(t_steps,0,-1):
            print("rollback: "+str(t))
            dt=T/t_steps
            ttu=tt.amen.amen_solve(-dt*ttLx + I_matrix,ttu,ttu,self.epsilon,verb=1)
            #print("solution now\n-----------------------\n")
            #print(ttu)
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


class ListDiscretizationDimension:
    def __init__(self,items):
        self.items=items
    
    def __getitem__(self,j)->DiscretizationDimension:
        return self.items[j]
    
    def get_len(self):
        return len(self.items)

    def get_range(self):
        return range(len(self.items))
    
    def get_index_shape(self):
        n=[]
        shapes=[self.__getitem__(i).get_modes()  for i in self.get_range()]
        for i in shapes:
            n+=i
        return np.array(n)

def index_util(x_dim : ListDiscretizationDimension):
    def index_to_var(qtt_ind):
        offset=0
        sl1=[-1]
        j=0
        s1={}
        sp1=[]
        for i in x_dim.get_range():
            x1_dim = x_dim[i]
            #sl1=[i for i in range(len(x1_dim.get_modes()))]
            sl1=[i for i in range(sl1[-1]+1,sl1[-1]+1+len(x1_dim.get_modes()))]
            s1[i] = qtt_ind[:,sl1]
            spots1=x1_dim.get_s_grid(1)
            ls1=[x1_dim.qtt_setting.lin_ind(s1[i][j][::-1]) for j in range(s1[i].shape[0])]
            sp1.append(spots1[ls1])
            res = np.array(sp1).T
            j+=1
        #print(res)
        return res
    
        
    return index_to_var


def get_payoff_2dcall(strikes):
    def function_of_var(x):
        payoff = np.maximum(.5*(x[:,0]+x[:,1])-strikes,0)
        return payoff

    return function_of_var

def get_smooth_max(alpha):
    def smooth_max(a,b):
        e1=np.exp(alpha*a)
        e1i=np.exp(-alpha*a)
        return a/(e1i+1)
    return smooth_max

def smooth_max3(a,b):
    
    if a<0:
        v=0.
    elif a<(.3)**(1./3.):#a<0.6299605249474366:
        v=a**3+0.0 
    else: 
        v=a+0.0
    return v


class VolatilityFunctionManager:
    def __init__(self,funcs_of_s):
        self.funcs_of_s = funcs_of_s
    
    def apply(self,s):
        pass

class VolatilityFunction:
    def __init__(self, x1_dim : DiscretizationDimension, func_of_s):
        self.func_of_s=func_of_s
        self.x1_dim = x1_dim
    
    def get_func_of_s(self):
        return self.func_of_s
    
    def get_qtt_matrix(self):
        x1_dim = self.x1_dim
        list_x_dim = ListDiscretizationDimension(1*[x1_dim])
        idx_to_var = index_util(list_x_dim)
        qttApx1=QTTFunctionAproximation(self.func_of_s,idx_to_var,list_x_dim.get_index_shape())
        return qttApx1.get_qtt_matrix()




def get_payoff_anydcall(strikes,max_fun):
    def function_of_var(x):
        payoff = max_fun((np.sum(x,axis=1)/x.shape[1]-strikes),0)
        return payoff
    return function_of_var

def get_nomax_payoff_anydcall(strikes):
    def function_of_var(x):
        payoff = (np.sum(x,axis=1)/x.shape[1]-strikes)
        return payoff
    return function_of_var

class QTTMaxofApproximation:
    def __init__(self,func_of_var,index_to_var,index_shape):
        self.function_of_variables = func_of_var
        self.index_to_var = index_to_var
        self.index_shape=index_shape
    
    def func_of_ind(self,x):
        y = x.astype(int)
        return self.function_of_variables(self.index_to_var(y))
    
    def payoff_function_ind_batch(self,ind_batch):
        y = ind_batch.astype(int)
        return np.apply_along_axis(self.func_of_ind,1,y)

    def get_maxf(self):
        def maxf(x):
            return np.maximum(x,0)
        return np.vectorize(maxf)

    def payoff(self,x):
        return self.get_maxf()(self.function_of_variables(x))
    
    def train(self,nswp=10):
        n = self.index_shape
        x0 = tt.rand(n)
        print("First doing cross for payoff")
        x1 = rect_cross.cross(self.func_of_ind, x0, nswp=nswp, kickrank=1, rf=2)
        X=[x1]
        print("Second fitting max of ")
        x1= tt.multifuncrs2(X,self.get_maxf(),nswp=nswp, eps=1E-10)
        return x1

    def train2(self,nswp=10,guess=None):
        def testf(x):
            y1=x[:,0]
            y2=x[:,1]
            y=.5*(np.power(y1,.5)+y2)
            return y
        n = self.index_shape
        x0 = tt.rand(n)
        print("First doing cross for payoff")
        x1 = rect_cross.cross(self.func_of_ind, x0, nswp=nswp, kickrank=1, rf=2)
        X=[x1*x1,x1]
        print("Second fitting max of ")
        b=tt.multifuncrs2(X,testf,nswp=nswp,eps=1e-10,y0=guess)
        def payoff(x):
            y2=self.function_of_variables(x)
            p=.5*(np.power(y2*y2,.5) + y2)
            return p

        return b, payoff
    
    def train3(self,nswp=10):
        def testf(x):
            y1=x[:,0]
            #y2=x[:,1]
            y=np.power(0.01+y1,.5)
            return y
        n = self.index_shape
        x0 = tt.rand(n)
        print("First doing cross for payoff")
        x1 = rect_cross.cross(self.func_of_ind, x0, nswp=nswp, kickrank=1, rf=2)
        X=[x1*x1]
        print("Second fitting max of ")
        b=tt.multifuncrs2(X,testf,nswp=nswp,eps=1e-10)
        def payoff(x):
            y2=self.function_of_variables(x)
            p=.5*(np.power(y2*y2+0.01,.5) + y2)
            return p
        return .5*(b+x1),payoff
    
    def train4(self,nswp=10,lamd=10.0,guess=None):
        def testf(x):
            y1=x[:,0]
            #y2=x[:,1]
            y=np.power(lamd+y1,.5)
            return y
        n = self.index_shape
        x0 = tt.rand(n)
        print("First doing cross for payoff")
        x1 = rect_cross.cross(self.func_of_ind, x0, nswp=nswp, kickrank=1, rf=2)
        X=[x1*x1]
        print("Second fitting max of ")
        b=tt.multifuncrs2(X,testf,nswp=nswp,eps=1e-10,y0=guess)
        def payoff(x):
            y2=self.function_of_variables(x)
            p=.5*(np.power(y2*y2+lamd,.5) + y2)
            return p
        return .5*(b+x1),payoff

    def train5(self,nswp=10,lamd=10.0,guess=None):
        def testf(x):
            y1=x[:,0]
            #y2=x[:,1]
            y=(y1>lamd)*y1*(lamd-y1)-.5*(lamd**2-y1**2)
            return y
        n = self.index_shape
        x0 = tt.rand(n)
        print("First doing cross for payoff")
        x1 = rect_cross.cross(self.func_of_ind, x0, nswp=nswp, kickrank=1, rf=2)
        X=[x1]
        print("Second fitting max of ")
        b=tt.multifuncrs2(X,testf,nswp=nswp,eps=1e-10,y0=guess)
        def payoff(x):
            y2=self.function_of_variables(x)
            p=np.maximum(y2,0)
            return p
        return b,payoff
    
    def get_train6MCPayoff(self,num_terms=100,nswp=10,lamd=400.0):
        def ffur(t1,K,M):
            #print(t1)
            t=1/M*(np.pi/2)*t1
            y=np.zeros_like(t)
            for k in [i+1 for i in range(K)]:
                y+=np.cos((2*k-1)*t)/(2*k-1)**2
            #y11=-4/np.pi*np.sum([np.cos((2*k-1)*t)/(2*k-1)**2 for k in [i+1 for i in range(K)]])
            y=-4/np.pi*y
            y=y+np.pi/2.*np.ones_like(y)#y11+np.pi/2.
            y1=M/(np.pi/2)*y
            return y1
        def payoff(x):
            y2=self.function_of_variables(x)
            p=.5*(ffur(y2,num_terms,lamd) + y2)
            return p
        return payoff

    def train6(self,num_terms=100,nswp=10,lamd=400.0,guess=None):
        
        def fur(t,K,M):
            def get_cosf(k):
                def cosf(xx):
                    return 4/np.pi*np.cos((2*k-1)*xx)/(2*k-1)**2
                return cosf
            t_orig = t.copy()
            t=(1/M)*(np.pi/2)*t
            list_b=[]
            y=np.pi/2.*tt.ones(t.n)
            list_b.append(np.pi/2.*tt.ones(t.n))
            t=[t]
            for k1 in range(K):
                k=k1+1
                cosf=get_cosf(k)
                b=tt.multifuncrs2(t,cosf,nswp=nswp,eps=1e-10,y0=guess)
                list_b.append(-1*b.copy())
                r_new = sum([np.prod(x.shape) for x in b.to_list(b)])
                print("b",r_new)
                y=y-b
                y=y.round(1e-10) 
                r_new = sum([np.prod(x.shape) for x in y.to_list(y)])
                print("y",r_new,y.erank)
            l_b=[]
            for i in range(len(list_b)):
                x=M/(np.pi/2)*list_b[i]
                x=.5*x
                l_b.append(x)
            l_b.append(.5*t_orig)

            y=M/(np.pi/2)*y
            y=y.round(1e-10) 
            return y,l_b
                
        
        n = self.index_shape
        x0 = tt.rand(n)
        print("First doing cross for payoff")
        x1 = rect_cross.cross(self.func_of_ind, x0, nswp=nswp, kickrank=1, rf=2)
        X=x1
        print("Second fitting max of ")
        b,list_b=fur(X,num_terms,lamd)
        def ffur(t1,K,M):
            #print(t1)
            t=1/M*(np.pi/2)*t1
            y=np.zeros_like(t)
            for k in [i+1 for i in range(K)]:
                y+=np.cos((2*k-1)*t)/(2*k-1)**2
            #y11=-4/np.pi*np.sum([np.cos((2*k-1)*t)/(2*k-1)**2 for k in [i+1 for i in range(K)]])
            y=-4/np.pi*y
            y=y+np.pi/2.*np.ones_like(y)#y11+np.pi/2.
            y1=M/(np.pi/2)*y
            return y1
        def payoff(x):
            y2=self.function_of_variables(x)
            p=.5*(ffur(y2,num_terms,lamd) + y2)
            return p
        return .5*(b+x1),payoff,list_b


class QTTFunctionAproximation:
    def __init__(self,func_of_var,index_to_var,index_shape):
        self.function_of_variables = func_of_var
        self.index_to_var = index_to_var
        self.index_shape=index_shape

    def func_of_ind(self,x):
        y = x.astype(int)
        return self.function_of_variables(self.index_to_var(y))

    def train(self,nswp=10):
        n = self.index_shape
        x0 = tt.rand(n)
        x1 = rect_cross.cross(self.func_of_ind, x0, nswp=nswp, kickrank=1, rf=2)
        return x1

    def payoff_function_ind_batch(self,ind_batch):
        y = ind_batch.astype(int)
        return np.apply_along_axis(self.func_of_ind,1,y)

    def exact_payoff(self,x_dims):
        
        payoff=np.zeros( tuple([x.get_n() for x in x_dims]) )
        for x in itertools.product(*[enumerate(x.get_s_grid(1)) for x in x_dims]):
            ind = []
            pt = []
            for i in x_dims.get_range():
                ind.append(x[i][0])
                pt.append(x[i][1])
            payoff[tuple(ind)] = self.function_of_variables(np.array(pt).reshape(1,-1))

        return tt.vector(payoff.reshape(x_dims.get_index_shape()))
    
    def get_qtt_matrix(self):
        trained=self.train()
        q2 = tt.diag(trained)
        return q2





class FixedIndexManager:
    def __init__(self,num_axis,fixed_axis,fixed_index):
        self.num_axis = num_axis
        self.fixed_axis = fixed_axis  
        self.fixed_index = fixed_index
    
    def get(self,ii):
        t=self.num_axis*[0]
        for i in range(len(self.fixed_axis)):
            t[self.fixed_axis[i]]=self.fixed_index[i]
        non_fixed_axis =list(set(range(self.num_axis)).difference(set(self.fixed_axis)))
        non_fixed_axis.sort()
        for i in range(len(non_fixed_axis)):
            t[non_fixed_axis[i]]=ii[i]
        return t


class SumOfTenosrs:
    def __init__(self,tensor_list,max_terms=None):
        self.tensor_list = tensor_list
        self.max_terms = max_terms
    def __getitem__(self,j):
        if self.max_terms == None:
            return np.sum([k[j] for k in self.tensor_list])
        else:
            return np.sum([k[j] for e,k in enumerate(self.tensor_list) if e<self.max_terms])+self.tensor_list[-1][j]

class TestExampleListPayoff:
    def __init__(self,d,r,sigma,corr,T,T_steps,qttpayoff,mcLoops):
        self.d=d
        self.r=r
        self.sigma=sigma
        self.corr=corr
        self.T=T
        self.T_steps = T_steps
        self.qttpayoff=qttpayoff
        self.N = mcLoops

    def get_num_assets(self):
        return self.sigma.shape[0]

    def get_from_old(self,setup,transform):
        return my_bunchify('SetUp',{'x_dim':setup.x_dim,"sol":transform(setup.sol),'pde':setup.pde})        

    def get_setup(self,epsilon=1e-4):
        d=self.d
        qtt_settings = QTTSettings(d)
        x_dims=[DiscretizationDimension([-3,6],qtt_settings) for _ in range(self.get_num_assets())]
    
        pdeUtils = PDEUtils(x_dims,None)
        pdeFactory = TTBlackScholesPDEFactory(pdeUtils)
        
        
        sigma = self.sigma 
        corr= self.corr
        r=self.r
        T=self.T
        T_steps=self.T_steps
        pde = pdeFactory.new_instance(3,sigma,corr,r,epsilon)
        solutions=[]
        for en,q in enumerate(self.qttpayoff):
            print("iteration ",en)
            solutions.append(pde.roll_back(q,T,T_steps))

        ####################################
        # test fit against MC              #
        ####################################
        return my_bunchify('SetUp',{'x_dim':x_dims,"sol":SumOfTenosrs(solutions),'pde':pde})



class TestExample:
    def __init__(self,d,r,sigma,corr,T,T_steps,qttpayoff,mcLoops):
        self.d=d
        self.r=r
        self.sigma=sigma
        self.corr=corr
        self.T=T
        self.T_steps = T_steps
        self.qttpayoff=qttpayoff
        self.N = mcLoops

    def get_num_assets(self):
        return self.sigma.shape[0]

    def get_setup(self,epsilon=1e-4):
        d=self.d
        qtt_settings = QTTSettings(d)
        x_dims=[DiscretizationDimension([-3,6],qtt_settings) for _ in range(self.get_num_assets())]
        #x1_dim = DiscretizationDimension([-8,8],qtt_settings)
        #x2_dim = DiscretizationDimension([-8,8],qtt_settings)
        #x_dims=[x1_dim,x2_dim]

        pdeUtils = PDEUtils(x_dims,None)
        pdeFactory = TTBlackScholesPDEFactory(pdeUtils)
        
        
        sigma = self.sigma 
        corr= self.corr
        r=self.r
        T=self.T
        T_steps=self.T_steps
        pde = pdeFactory.new_instance(3,sigma,corr,r,epsilon)
        solution1=pde.roll_back(self.qttpayoff,T,T_steps)

        ####################################
        # test fit against MC              #
        ####################################
        ttu=solution1
        u0_full = QTTWrapper(ttu,x_dims) #ttu.full().reshape(2**d,2**d)
        return my_bunchify('SetUp',{'x_dim':x_dims,"sol":solution1,"sol_full":u0_full,'pde':pde})

    def test_single(self,payoff,x1_dim,indexes=[90,113]):
        bsMA_input = self.get_bsma_input(x1_dim,indexes)
        bsma1 = BlackScholesMCMAGeneric(payoff)
        pr=0.0#bsma1.price(bsMA_input)
        return pr,bsMA_input

    def random_test(self,payoff,number_examples = 14):
        index_test_set = [tuple([np.random.randint(0,x) for x in [np.product(i.get_modes()) for i in su.x_dim]]) for _ in range(number_examples)]
        for index_test in index_test_set:
            test_result=self.test_single(payoff,x1_dim,[*index_test])
            print("spots= ",test_result[1].S, "\ntt solution= ",su.sol_full[index_test],"\nMC solution= ",test_result[0],"\n-------------")        

    def get_bsma_input(self,x1_dim,x1i):
        r=self.r
        T=self.T
        x=x1_dim.get_x_grid()
        spots = x1_dim.get_s_grid(1.0)
        rho = self.corr[0,1]
        strikes=None
        sigma_a = [s.get_func_of_s() for s in self.sigma]
        spots = get_spots_from_x(x[x1i],r,T,np.array(sigma_a))
        bsMA_input = BlackScholesMCInputMA(spots,strikes,r,T,sigma_a,self.corr,self.N)
        return bsMA_input

    def plot(self,x_dim,solution1,bsMA_input,fixed_axis,fixed_index,payoff,s_range=[20,380],mc_repeats=1):
        x1_dim, *other = x_dim
        num_assets = self.get_num_assets()
        fi=FixedIndexManager(num_assets,fixed_axis,fixed_index)
        bsma1 = BlackScholesMCMAGeneric(payoff)
        bs_inp_facotry = BlackScholesMCInputMAFactory()
        inp_test = bs_inp_facotry.new_from_old(bsMA_input)
        u0 = QTTWrapper(solution1,x_dim)
        s_grid=x1_dim.get_s_grid(1.0)
        inp_test.S[fixed_axis]=s_grid[fixed_index]
        for fiii in fixed_index:
            print ("S"+str(fiii)," ",s_grid[fiii])

        x_range = [i for i,xx in enumerate(s_grid) if xx>s_range[0] and xx<s_range[1]]
        def input_generator():
            for i in range(len(s_grid)):
                inp1 = bs_inp_facotry.new_from_old(inp_test)
                ffi=fi.get([i])
                #print(ffi)
                inp1.S=s_grid[ffi]
                yield inp1
        x_mc1 = [inp_gen for inp_gen in input_generator()]
        y_mc1 = [(m,[bsma1.pricelv(inp_gen,self.T_steps) for inp_gen in x_mc1]) for m in range(mc_repeats)]
        fig = plt.figure()
        ax = fig.gca()

        x_plot=s_grid[x_range]
        y_pde=np.array([u0[fi.get([i])] for i  in range(len(s_grid))])[x_range]
        y_mc=[(r,np.array([y_mc11[i] for i  in range(len(s_grid))])[x_range]) for r,y_mc11 in y_mc1]
        plt.scatter(x_plot,y_pde,label="PDE solution",s=29,marker = '*')
        for r,y_mc1 in y_mc:
            plt.scatter(x_plot,y_mc1,label = "MC price "+str(r),s=7,marker='.')
        plt.legend()
        plt.grid()
        plt.show()
        return x_plot,y_mc,y_pde,[x_mc1[i] for i in x_range]

def total_mem(qttpayoff):
    return sum([np.prod(x) for x in [x.shape for x in tt.vector.to_list(qttpayoff)]])

def find_nearest(array, value):
    array = np.asarray(array)
    
    idx = (np.abs(array - value)).argmin()
    return idx

class QTTPayoff:
    def __init__(self,s,qttTensor):
        self.s = s
        self.qttTenosr=qttTensor
    
    def payoff(self,s):
        # convert s to index set
        i=[find_nearest(self.s,si) for si in s]
        print(i)
        return self.qttTenosr[i]

class FindNearest:
    def __init__(self,arr):
        self.arr = arr

    def find_n(self,v):
        array = np.asarray(self.arr)
        idx = (np.abs(array - v)).argmin()
        return idx
    def getf(self):
        return np.vectorize(self.find_n)

class QTTPayoff:
    def __init__(self,qttTensor,find_n):
        self.qttTenosr=qttTensor
        self.find_n=find_n
    

    def payoff(self,s):
        # convert s to index set
        #print("payoff",s.shape)
        #ii=[]
        # for ss in s:
        #    i=[find_nearest(self.s,si) for si in ss]
        #    ii.append(self.qttTenosr[np.array(i)])
        
        idxs=self.find_n(s)
        
        return np.array([self.qttTenosr[idx] for idx in idxs])
        
        #return x(s)#np.array(ii)


def extrapolate_qtt(qttpayoff_l,idx_test,t=0.0001):
    l=list(qttpayoff_l.keys())
    l.sort()
    vals = np.array([qttpayoff_l[k][idx_test] for k in l])
    if len(l)>1:
        extr_f = interpolate.interp1d(np.array(l)\
            ,vals,kind='quadratic',\
                fill_value='extrapolate')
        extr=extr_f(t)
        return extr
    else:
        return vals[0]

def extrapolate(x,t=0.0001):
    x=np.array([[x]])
    xm=payoff_nomax(x)
    v15=.5*((xm**2+15.)**.5+xm)
    v10=.5*((xm**2+10.)**.5+xm)
    v05=.5*((xm**2+5.0)**.5+xm)
    v2p5=.5*((xm**2+2.5)**.5+xm)
    extr_f = interpolate.interp1d(np.array([5.0,10.0,15.0])\
        ,np.array([v05,v10,v15]).squeeze(),kind='quadratic',\
            fill_value='extrapolate')
    extr=extr_f(t)
    return x,xm,extr,v2p5,v05,v10,extr_f

def plot_extr(x,xm,extr,v2p5,v05,v10):
    
    plt.scatter(x,extr,label="Extrapolation",s=11,marker = '*')
    plt.scatter(x,v05,label="v05",s=1,marker = '*')
    plt.scatter(x,v2p5,label="v2p5",s=1,marker = '*')
    plt.scatter(x,np.maximum(0,xm),label = "MAX",s=20,marker='o')
    plt.legend()
    plt.grid()
    plt.show()


def print_stats(x,xm,extr,v2p5,v05,v10):
    print("manual 10   :",v10)
    print("manual 05   :",v05)
    print("extrapolated:",extr)
    print("manual 00   :",.5*((xm**2+0)**.5+xm))
    print("payoff      :",payoff(x.reshape(-1,1)))
    print("max         :",np.maximum(0,xm))

def test_extr (x,xm,extr,v2p5,v05,v10):
    r1=[e for e in extr if e<=0]
    r2=[e for e in list((zip(list(extr),list(v2p5),list(np.maximum(0,xm))))) if np.abs(e[2]-e[0])>np.abs(e[2]-e[1]) ]
    status = len(r1)==0 and len(r2)==0

    a=np.array([(y[0],y[1]-y[2])\
     for y in list(zip(list(x),list(extr),list(np.maximum(0,xm)))) ])
    asorted=a[a[:,1].argsort()]
    
    return {"IsSuccess":status,"Bigest diff":asorted[-1]}




#%%

# %% fit payoff 
d=7
strikes = 110.0
num_assets=5
qtt_settings = QTTSettings(d)
x1_dim = DiscretizationDimension([-3,6],qtt_settings)
#x2_dim = DiscretizationDimension([-8,8],qtt_settings)
#x3_dim = DiscretizationDimension([-8,8],qtt_settings)
max_fun = np.vectorize(smooth_max3)
max_fun=np.maximum#get_smooth_max(0.1)
payoff_real=get_payoff_anydcall(strikes,max_fun)
payoff_nomax = get_nomax_payoff_anydcall(strikes)
list_x_dim = ListDiscretizationDimension(num_assets*[x1_dim])
idx_to_var = index_util(list_x_dim)
qttApx=QTTFunctionAproximation(payoff_real,idx_to_var,list_x_dim.get_index_shape())
#qttApxMultifuncr = QTTMaxofApproximation(payoff_real,idx_to_var,list_x_dim.get_index_shape())
qttpayoff=qttApx.train(20) # 20 waas good
#%%
qttpayoff_l={}
payoff_l={}
lamds = [2.5,5.,10.]
lamds=[1,-1]#[0.1]#[10.0]
lamds=[0]
for lamd in lamds:
    qttpayoff,payoff,list_qttp=qttApxMultifuncr.train(20)#.train5(30,lamd,None) # 20 waas good
    #qttpayoff = qttpayoff.round(1e-5)
    qttpayoff_l[lamd]=qttpayoff
    payoff_l[lamd]=payoff
    print(sum([np.prod(x.shape) for x in qttpayoff.to_list(qttpayoff)]))
#%%
# reduce rank
def reduce_rank(qtt,eps):
    r_new=sum([np.prod(x.shape) for x in qtt.to_list(qtt)])
    r=1.791e+308
    while r_new<r:
        r=r_new
        qtt=qtt.round(eps)
        r_new = sum([np.prod(x.shape) for x in qtt.to_list(qtt)])
        print(r_new)
    return qtt

for q in qttpayoff_l.items():
    qttpayoff_l[q[0]]=reduce_rank(q[1],1e-3)
    print(qttpayoff_l[q[0]].erank)
#%%
#tmp1=1/(lamds[0]-lamds[1])*(qttpayoff_l[lamds[0]]-qttpayoff_l[lamds[1]])
tmp1=qttpayoff_l[lamds[0]]
for _ in range(10):
    idx_test = np.random.choice([0,1],len(list_x_dim.items)*d)
    ext1=0#extrapolate_qtt(qttpayoff_l,idx_test,0.0) 
    last_qtt_payoff=0#qttpayoff[idx_test]
    jjj = [j[idx_test] for j in list_qttp]
    jj=np.sum(jjj)
    #print(jjj)
    print(last_qtt_payoff,(tmp1[idx_test]),"jj=",jj,ext1 ,payoff(idx_to_var(np.array([idx_test]))),payoff_real(idx_to_var(np.array([idx_test]))))
#%%
lamds=[0]
qttpayoff_l={0:tmp1}
payoff_l ={0:payoff_l[list(payoff_l.keys())[0]]}
#%%
#qttpayoff_l = {2.5:qttpayoff_l[2.5]}
#%%
def ttt():
    d=10
    strikes = 110.0
    qtt_settings = QTTSettings(d)
    x1_dim = DiscretizationDimension([-8,8],qtt_settings)
    x2_dim = DiscretizationDimension([-8,8],qtt_settings)
    x3_dim = DiscretizationDimension([-8,8],qtt_settings)

    payoff=get_payoff_anydcall(strikes,max_fun)
    list_x_dim = ListDiscretizationDimension([x1_dim,x2_dim,x3_dim])
    idx_to_var = index_util(list_x_dim)
    qttApx=QTTFunctionAproximation(payoff,idx_to_var,list_x_dim.get_index_shape())
    qttpayoff=qttApx.exact_payoff(list_x_dim)
    for _ in range(10):
        idx_test = np.random.choice([0,1],len(list_x_dim.items)*d)
        print(qttpayoff[idx_test],payoff(idx_to_var(np.array([idx_test]))))
    return payoff,qttpayoff,list_x_dim

payoff,exact_payoff,x_dims= ttt()
x1_dim=x_dims[0]

#%% fit QTT PDE solution
corr = np.array([[	8764,	1972,	4881,	7792,	1421,	9787,	4863,	9527,	9580,	3361,	3219,	9171,	1594,	617,	7238,	6028,	9322,	5553,	1042,	3500,	6525,	2485,	9301,	8509,	7374,	6645,	613,	6113,	2490,	9800,	5538,	8733,	2946,	6151,	7081,	2321,	4830,	6223,	3999,	5599,	1195,	493,	7881,	3724,	3839,	6658,	1681,	8596,	9527,	9314,	1678,	6806,	2238,	2804,	1806,	1552,	5909,	6998,	9485,	964,	2021,	7945,	1193,	9175,	1860,	653,	5861,	1533,	4586,	9684,	2218,	6157,	4046,	8225,	3067,	1229,	840,	2146,	7232,	2152,	8184,	5149,	1032,	27,	3752,	8929,	777,	1310,	3621,	6702,	9452,	2355,	487,	321,	582,	2851,	9198,	7011,	4930,	6654,	6038,	4086,	622,	5676,	6578,	8921,	5537,	5791,	8552,	4493,	6773,	6902,	9177,	6526,	6995,	8061,	476,	9250,	4484,	9630,	939,	4145,	],
[	6507,	3986,	2805,	5234,	5795,	3571,	6984,	205,	2074,	8537,	2296,	7611,	4348,	6159,	4081,	4147,	5168,	361,	6714,	6904,	9606,	4841,	4370,	7063,	502,	6917,	4444,	1583,	8162,	903,	1228,	8266,	6864,	956,	6614,	6651,	3189,	9346,	3149,	6564,	3260,	688,	8706,	9595,	1832,	8315,	7017,	1360,	7795,	7776,	7172,	7486,	4807,	5641,	2799,	3536,	8662,	73,	4257,	5349,	304,	3378,	5629,	2629,	1766,	4585,	6522,	8726,	1276,	9210,	5822,	8553,	2719,	1056,	8212,	5779,	9411,	4188,	2030,	6744,	7458,	4754,	7602,	8709,	9059,	1273,	6910,	2707,	2523,	7622,	8637,	8337,	7782,	8516,	3792,	8780,	3499,	7553,	1331,	4495,	2179,	1336,	2180,	6474,	4340,	4293,	2910,	1366,	2367,	7614,	2231,	6871,	3464,	1198,	8365,	790,	1482,	8035,	7053,	3046,	8434,	2447,	],
[	1960,	1536,	4912,	1816,	2044,	9997,	2723,	9550,	4125,	8741,	1158,	6346,	8782,	2996,	7375,	3678,	9150,	2122,	6385,	6465,	4470,	275,	219,	2864,	9082,	5848,	487,	5698,	4034,	6801,	7417,	599,	6040,	2186,	3910,	7821,	5015,	2614,	3277,	4806,	8452,	5603,	7118,	7824,	5012,	617,	7787,	1670,	5828,	8423,	7494,	4025,	4615,	3497,	1557,	4096,	5272,	1403,	1617,	2823,	3787,	1610,	2319,	9071,	4505,	8430,	9277,	2756,	7469,	2124,	4292,	9095,	8738,	4500,	3783,	8843,	1152,	5815,	7069,	9435,	6319,	9819,	1033,	9122,	9514,	9019,	1142,	8577,	9212,	561,	3491,	7673,	522,	1638,	1508,	639,	7744,	3230,	2260,	2021,	3305,	8570,	9999,	7474,	7214,	150,	7318,	5031,	3175,	1981,	7324,	312,	8903,	6479,	7275,	7062,	5118,	3676,	1828,	9740,	7629,	8477,	],
[	847,	7157,	8868,	5865,	8721,	8920,	6697,	2901,	4015,	323,	7409,	3586,	3009,	6337,	793,	5329,	9438,	7904,	3762,	4958,	7149,	9853,	3265,	3741,	9022,	8000,	3478,	5170,	8918,	2926,	6278,	4347,	5800,	1059,	9336,	7399,	8652,	6248,	5771,	7789,	6088,	3543,	7813,	9520,	4175,	6298,	7150,	7305,	5801,	3221,	2891,	2820,	4651,	4700,	1664,	566,	541,	4273,	5534,	5665,	2382,	700,	7802,	8339,	2084,	138,	1755,	600,	1650,	8345,	9672,	8088,	5983,	5620,	1262,	8196,	8480,	3215,	3036,	4399,	5645,	5873,	7809,	8863,	2511,	9723,	320,	6539,	5340,	503,	5440,	5504,	4168,	2672,	5935,	5879,	7349,	69,	9715,	1688,	3642,	2585,	3204,	5558,	5449,	3487,	3029,	9800,	2668,	9438,	8672,	4668,	679,	8508,	1859,	1109,	5451,	7152,	3539,	6867,	1303,	5592,	],
[	8623,	2361,	9746,	3668,	5332,	8775,	6181,	9419,	5693,	9264,	5276,	2415,	8442,	509,	1312,	7197,	7646,	7346,	9462,	4327,	3817,	3588,	5209,	9765,	269,	6033,	9789,	6689,	5428,	4905,	1244,	5362,	1136,	3164,	3770,	2509,	2098,	3786,	3848,	5487,	3494,	3774,	2688,	8812,	4279,	7326,	1434,	2261,	3837,	7641,	6583,	3210,	5907,	6711,	6208,	9776,	739,	3382,	5967,	6555,	8197,	8016,	8800,	439,	240,	5405,	2617,	5939,	4990,	1290,	4275,	6153,	7552,	4709,	6858,	2998,	6641,	6943,	6938,	3901,	4532,	9704,	3698,	8466,	8191,	5916,	2398,	5069,	1228,	1257,	1124,	8527,	9485,	7568,	8725,	5231,	9930,	5835,	6613,	3825,	9440,	7192,	3220,	2794,	4937,	7874,	4681,	670,	8171,	2900,	3463,	3455,	225,	4280,	3236,	3909,	5453,	2199,	4391,	4847,	3394,	7738,	],
[	3800,	7779,	8700,	2539,	9470,	379,	2416,	4637,	4587,	3608,	3881,	5281,	5200,	7301,	2851,	3677,	2997,	2931,	371,	9863,	2828,	2770,	2581,	9529,	4845,	4485,	5484,	7362,	887,	8999,	9195,	2004,	4863,	8331,	1736,	4998,	6231,	9903,	2323,	6030,	1647,	1641,	2834,	9909,	1051,	2813,	5642,	4360,	4082,	6759,	3324,	3480,	5103,	3606,	8027,	8928,	6498,	9147,	6302,	2602,	1732,	7168,	6839,	9366,	3050,	9322,	1016,	2235,	5438,	3444,	2579,	5897,	6673,	1896,	7831,	2024,	7108,	6350,	752,	101,	3253,	8991,	898,	2095,	5562,	8653,	9717,	3520,	4230,	6616,	9617,	8446,	4006,	1633,	3159,	4246,	762,	7438,	8625,	4200,	5737,	3836,	1149,	7628,	210,	3936,	1422,	6122,	2931,	2574,	4568,	5796,	7276,	5738,	2113,	4497,	1437,	6084,	8087,	540,	9306,	2310,	],
[	5245,	2989,	4064,	8314,	6339,	7719,	3005,	6291,	7602,	9086,	8764,	8686,	5165,	5988,	9742,	7597,	1127,	8988,	3712,	1146,	3281,	5367,	7938,	7218,	2412,	4162,	4002,	6567,	5351,	1825,	9313,	5155,	6880,	7421,	7581,	9450,	7396,	6549,	8780,	9594,	1999,	3769,	8913,	5506,	3576,	4142,	7492,	9641,	1326,	5791,	819,	2969,	7145,	7541,	4986,	9217,	9105,	3281,	5756,	6415,	6551,	7124,	5847,	5545,	423,	30,	2739,	8542,	2316,	2867,	5085,	6139,	3501,	8995,	5878,	6937,	2862,	1119,	5528,	4105,	3721,	9360,	5653,	1499,	3370,	5822,	1665,	9946,	7539,	4088,	9277,	8213,	3822,	7192,	7087,	9149,	1917,	7324,	596,	2227,	7002,	3122,	302,	894,	5826,	5186,	8846,	4305,	7505,	23,	9312,	5725,	5841,	7460,	4757,	5515,	591,	3562,	8152,	4272,	8131,	3513,	],
[	8480,	6688,	6346,	6500,	6317,	8957,	2723,	2441,	2921,	5444,	2871,	555,	7237,	5847,	5934,	5339,	216,	5684,	8042,	6253,	6531,	954,	7578,	8495,	7907,	8337,	2354,	1414,	9478,	7898,	7752,	9868,	9643,	2096,	7879,	762,	6496,	3571,	9992,	4281,	9464,	2795,	4139,	7191,	7975,	7636,	648,	9959,	260,	7545,	8763,	9451,	2184,	7663,	6187,	4306,	8681,	890,	7281,	7363,	3815,	5423,	9353,	4378,	628,	2167,	6410,	6227,	3151,	8590,	2858,	5717,	7069,	923,	4356,	3136,	3852,	4686,	2066,	4220,	3364,	7171,	3188,	5260,	3710,	6147,	9709,	485,	2968,	4175,	2991,	5598,	448,	1942,	3463,	3593,	1519,	8315,	4368,	5620,	8045,	3077,	3292,	6510,	8717,	498,	643,	9092,	2857,	1757,	1386,	3838,	3425,	2121,	7724,	6488,	6726,	6975,	3301,	1378,	3514,	6767,	],
[	6841,	7875,	1784,	3964,	1797,	2884,	7502,	7354,	2200,	7754,	5793,	9441,	9776,	6080,	189,	53,	8808,	3484,	2408,	5505,	7343,	8557,	9019,	4706,	2583,	8779,	3473,	1143,	4660,	8324,	9367,	7973,	1676,	5624,	9984,	8812,	2172,	2707,	5797,	2875,	4397,	5152,	2721,	5243,	410,	867,	5826,	6767,	5985,	1926,	5546,	7520,	3471,	8113,	2125,	9636,	6194,	9513,	9639,	3278,	6569,	62,	9186,	1400,	4217,	708,	9383,	2612,	528,	8132,	4884,	5194,	3313,	6955,	7765,	9426,	432,	4984,	2147,	8534,	2184,	8808,	7802,	1071,	949,	2219,	7691,	4104,	6666,	5784,	9534,	7119,	9391,	4958,	9808,	6042,	8752,	5006,	7820,	3616,	2877,	4591,	8212,	4206,	3495,	3575,	7894,	5666,	9972,	8473,	7550,	5794,	1225,	9987,	7988,	7712,	5840,	2390,	3192,	5955,	9970,	5800,	],
[	6433,	7836,	7891,	150,	2678,	3117,	1364,	9890,	2079,	7683,	8243,	1104,	9706,	8740,	6102,	5835,	4790,	4869,	6378,	2949,	5498,	6194,	6792,	6891,	2315,	7663,	7132,	3526,	2421,	9433,	2126,	8906,	2937,	7221,	2886,	5860,	1187,	973,	7419,	4959,	8709,	9659,	1602,	7290,	711,	310,	2857,	6377,	6571,	8357,	251,	9641,	1419,	8498,	9724,	7108,	1227,	834,	6622,	4570,	6867,	8129,	313,	3139,	8156,	4642,	2915,	5078,	4728,	5330,	9724,	4122,	3533,	8603,	7125,	1384,	7399,	4472,	2176,	4552,	8470,	8893,	5222,	2355,	7325,	7024,	8431,	6651,	599,	7700,	6374,	3353,	6686,	6330,	1014,	3711,	6126,	8351,	9345,	2501,	2739,	3632,	5151,	501,	3299,	8108,	5012,	6681,	7198,	209,	9203,	3028,	9324,	6194,	5246,	3242,	7158,	3905,	3050,	4766,	1304,	5960,	],
[	4908,	2426,	8698,	4568,	4779,	3596,	6450,	4736,	3471,	5338,	1383,	3025,	9033,	9970,	8487,	7350,	3045,	2934,	4316,	8343,	702,	2866,	5967,	436,	9873,	7100,	560,	1676,	5944,	6194,	8527,	4345,	5078,	810,	7991,	6878,	2732,	316,	3454,	6108,	6271,	7656,	9759,	4535,	1772,	2359,	5054,	9524,	3338,	6679,	5139,	6993,	5075,	2590,	5600,	9893,	4358,	7340,	9803,	6229,	5350,	6933,	2749,	9745,	166,	3948,	2519,	4897,	6275,	1559,	765,	5145,	2158,	2200,	4765,	6230,	9268,	7374,	977,	4150,	1118,	5392,	3951,	8685,	7433,	249,	6979,	3319,	7359,	4085,	8232,	8223,	1414,	4500,	2028,	5582,	7114,	8707,	380,	7705,	9235,	9827,	3216,	4259,	30,	5549,	8630,	4893,	5252,	6966,	6595,	7381,	6129,	7598,	5296,	8983,	6544,	4996,	5270,	7530,	8147,	1903,	],
[	1433,	609,	2756,	9165,	3053,	4961,	5262,	2509,	5102,	3562,	6478,	1076,	8625,	6379,	3741,	2823,	8816,	9491,	9940,	4684,	3029,	6463,	4120,	9920,	8847,	4357,	7432,	4037,	2513,	3370,	2430,	7031,	2736,	1057,	6098,	2320,	1107,	907,	9405,	6705,	2950,	7309,	7184,	9334,	7137,	1087,	6999,	3815,	6861,	8167,	4090,	6687,	7544,	1675,	2737,	7003,	2153,	3771,	8228,	5074,	3863,	4808,	9092,	1757,	4747,	2626,	7369,	724,	8917,	9952,	4468,	8544,	5332,	4490,	5487,	3601,	2812,	8731,	9649,	195,	2403,	557,	8210,	5293,	1289,	9310,	9663,	5289,	9703,	3511,	1624,	2314,	4033,	5373,	4660,	517,	3211,	4987,	770,	3619,	2357,	610,	9881,	2513,	2766,	2972,	6917,	5207,	4785,	9868,	7390,	6886,	2826,	4174,	3884,	3545,	8578,	4319,	5784,	1601,	3943,	8764,	],
[	8510,	4391,	9225,	6871,	6162,	4559,	1482,	1877,	8287,	8597,	2632,	7830,	1708,	4111,	3994,	9164,	4379,	9733,	382,	2039,	7628,	1430,	6899,	2324,	2905,	6711,	7704,	7784,	7375,	738,	7732,	5313,	885,	8415,	7080,	5012,	6947,	1575,	4710,	2402,	7712,	631,	9045,	4809,	4973,	1758,	3301,	8140,	9655,	5218,	9457,	884,	6762,	160,	9487,	187,	7250,	8740,	9684,	6906,	1413,	4852,	2581,	8011,	4138,	5794,	7494,	4134,	4823,	2341,	756,	2184,	3818,	939,	8073,	7171,	556,	3969,	1457,	4305,	5708,	6924,	1183,	2660,	2963,	7188,	3922,	124,	2162,	1388,	665,	2378,	5107,	4898,	1103,	9833,	6084,	7968,	7466,	6881,	1966,	7492,	2938,	3551,	9950,	8339,	9207,	1895,	4926,	1891,	5708,	1541,	5690,	8423,	6704,	7937,	5156,	4903,	2021,	3633,	2607,	2696,	],
[	8701,	5803,	327,	8042,	2516,	6529,	5051,	2847,	1336,	6544,	3622,	9596,	9279,	5356,	3597,	2561,	7623,	5848,	4866,	37,	3218,	4197,	7273,	9427,	3950,	463,	173,	8598,	6931,	2438,	600,	2523,	1367,	7936,	732,	6386,	8927,	1907,	1570,	4481,	2893,	2057,	958,	2125,	5634,	5426,	3101,	7609,	1670,	202,	8187,	4300,	3816,	3277,	9370,	2704,	8023,	3558,	3892,	1707,	2152,	6470,	3084,	4829,	2188,	8094,	7933,	7116,	7842,	4115,	9017,	5234,	1404,	8126,	8064,	3904,	799,	867,	7752,	9991,	4781,	7831,	2081,	7805,	3002,	4505,	4454,	3588,	5909,	6033,	2022,	9365,	5079,	8628,	7180,	8520,	2726,	9229,	7540,	6542,	892,	4526,	2511,	2404,	3809,	1545,	6739,	3814,	2715,	5117,	8416,	9146,	3919,	3103,	3559,	9370,	8427,	4327,	7476,	5634,	9760,	7011,	],
[	7217,	1992,	1636,	1496,	6290,	7069,	3409,	1112,	3978,	1647,	8498,	1978,	9312,	5028,	7765,	4937,	8755,	8728,	7644,	2767,	8840,	3085,	5819,	24,	9661,	1830,	6179,	9680,	5120,	2786,	1739,	200,	3900,	2082,	4704,	5555,	3697,	6239,	6902,	8380,	5529,	9168,	5229,	5669,	9529,	8667,	6427,	938,	9401,	3230,	5747,	3309,	7267,	3686,	4505,	43,	6204,	8276,	3555,	9830,	6657,	9547,	4401,	9542,	8926,	9711,	3616,	4943,	947,	4374,	6148,	1129,	4817,	5834,	852,	9616,	2642,	3142,	8899,	6518,	3501,	8864,	4768,	2131,	1156,	5032,	839,	1977,	1005,	649,	8575,	9140,	6935,	1103,	3659,	5913,	3373,	7115,	9246,	6556,	5186,	1777,	9129,	7507,	5423,	9710,	1150,	5066,	4499,	3045,	7027,	5759,	1720,	6230,	3308,	1206,	9888,	9074,	5529,	2609,	4478,	8007,	],
[	8897,	3551,	2215,	753,	3025,	4529,	5439,	7994,	6599,	4361,	4100,	5972,	1364,	9711,	8008,	1523,	9935,	8654,	5085,	3956,	235,	1726,	4699,	9596,	4903,	1003,	6803,	3216,	6395,	7300,	4137,	7706,	7821,	3355,	1364,	364,	5031,	676,	8229,	4229,	4907,	4209,	8025,	323,	4006,	373,	4034,	7822,	7110,	6440,	2809,	9098,	7490,	6570,	4769,	7881,	8135,	2021,	7907,	1143,	8912,	5678,	1477,	949,	7698,	2694,	2627,	9,	4825,	4520,	9092,	6239,	9096,	4197,	2144,	1192,	8627,	6588,	6126,	2578,	6473,	64,	5901,	3501,	4530,	350,	7054,	3480,	8143,	7843,	6911,	8379,	631,	7416,	1994,	9433,	5307,	6988,	1605,	9065,	2557,	8394,	3235,	57,	2524,	991,	2420,	707,	876,	5080,	3372,	5926,	1788,	762,	9723,	2112,	5948,	8588,	5418,	8812,	2143,	99,	],
[	9357,	5154,	8818,	1693,	9807,	7,	4898,	9358,	1890,	4256,	1627,	7463,	8156,	4981,	4478,	4459,	1679,	4,	3099,	5859,	9280,	7859,	5401,	4769,	970,	6623,	7798,	5743,	281,	7536,	657,	9570,	1828,	4076,	3856,	1193,	6794,	7130,	5215,	3745,	8183,	2866,	1568,	7294,	8012,	5092,	3729,	1615,	2196,	1433,	6474,	4441,	6830,	8267,	1714,	9795,	1634,	3033,	4860,	3766,	7752,	4085,	6153,	7303,	1213,	8772,	5723,	5495,	2843,	1186,	5719,	1914,	2452,	9540,	2399,	240,	6169,	7670,	5066,	524,	8918,	3655,	3681,	9723,	9901,	2505,	936,	3287,	3321,	5066,	3956,	8704,	7431,	2780,	623,	844,	8859,	7111,	7655,	7771,	475,	8168,	1960,	7210,	6221,	6574,	5739,	1687,	4389,	3430,	4459,	8579,	660,	6200,	3750,	909,	6888,	9067,	9978,	8602,	9537,	2887,	],
[	4030,	8163,	3061,	5193,	4618,	7305,	4986,	2114,	4595,	8100,	6497,	2169,	8807,	5202,	4482,	5226,	3574,	63,	7872,	3253,	7153,	4896,	8132,	9225,	4714,	9266,	6818,	6876,	5409,	6656,	7547,	4112,	7162,	1997,	6499,	874,	5480,	4498,	8907,	2268,	7785,	8148,	2258,	7327,	4362,	7644,	6741,	5714,	7349,	1724,	6598,	228,	1247,	8610,	3753,	401,	1929,	9963,	3006,	4951,	8582,	3218,	1835,	6015,	1631,	4093,	1159,	9502,	1142,	5273,	7716,	7428,	6635,	8886,	1897,	1093,	7756,	2808,	4926,	4784,	1529,	7814,	7244,	4628,	8752,	4507,	582,	8303,	4347,	3070,	7888,	3442,	1494,	5416,	5128,	5594,	4880,	6934,	4705,	3414,	6904,	1936,	2386,	7790,	1966,	1348,	7284,	1517,	3888,	2678,	9837,	1534,	2589,	3594,	9099,	583,	1975,	4155,	2897,	296,	9495,	9588,	],
[	5787,	4280,	5225,	574,	9925,	643,	4737,	6926,	552,	8457,	8475,	5644,	9442,	9934,	6944,	4521,	3919,	2374,	614,	3535,	2017,	7910,	8640,	6725,	9330,	2962,	7997,	9939,	6338,	7829,	2832,	4005,	7371,	996,	1941,	9966,	672,	9248,	7336,	2925,	8367,	7227,	5178,	8497,	5022,	2702,	4229,	4669,	915,	9495,	2020,	7908,	6758,	1710,	2869,	6691,	6555,	9871,	3903,	1012,	6605,	5576,	3561,	5004,	6357,	1348,	1007,	4504,	4003,	256,	3758,	8796,	5943,	8087,	8613,	1730,	4708,	9248,	3465,	39,	8379,	3401,	4890,	9203,	8076,	3649,	9300,	4335,	3086,	7020,	2748,	7444,	5643,	1670,	1955,	1523,	3550,	3449,	4269,	4575,	1043,	4214,	5731,	8386,	2633,	348,	5198,	8312,	4649,	1717,	3429,	1459,	4659,	5615,	9674,	8657,	5140,	9699,	9357,	666,	4731,	2323,	],
[	2199,	3402,	677,	7299,	3590,	6231,	4859,	1738,	7664,	9388,	8843,	7411,	5413,	6893,	9495,	7462,	2095,	1453,	6526,	5019,	2608,	646,	2298,	5655,	56,	35,	5369,	9822,	320,	2414,	9395,	6680,	1515,	9689,	2183,	4903,	2376,	2851,	2456,	4695,	9829,	7176,	6681,	3413,	9740,	7227,	5680,	6160,	8551,	2477,	4845,	5275,	7521,	4818,	2126,	3276,	272,	2321,	1294,	997,	307,	34,	1683,	9887,	7440,	1755,	9344,	9704,	137,	620,	9889,	7925,	8294,	4089,	836,	4202,	6774,	5851,	177,	3482,	2924,	2623,	3964,	8939,	2190,	1647,	2258,	2809,	8253,	1715,	7196,	8811,	7607,	5303,	6670,	907,	564,	8539,	8055,	5088,	8612,	8462,	3580,	2775,	2589,	2418,	5810,	2630,	53,	1182,	8469,	8403,	3335,	8571,	8332,	1257,	7522,	3835,	199,	3871,	636,	4193,	]]
)
corr = np.corrcoef(corr)
for o in range(num_assets):
    print(np.linalg.eig(corr)[0][o])
#corr = np.eye(3)

def get_vol_f(v):
    def vol_f(s):
        return v+0*s
    return vol_f

def get_local_vol_f(center,val,lam=1):
    def local_volf(x):
        return val+0.001*(center-x)**2

    def vol_f(s):
        return val+0*s
    
    xs=np.array([0,.5*center,center,1.25*center,1.5*center,5*center,10*center])
    ys=np.array([2*val,1.5*val,val,.8*val,.9*val,.5*val,.5*val])
    
    i=interpolate.interp1d(xs,ys,kind='linear',fill_value='extrapolate')
    def f(x):
        return i(x)

    xs1=np.array([0,.5*center,center,1.25*center,1.5*center,5*center,30*center])
    ys1=np.array([3*val,2*val,val,1.25*val,1.5*val,1.0*val,.5*val])
    i1=interpolate.interp1d(xs1,ys1,kind='linear',fill_value='extrapolate')
    def f1(x):
        return i1(x)

    def temp1(x):
        return np.minimum(.5,np.sqrt(  (val*np.exp(-lam*.05*(x-center))+.001*(x-center))**2+0.0001))
    return f1#temp1#vol_f


vol_list1=[.2,.3,.25,.44,.19,.23,.33,.125,.344,.319,.22,.32,.25,.44,.109,.223,.433,.125,.2344,.3319]
vol_list = [VolatilityFunction(x_dim,get_local_vol_f(100.,v)) for v,x_dim in zip(vol_list1, list_x_dim.items)]
sigmas=np.array(vol_list)
#sigmas=np.array([.4,.4,.4])
#sigmas=np.array([.03,.03,.03])
corr=corr[:num_assets,:num_assets]
sigmas=sigmas[:num_assets]
TTM = 1./12.
t_steps=10
test1 = {l:TestExample(d,0.03,sigmas,corr,TTM,t_steps,q,int(100000/20)) for l,q in qttpayoff_l.items()}
test2 = TestExampleListPayoff(d,0.03,sigmas,corr,TTM,t_steps,list_qttp,int(100000/20))
#test2 = TestExample(7,0.03,sigmas,corr,1,t_steps,exact_payoff,100000)
#%%
su={l:t1.get_setup(1e-6) for l,t1 in test1.items()}
#%%
su=test2.get_setup(1e-9)
#%%
ts = test1.test_single(payoff,x1_dim,[90,113,120])
#%%
ts = test1.test_single(payoff,x1_dim,[350,400,422])
#%%

def find_nearest(array, value):
    array = np.asarray(array)
    #print(array.shape,value)
    idx = (np.abs(array - value)).argmin()

    return idx


#q1=QTTPayoff(QTTWrapper(qttpayoff,su.x_dim),FindNearest(x1_dim.get_s_grid(1)).getf())
#q1.payoff(np.array([110,110,110]))

def plotting2d(test1,su,x1_dim,payofff):
    ts = test1.test_single(payoff,x1_dim,[400,400,400])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,2],[700,850],payofff,[50,110])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[1,2],[700,850],payofff,[50,110])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[1,2],[2*400,2*350],payofff)
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,1],[2*400,2*350],payofff)
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,2],[2*400,2*350],payofff)
    pl=test1.plot(su.x_dim,su.sol,ts[1],[1,2],[700,850],payofff)
    return ts

def plotting(test1,su,x1_dim,payofff):
    ts = test1.test_single(payoff,x1_dim,[400,400,600,400,700,400,400,600,400,700])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,2,3,4,5,6,7,8,9],[760,850,810,790,760,850,810,790,800],payofff,[40,210])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[1,2,3,4,5,6,7,8,9],[760,850,810,790,760,850,810,790,800],payofff,[40,210])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,1,3,4,5,6,7,8,9],[760,850,810,790,760,850,810,790,800],payofff,[40,210])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,1,2,4,5,6,7,8,9],[760,850,810,790,760,850,810,790,800],payofff,[40,210])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,1,2,3,4,5,6,7,8],[760,850,810,790,760,850,810,790,800],payofff,[40,210])
    
    return ts

def plotting_bad_stuff(test1,su,x1_dim,payofff):
    ts = test1.test_single(payoff,x1_dim,[400,400,600,400,700])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,1,2,3],[2*400,2*350,700,720],payofff,[80,140])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,1,2,4],[2*400,2*350,700,720],payofff,[80,140])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,2,3,4],[700,850,700,720],payofff,[80,140])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[1,2,3,4],[700,850,700,720],payofff,[80,140])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[1,2,3,4],[2*400,2*350,700,720],payofff,[80,140])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,1,3,4],[2*400,2*350,700,720],payofff,[80,140])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[0,2,3,4],[2*400,2*350,700,720],payofff,[80,140])
    pl=test1.plot(su.x_dim,su.sol,ts[1],[1,2,3,4],[700,850,700,720],payofff,[80,140])
    return ts

  

def plotting(test1,su,x1_dim,payofff,mc_repeats=1):
    spots=x1_dim.get_s_grid(1)
    FN = FindNearest(spots)
    fnf=FN.getf()
    #fnf(np.array([100.,110.]))
    ts = test1.test_single(payoff,x1_dim,[0,0,0,0,0])
    
    pl1=test1.plot(su.x_dim,su.sol,ts[1],[0,1,2,3],fnf(np.array([80,110,90,75])),payofff,[40,180],mc_repeats)
    pl2=test1.plot(su.x_dim,su.sol,ts[1],[0,1,2,4],fnf(np.array([80,130,90,75])),payofff,[40,180],mc_repeats)
    pl3=test1.plot(su.x_dim,su.sol,ts[1],[0,2,3,4],fnf(np.array([95,90,130,100])),payofff,[40,180],mc_repeats)
    pl4=test1.plot(su.x_dim,su.sol,ts[1],[1,2,3,4],fnf(np.array([95,90,130,100])),payofff,[40,180],mc_repeats)
    pl5=test1.plot(su.x_dim,su.sol,ts[1],[1,2,3,4],fnf(np.array([70,110,90,75])),payofff,[20,220],mc_repeats)
    pl6=test1.plot(su.x_dim,su.sol,ts[1],[1,2,3,4],fnf(np.array([60,110,90,75])),payofff,[40,180],mc_repeats)
    #pl_d={6:pl6}
    pl7=test1.plot(su.x_dim,su.sol,ts[1],[0,1,3,4],fnf(np.array([80,110,90,75])),payofff,[40,180],mc_repeats)
    pl8=test1.plot(su.x_dim,su.sol,ts[1],[0,2,3,4],fnf(np.array([80,55,90,165])),payofff,[40,180],mc_repeats)
    pl9=test1.plot(su.x_dim,su.sol,ts[1],[1,2,3,4],fnf(np.array([95,90,130,100])),payofff,[40,180],mc_repeats)
    pl_d={1:pl1,2:pl2,3:pl3,4:pl4,5:pl5,6:pl6,7:pl7,8:pl8,9:pl9}
    return ts,pl_d

def get_vol_fun(slope,central_vol):
    def vol_fun(x):
        return central_vol+slope*(x-central_vol)**2 


def setup_wrapper(su):
    kk=list(su.keys())
    kk.sort()
    k0=kk[0]
    solution1
    return my_bunchify('SetUp',{'x_dim':su[k0].x_dims,"sol":solution1,"sol_full":None})

lam=lamds[0]
payoff_for_mc = payoff_l[lam] #payoff_real
su_for_test = su#su[lam]
ts,pl=plotting(test1[lam],su_for_test,x1_dim,payoff_for_mc,1)
#for x in range(pl[0].shape[0]):
#    print (pl[0][x],",",pl[1][x],",",pl[2][x],pl[3][x].S)
#%%

num_terms=20
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a,axis=0), sci.stats.sem(a,axis=0)
    h = se * sci.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def pplotting(test1,su,x1_dim,payofff,mc_repeats=1):
    spots=x1_dim.get_s_grid(1)
    FN = FindNearest(spots)
    fnf=FN.getf()
    ts = test1.test_single(payoff,x1_dim,[0,0,0,0,0])
    pl6=test1.plot(su.x_dim,su.sol,ts[1],[1,2,3,4],fnf(np.array([60,110,90,75])),payofff,[40,180],mc_repeats)
    pl_d={6:pl6}
    return ts,pl_d

payoff = qttApxMultifuncr.get_train6MCPayoff(num_terms)
def trans1(sol):
    return SumOfTenosrs(sol.tensor_list,num_terms+1)
su2=test2.get_from_old(su,trans1)
su_for_test = su2#su#su[lam]
ts,pl=pplotting(test1[lam],su_for_test,x1_dim,payoff,10)
aa = np.array([pl[6][1][j][1][:] for j in range(10)])
bb = np.array([pl[6][2]])
m=mean_confidence_interval(aa,.99)
#%%
x=pl[6][0]
plt.scatter(x,m[0],label='mean',s=7,marker='o')
plt.scatter(x,m[1],label='lower 99',s=7,marker='.')
plt.scatter(x,m[2],label='uppper 99',s=7,marker='.')
plt.scatter(x,pl[6][2],label='PDE',s=20,marker='*')
plt.legend()
plt.grid()
plt.show()
for i in range(x.shape[0]):
    print('x={:7.4f},[{:7.4f},{:7.4f}],M={:7.4f},P={:7.4f}'.format(x[i],m[1][i],m[2][i],m[0][i],pl[6][2][i]))

#%%
ts = test1.test_single(payoff,x1_dim,[350,400,400])
pl=test1.plot(su.x_dim,su.sol,ts[1],[0,1],[350,400],q1.payoff)
pl=test1.plot(su.x_dim,su.sol,ts[1],[1,2],[350,400],q1.payoff)
# %%
test1.random_test(payoff,100)

# %%
d=test1.d
qtt_settings = QTTSettings(d)
x_dims=[DiscretizationDimension([-8,8],qtt_settings) for _ in range(test1.get_num_assets())]

pdeUtils = PDEUtils(x_dims,None)
pdeFactory = TTBlackScholesPDEFactory(pdeUtils)


sigma = test1.sigma 
corr= test1.corr
r=test1.r
T=test1.T
T_steps=test1.T_steps

pde = pdeFactory.new_instance(3,sigma,corr,r,0.01)
# %%
solution1=pde.roll_back(test1.qttpayoff,T,T_steps)

# %%
test1.qttpayoff
# %%
u0_full = QTTWrapper(solution1,x_dims) #ttu.full().reshape(2**d,2**d)
su= my_bunchify('SetUp',{'x_dim':x_dims,"sol":solution1,"sol_full":u0_full})

# %%
#%%
def ttt(su,test1,payoff,l=[2*400,2*350,2*425,820,800,2*400,2*350,2*425,820,800]):

    sol = QTTWrapper(su.sol,su.x_dim)
    perms = list(itertools.permutations(l))[10:20]
    spots = x1_dim.get_s_grid(1.0)
    for p in perms:
        ts = test1.test_single(payoff,x1_dim,list(p))
        s=spots[list(p)].reshape(1,-1)
        print('''s,p, '''"tt",sol[p],"mc",ts[0])
    return
    print("---------payoff apx ---------------")
    wpayoff = QTTWrapper(qttpayoff,su.x_dim)
    for p in perms:
        x=x1_dim.get_x_grid()
        spots = x1_dim.get_s_grid(1.0)
        w1=wpayoff[p]
        s=spots[list(p)].reshape(1,-1)
        print(s, p, w1,payoff(s))

#ttt(su,test,payoff,[100,90,110])
ttt(su,test1,payoff)
# %%
def smooth_max3(a,b):
    
    if a<0:
        v=0.
    elif a<(.5)**(.5):#a<0.6299605249474366:
        v=a**2+0.0 
    else: 
        v=a+0.0
    #if a>0 and a<4:
    #    print(a,v,np.maximum(a,0))
    return v
    

smax10=get_smooth_max(10.0)
smax1=get_smooth_max(10.0)
smax1over1=get_smooth_max(0.5)
s1=x1_dim.get_s_grid(1.0)
r = [i for i in range(s1.shape[0]) if s1[i]>97 and s1[i]<107]
plt.plot(s1[r],np.maximum(s1-100.0,0)[r],label="infi")
#plt.plot(s1[r],smax1(s1-100.0,0)[r],label="1")
#plt.plot(s1[r],smax10(s1-100.0,0)[r],label="10")
#plt.plot(s1[r],smax1over1(s1-100.0,0)[r],label="1/2")
plt.plot(s1[r],np.vectorize(smooth_max3)(s1-100.0,0)[r],label="vmax3")
#plt.plot(s1[r],np.array([smooth_max3(s11-100.0,0) for s11 in s1])[r],label="max3")
plt.grid()
plt.legend()
plt.show()
# %%
np.array([smooth_max3(s11-100.0,0) for s11 in s1])[r] - np.vectorize(smooth_max3)(s1-100.0,0)[r]
# %%
np.array([smooth_max3(s11-100.0,0) for s11 in s1])[r]
# %%
np.vectorize(smooth_max3)(s1-100.0,0) [r]
# %%
r
# %%

def extrapolate(x,t=0.0001):
    x=np.array([[x]])
    xm=payoff_nomax(x)
    v15=.5*((xm**2+15.)**.5+xm)
    v10=.5*((xm**2+10.)**.5+xm)
    
    v05=.5*((xm**2+1.84)**.5+xm)
    v2p5=.5*((xm**2+.9)**.5+xm)

    v05_=.5*((xm**2+(2*np.sqrt(.9))**2)**.5+xm)
    v2p5_=.5*((xm**2+np.sqrt(.9)**2)**.5+xm)

    
    extr_f = interpolate.interp1d(np.array([.9,1.84,10])\
        ,np.array([v2p5,v05,v10]).squeeze(),kind='quadratic',\
            fill_value='extrapolate')
    extr=extr_f(t)
    extr_rich = (4*v2p5_-v05_)/3.
    return x,xm,extr,v2p5,v05,v10,extr_f,extr_rich

def plot_extr(x,xm,extr,v2p5,v05,v10,extr_rich):
    
    plt.scatter(x,extr,label="Extrapolation",s=11,marker = '*')
    plt.scatter(x,v05,label="v05",s=10,marker = '*')
    plt.scatter(x,v2p5,label="v2p5",s=10,marker = '*')
    plt.scatter(x,extr_rich,label="extr_rich",s=15,marker = '.')
    plt.scatter(x,np.maximum(0,xm),label = "MAX",s=20,marker='o')
    plt.legend()
    plt.grid()
    plt.show()


def print_stats(x,xm,extr,v2p5,v05,v10):
    print("manual 10   :",v10)
    print("manual 05   :",v05)
    print("extrapolated:",extr)
    print("manual 00   :",.5*((xm**2+0)**.5+xm))
    print("payoff      :",payoff(x.reshape(-1,1)))
    print("max         :",np.maximum(0,xm))

def test_extr (x,xm,extr,v2p5,v05,v10):
    r1=[e for e in extr if e<=0]
    r2=[e for e in list((zip(list(extr),list(v2p5),list(np.maximum(0,xm))))) if np.abs(e[2]-e[0])>np.abs(e[2]-e[1]) ]
    status = len(r1)==0 and len(r2)==0

    a=np.array([(y[0],y[1]-y[2])\
     for y in list(zip(list(x),list(extr),list(np.maximum(0,xm)))) ])
    asorted=a[a[:,1].argsort()]
    
    return {"IsSuccess":status,"Bigest diff":asorted[-1]}

x,xm,extr,v2p5,v05,v10,extr_f,extr_rich=np.vectorize(extrapolate)(np.arange(108,112,.5),1e-8)
print(test_extr(x,xm,extr,v2p5,v05,v10))
#%%
plot_extr(x,xm,extr,v2p5,v05,v10,extr_rich)
# %%

# %%

# %%
q11 = np.array([1.1,2.1,3.1])
q1 = tt.vector(q11)
q1m = tt.matrix (np.array(3*[[1,1,1]]))
def f(x):
    
    y = x.astype(int)
    
    z=q11[ [y[:,0]] ]
    #print(z)
    return 4+(z-4)**2

def fitf(f):
    x0 = tt.rand(np.array([3]))
    x1 = rect_cross.cross(f, x0, nswp=10, kickrank=1, rf=2,)
    return x1

fitff=fitf(f)
print("f:",fitff.full())
q2 = tt.diag(fitff)
print("q1m",q1m.full())
print("q2",q2.full())
#mv1=q2.__matmul__(q1m)
mv1=q2*q1m

mv1.full()
# %%
np.array([3*[1,2,3]])
# %%
def f(z):
    #print("z=",z)
    return z

d=3
qtt_settings = QTTSettings(d)
x1_dim = DiscretizationDimension([-8,8],qtt_settings)
list_x_dim = ListDiscretizationDimension(1*[x1_dim])
idx_to_var = index_util(list_x_dim)
qttApx1=QTTFunctionAproximation(f,idx_to_var,list_x_dim.get_index_shape())

# %%
m1=qttApx1.get_qtt_matrix().full().reshape(8,8)
# %%
m1[[[i,i] for i in range(8)]]
# %%
m1.shape
# %%
x1_dim.get_s_grid(1.0).reshape(-1)
# %%
tr1=qttApx1.train()
# %%
tr1.full().reshape(-1)
# %%
### center is vector of centors for all the assets
### val is vector of values for all the assets
def get_local_vol_f(center,val,lam=1):
    def local_volf(x):
        return val+0.001*(center-x)**2

    def temp1(x):
        return np.sqrt(  (val*np.exp(-lam*.05*(x-center))+.001*(x-center))**2+0.0001)
    return temp1

    return local_volf,temp1

def price(inp : BlackScholesMCInputMA,Nt,lv):
        nAssets=len(inp.S)
        rr = inp.r*np.ones(nAssets)
        meanz=np.zeros(nAssets)
        mean = meanz+ (rr - .5*np.array(inp.sigma)**2)*inp.T
        cov = inp.C
        sigs=np.concatenate([sig*np.ones([nAssets,1]) for sig in inp.sigma],axis=1)
        sigs2=np.concatenate([sig*np.ones([nAssets,1]) for sig in inp.sigma],axis=1)
        cov=cov*sigs*sigs.T*inp.T
        w=np.random.multivariate_normal(meanz,inp.C,size=inp.N) 
        w2=np.random.multivariate_normal(mean,cov,size=inp.N) 
                
        S = np.array(inp.S)
        newSigma1 = lv(S)
        #newSigma1=inp.sigma
        dt =  inp.T/(Nt+0.0)
        #print(dt)
        for i in range(Nt):
            w=np.random.multivariate_normal(meanz,inp.C,size=inp.N) 
            nssig=np.array(newSigma1)
            Snext=S*np.exp((rr - .5*nssig**2)*dt +nssig*np.sqrt(dt)*w)
            S=Snext
            newSigma1 = lv(S)
                
        ssig=np.array(inp.sigma)
        #S = np.array(inp.S)*np.exp((rr - .5*ssig**2)*inp.T +ssig*np.sqrt(inp.T)*w)
        S2 = np.array(inp.S)*np.exp(w2)
        Karr=np.array(inp.K)
        SminusK=S-Karr
        SminusK2=S2-Karr
        payoff = np.amax(np.maximum(SminusK,0,),axis=1)
        payoff2 = np.amax(np.maximum(SminusK2,0,),axis=1)
        pv = np.mean(payoff)*np.exp(-inp.r*inp.T)
        pv2 = np.mean(payoff2)*np.exp(-inp.r*inp.T)
        return pv,pv2#,vvv#,w,newSigma,S,Snext
s=[100.,100.]
k=110.
r=.03
t=1.5
C=np.array([[1.,.3],[.3,1.]])
sigma=[.12,.13]
N=1000000
inp1 = BlackScholesMCInputMA(s,k,r,t,sigma,C,N)
r=price(inp1,4,get_local_vol_f(np.array([100.,95.])[1],np.array(sigma),4))
print("pv=",r[0],r[1])
# %%
def tempf1(x):
    return x**2

def tempf2(x):
    return 2*x

x = np.array([[100.,101.],[90.,110.],[180.,50.]])
fs = [tempf1,tempf2]
ys = np.array([f(x[:,j]) for j,f in zip(range(x.shape[1]),fs)]).T

def apply_fs(fs,x):
    ys = np.array([f(x[:,j]) for j,f in zip(range(x.shape[1]),fs)]).T    
    return ys

# %%
ys
# %%
x
# %%
a=np.array([[0, 10, 4, 2],
 [1, 3, 0, 2],
 [3, 2, -4, 4]])

b=np.array([[6, 9, 8, 6],
 [7, 7, 9, 6],
 [8, 6, 5, 7]])

c=np.zeros_like(a)
#a*((a>b))
(a>=4)*a
# %%
def train5(self,nswp=10,lamd=10.0,guess=None):
    def testf(x):
        y1=x#x[:,0]
        #y2=x[:,1]
        y=(y1>lamd)*y1*(lamd-y1)-.5*(lamd**2-y1**2)
        
        return y
    return  testf
s=np.arange(-10,10,.5)#x1_dim.get_s_grid(1)
#s=s-100
y1=train5(None,1,0.1)(s)
y2=train5(None,1,-0.1)(s)
plt.plot(s,1/.2*(y1-y2))
# %%
t1=tt.vector(np.array([[1,2],[3,4]]))
t2=tt.vector(np.array([[2,3],[8,9]]))
t3=t1*t2
# %%
t3
# %%
qttpayoff_l
# %%
q1=1/(lamds[0]-lamds[1])*(qttpayoff_l[lamds[0]]-qttpayoff_l[lamds[1]])
q1 = q1*q1
# %%
q1=q1.round(1e-5)
# %%
q1
# %%
t3=tt.kron(t1,t2)
# %%
tm=tt.matrix(t3)
# %%
tt.matrix.to_list(tm)
# %%
t3.n
# %%
tm
# %%
def __diag__(self):
        """ Computes the diagonal of the TT-matrix"""
        _vector=tt.vector
        c = tt.vector()
        c.n = self.n.copy()
        c.r = self.tt.r.copy()
        c.d = self.tt.d  # Number are NOT referenced
        c.get_ps()
        c.alloc_core()
        # Actually copy the data
        for i in range(c.d):
            cur_core1 = np.zeros((c.r[i], c.n[i], c.r[i + 1]))
            cur_core = self.tt.core[self.tt.ps[i] - 1:self.tt.ps[i + 1] - 1]
            print(cur_core.shape,"->",
                c.r[i], self.n[i], self.m[i], c.r[
                    i + 1 ])
            cur_core = cur_core.reshape(
                c.r[i], self.n[i], self.m[i], c.r[
                    i + 1 ], order='F')
            for j in range(c.n[i]):
                cur_core1[:, j, :] = cur_core[:, j, j, :]
                c.core[c.ps[i] - 1:c.ps[i + 1] - 1] = cur_core1.flatten('F')
        return c

#__diag__(tm)
# %%

tm1=tt.matrix(t3.full())
#%%
__diag__(tm1)
# %%
__diag__(tm)
# %%
tm1
# %%
tm
# %%
tm1
# %%
t3
# %%
t1
# %%
t2
# %%
t3_2 = tt.kron(tt.matrix(t1.full()),tt.matrix(t2.full()))
#%%
t3_2.__diag__()
# %%
tm
# %%
x=tt.matrix(t3,n=np.array([2,2]),m=np.array([2,2]))
# %%
x
# %%
x.__diag__()
# %%
t1.n
# %%
def fur(t,K,M):
    t=t/M*(np.pi/2)
    y=np.pi/2.-4/np.pi*np.sum([np.cos((2*k-1)*t)/(2*k-1)**2 for k in [i+1 for i in range(K)]])
    y=y*M/(np.pi/2)
    return y
# %%
fur(.005,10000,250)
# %%
