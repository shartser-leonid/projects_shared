#%%
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
            inp.r,
            inp.K,
            inp.T,
            inp.sigma,
            inp.N,
            inp.C
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


class QTTWrapper:
    def __init__(self,qtt_vector,x_dim : list):
        self.qtt_vector=qtt_vector
        self.x_dim = x_dim
        
    def __getitem__(self,i):
        return self.qtt_vector[np.concatenate([\
            self.x_dim[j].qtt_setting.ind2subi(i[j]) for j in range(len(self.x_dim))\
                ])]
    
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
        payoff = np.maximum(x[:,0]+x[:,1]-strikes,0)
        return payoff

    return function_of_var


class QTTFunctionAproximation:
    def __init__(self,func_of_var,index_to_var,index_shape):
        self.function_of_variables = func_of_var
        self.index_to_var = index_to_var
        self.index_shape=index_shape

    def func_of_ind(self,x):
        y = x.astype(int)
        return self.function_of_variables(self.index_to_var(y))

    def train(self):
        n = self.index_shape
        x0 = tt.rand(n)
        x1 = rect_cross.cross(self.func_of_ind, x0, nswp=5, kickrank=1, rf=2)
        return x1

    
# %%
d=4
qtt_settings = QTTSettings(d)
x1_dim = DiscretizationDimension([-8,8],qtt_settings)
x2_dim = DiscretizationDimension([-8,8],qtt_settings)
strikes = 110.0
f=get_payoff_2dcall(strikes)
list_x_dim = ListDiscretizationDimension([x1_dim,x2_dim])
idx_to_var = index_util(list_x_dim)
qttApx=QTTFunctionAproximation(f,idx_to_var,list_x_dim.get_index_shape())
x1=qttApx.train()
idx_test = [1,1,1,1,0,1,1,1]
print(x1[idx_test],f(idx_to_var(np.array([idx_test]))))

# %%


class TestExample:
    def __init__(self,d,Karr,r,sigma,corr,T,T_steps):
        self.d=d
        self.Karr=Karr
        self.r=r
        self.sigma=sigma
        self.corr=corr
        self.T=T
        self.T_steps = T_steps

    def get_setup(self):
        d=self.d
        qtt_settings = QTTSettings(d)
        x1_dim = DiscretizationDimension([-8,8],qtt_settings)
        x2_dim = DiscretizationDimension([-8,8],qtt_settings)

        pdeUtils = PDEUtils([x1_dim,x2_dim],None)
        pdeFactory = TTBlackScholesPDEFactory(pdeUtils)
        call_payoff1 = call_payoff_2D(x1_dim,x2_dim) 

        Karr=self.Karr#np.array([100.0,150.0])
        sigma = self.sigma #np.array([.2,.3])
        corr= self.corr#np.array([[1,.22],[.22,1]])
        r=self.r
        T=self.T
        T_steps=self.T_steps
        pde = pdeFactory.new_instance(2,sigma,corr,r)
        solution1=pde.roll_back(call_payoff1.get_qtt(Karr),T,T_steps)

        ####################################
        # test fit against MC              #
        ####################################
        ttu=solution1
        u0_full = ttu.full().reshape(2**d,2**d)
        return my_bunchify('SetUp',{'x1':x1_dim,'x2':x2_dim,"sol":solution1,"sol_full":u0_full})

    def test_single(self,x1_dim):
        bsMA_input = self.get_bsma_input(x1_dim,[9,13])
        bsma1 = BlackScholesMCMAGeneric(f)
        print("MC price",bsma1.price(bsMA_input))
        

    def get_bsma_input(self,x1_dim,x1i):
        r=self.r
        T=self.T
        x=x1_dim.get_x_grid()
        spots = x1_dim.get_s_grid(1.0)
        rho = self.corr[0,1]
        strikes=self.Karr
        sigma_a = self.sigma
        spots = get_spots_from_x(x[x1i],r,T,np.array(sigma_a))
        print("spots=",spots)
        bsMA_input = BlackScholesMCInputMA(spots,strikes,r,T,sigma_a,np.array([[1,rho],[rho,1]]),10)
        return bsMA_input

test1 = TestExample(10,np.array([100,150]),0.03,np.array([.2,.3]),np.array([[1,-.3],[-.3,1]]),1,20)
test1.test_single(x1_dim)

# %%
