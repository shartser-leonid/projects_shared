#%%

import tt_blackscholes as ttbs
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple


def my_bunchify(name,d):
    d_named = namedtuple(name, d)(**d)
    return d_named


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
        qtt_settings = ttbs.QTTSettings(d)
        x1_dim = ttbs.DiscretizationDimension([-8,8],qtt_settings)
        x2_dim = ttbs.DiscretizationDimension([-8,8],qtt_settings)

        pdeUtils = ttbs.PDEUtils([x1_dim,x2_dim],None)
        pdeFactory = ttbs.TTBlackScholesPDEFactory(pdeUtils)
        call_payoff1 = ttbs.call_payoff_2D(x1_dim,x2_dim) 

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

    def test_single(self,x1_dim,x2_dim,solution1,u0_full):
        bsMA_input = self.get_bsma_input(x1_dim,x2_dim)
        bsma1 = ttbs.BlackScholesMCMA()
        x1i=[510,800]
        print("MC price",bsma1.price(bsMA_input))
        print("qtt price",u0_full[x1i[0],x1i[1]])

    def get_bsma_input(self,x1_dim,x1i):
        r=self.r
        T=self.T
        x=x1_dim.get_x_grid()
        spots = x1_dim.get_s_grid(1.0)
        rho = self.corr[0,1]
        strikes=self.Karr
        sigma_a = self.sigma
        spots = ttbs.get_spots_from_x(x[x1i],r,T,np.array(sigma_a))
        print("spots=",spots)
        bsMA_input = ttbs.BlackScholesMCInputMA(spots,strikes,r,T,sigma_a,np.array([[1,rho],[rho,1]]),10000)
        return bsMA_input

    def plot(self,x1_dim,x2_dim,solution1,bsMA_input,fixed_axis,fixed_index):
        fi=FixedIndexManager(2,[fixed_axis],[fixed_index])
        bsma1 = ttbs.BlackScholesMCMA()
        bs_inp_facotry = ttbs.BlackScholesMCInputMAFactory()
        inp_test = bs_inp_facotry.new_from_old(bsMA_input)
        u0=ttbs.QTTWrapper(solution1,[x1_dim,x2_dim])
        s_grid=x1_dim.get_s_grid(1.0)
        inp_test.S[fixed_axis]=s_grid[fixed_index]
        x_range = [i for i,xx in enumerate(s_grid) if xx>50 and xx<180]
        y_mc1 = [bsma1.price(bs_inp_facotry.new_update_S(inp_test,1-fixed_axis,s_grid[i])) for i in range(len(s_grid))]
        fig = plt.figure()
        ax = fig.gca()

        plt.scatter(s_grid[x_range],np.array([u0[fi.get([i])] for i  in range(len(s_grid))])[x_range],label="PDE solution",s=29,marker = '*')
        plt.scatter(s_grid[x_range],np.array([y_mc1[i] for i  in range(len(s_grid))])[x_range],label = "MC price",s=7,marker='.')
        plt.legend()
        plt.grid()
        plt.show()

test1 = TestExample(10,np.array([100,150]),0.03,np.array([.2,.3]),np.array([[1,-.3],[-.3,1]]),1,20)
setup1=test1.get_setup()

test1.plot(setup1.x1,setup1.x2,setup1.sol,test1.get_bsma_input(setup1.x1,[510,200]),0,500)
#%%
test1.plot(setup1.x1,setup1.x2,setup1.sol,test1.get_bsma_input(setup1.x1,[510,200]),1,500)
#%%
test1.plot(setup1.x1,setup1.x2,setup1.sol,test1.get_bsma_input(setup1.x1,[510,200]),0,500)
test1.plot(setup1.x1,setup1.x2,setup1.sol,test1.get_bsma_input(setup1.x1,[510,200]),1,500)
# %%

# %%
# plot QMC vs QTT
def test_qmc_vs_mc(u0_full,flip=False):
    test_index=500#850
    
    def get_inp_with_changing_x(i,TTT,flip=False):
        x1i=[i,test_index]
        if flip:
            x1i=[test_index,i]
        spots = ttbs.get_spots_from_x(x[x1i],r,TTT,np.array(sigma_a))
        return ttbs.BlackScholesMCInputMA(spots,strikes,r,TTT,sigma_a,np.array([[1,rho],[rho,1]]),10000)

    y_mc1 = [bsma1.price(get_inp_with_changing_x(i,0,flip)) for i in range(len(x))]
    y_mc2 = [bsma1.price(get_inp_with_changing_x(i,T,flip)) for i in range(len(x))]
    fig = plt.figure()
    ax = fig.gca()
    s12 = ttbs.get_spots_from_x(np.array([x,x]),r,T,sigma)
    s1=s12[0]
    s2=s12[1]
    x_range = [i for i,xx in enumerate(s1) if xx>80 and xx<180]
    plt.scatter(s1[x_range],np.array(y_mc1)[x_range],label="MC1 "+str(np.exp(x[test_index])),s=5)
    plt.scatter(s1[x_range],np.array(y_mc2)[x_range],label="MC2"+str(np.exp(x[test_index])),s=5)
    yqtt=u0_full[x_range,test_index] if not flip else u0_full[test_index,x_range]
    #yqtt1=u1_full[x_range,test_index] if not flip else u1_full[test_index,x_range]
    
    k=0
    bits = ttbs.get_bits_func(d)
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
