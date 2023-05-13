#%%
import dis
from gettext import npgettext
from importlib.metadata import requires
from re import S
import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

#%%

'''
The goal is to create a Monte Carlo pricer
for exotic options

After that use some sort of neural net frame work to to learn
the pricer.
Idealy, use pytorch, but can start with sci-kit learn.
Try some other functions e.g. polynomial instead of neural net.

1. Try to model the neural net directly
2. Try to model neural net by minimizing the loss of PDE

sub-tasks

1. generate market data for learning
2. create trainig loop
3. create testing/validation set




'''

#%%


class BsOption:
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r 
        self.sigma = sigma
        self.q = q
        
    
    @staticmethod
    def N(x):
        return norm.cdf(x)
    
    @property
    def params(self):
        return {'S': self.S, 
                'K': self.K, 
                'T': self.T, 
                'r':self.r,
                'q':self.q,
                'sigma':self.sigma}
    
    def d1(self):
        return (np.log(self.S/self.K) + (self.r -self.q + self.sigma**2/2)*self.T) \
                                / (self.sigma*np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.T)
    
    def _call_value(self):
        return self.S*np.exp(-self.q*self.T)*self.N(self.d1()) - \
                    self.K*np.exp(-self.r*self.T) * self.N(self.d2())
                    
    def _put_value(self):
        return self.K*np.exp(-self.r*self.T) * self.N(-self.d2()) -\
                self.S*np.exp(-self.q*self.T)*self.N(-self.d1())
    
    def price(self, type_ = 'C'):
        if type_ == 'C':
            return self._call_value()
        if type_ == 'P':
            return self._put_value() 
        if type_ == 'B':
            return  {'call': self._call_value(), 'put': self._put_value()}
        else:
            raise ValueError('Unrecognized type')
            
        


# %%
class PricingModel:
    
    def __init__(self,spots,volatility,spot_repo,corr,risk_free_rate,number_of_path,time_grid):
        self._spots = spots
        self._volatility = volatility
        self._spots_repo = spot_repo
        self._corr = corr
        self._risk_free_rate = risk_free_rate
        self._time_grid = time_grid
        self._number_of_path = number_of_path
    
    def get_discount(self):
        return np.exp(-self._risk_free_rate*self._time_grid)

    def get_path(self):
        #setup
        num_assets = self._spots.shape[0]
        N = self._number_of_path
        dt = np.array(self._time_grid[1:] - self._time_grid[:-1])
        dt=np.insert(dt,0,self._time_grid[0])
        steps = len(dt)
    
               
        # 1/ get normals assets x setps x simulation
        w = np.random.normal(size=(num_assets,steps,N))

        # 2/ correlate assets
        C = np.linalg.cholesky(self._corr) 
        z = np.tensordot(C,w,axes=([1],[0])) # assets x steps x simulations

        # 3/ create drift
        drift1 = (dt[:,None]*self._spots_repo[None,:]).T
        drift2 = -(dt[:,None]*.5*self._volatility[None,:]**2).T

        #4/ create diffusion
        volsSqrtDt = (np.sqrt(dt[:,None])*self._volatility[None,:]).T # assets x steps
        diffusion = volsSqrtDt[:,:,None]*z
        
        #5/ combine all together
        drift = drift1+drift2
        path = drift[:,:,None] + diffusion 
        return path


class PathManager:
    def __init__(self,spots,path,discounts):
        self._spot = spots
        self._path = path
        self._disc = discounts
    
    def get_discount(self):
        return self._disc
    
    def get_s(self):
        path = self._path
        return self._spot[:,None]*np.exp(path.sum(axis=1))
    
    def get_s_steps(self):
        path = self._path # log return, size: assets x steps x simulation
        prev=0
        path_sums=[]
        for j in range(path.shape[1]):
            prev=prev+path[:,j,:]
            path_sums.append(prev)
        s_steps=[]
        for j,p in enumerate(path_sums):
            s_steps.append(self._spot[:,None]*np.exp(p))
        s_steps=np.array(s_steps)
        s_steps=np.swapaxes(s_steps,0,1)
        return s_steps

class BasketOption:
    def __init__(self,s_init,weights,strike):
        self._s_init = s_init
        self._weights = weights
        self._strike = strike
    
    def price(self,path_manager : PathManager):
        s=path_manager.get_s()
        discount_factor = path_manager.get_discount()
        x1=s/self._s_init[:,None]
        x2=self._weights[:,None]
        s = np.dot(x1.T,x2)
        payoffs = np.maximum(s-self._strike,0)
        return payoffs.mean()*discount_factor[-1]*100

class BarrierOption:
    def __init__(self,s_init,weights,strike,barrier):
        self._s_init = s_init
        self._weights = weights
        self._strike = strike
        self._barrier = barrier
    
    def price(self,path_manager : PathManager):
        s=path_manager.get_s_steps() # asset x steps x simulation
        discount_factor = path_manager.get_discount()
        x1=s/self._s_init[:,None,None]  # asset x steps x simul
        x2=self._weights[None,:].T
        s = np.dot(x1.T,x2).T[0,:,:] # contract along asset dimension => steps x simul
        ko_set=[] 
        for j in range(s.shape[0]):
            ko = s[j]<self._barrier
            ko_set.append(ko)
        ko = np.array(ko_set)        
        ko=np.logical_not(np.logical_or.reduce(ko,axis=0)) # size simul

        payoffs = np.maximum(s[-1]-self._strike,0) # size simul
        payoffs = payoffs * ko

        return payoffs.mean()*discount_factor[-1]*100
        
def a(x):
    return np.array(x)
#%%
class DataGenerator:
    def __init__(self,number_of_samples,number_of_assets):
        self._number_of_samples,self._number_of_assets=\
            number_of_samples,number_of_assets

    def get_x(self):
        N = self._number_of_samples
        A = self._number_of_assets
        r = np.random.uniform(low=0.001,high=.3,size=(N,1))
        c = np.random.uniform(low=-1,high=1,size=(N,1))
        vol = np.random.uniform(low=0.001,high=.3,size=(N,A))
        w = np.random.uniform(low=0.1,high=1,size=(N,A))
        w=w/w.sum(axis=1)[:,None]
        k = np.random.uniform(low=0.1,high=2,size=(N,1))
        T = np.random.uniform(low=0.01,high=5,size=(N,1))
        return np.concatenate((r,c,vol,w,k,T),axis=1)

    def get_y(self,data_x,n_sim):
        A = self._number_of_assets
        data_y = np.zeros((data_x.shape[0],1))
        for i in range(data_x.shape[0]): 
            r = data_x[i][0]
            c = data_x[i][1]
            vol = data_x[i][2:A+2]
            w = data_x[i][A+2:A+A+2]
            k= data_x[i][A+A+2]
            T= data_x[i][A+A+3]
            repo = r*np.ones_like(vol)
            bo = BasketOption(np.array([100.,100.]),w,k)
            
            time_grid=np.array([T])
            corr=a([[1,c],[c,1]])

            pricing_model = PricingModel(a([100,100]),vol,repo,corr,\
                r,n_sim,time_grid)

            path_manager = PathManager(np.array([100,100]),pricing_model.get_path(),pricing_model.get_discount())
            data_y[i]=bo.price(path_manager)
        return data_y
        

#%%
#dg.get_y(data_x[0:5],10000)
#%%

dg=DataGenerator (50000,2)
data_x=dg.get_x()
data_y=dg.get_y(data_x,30)
data_y
#%%

class MarketDataset(torch.utils.data.Dataset):

  def __init__(self, X, y, scale_data=False):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

class MLP(nn.Module):
  def __init__(self,in_size):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(in_size, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )


  def forward(self, x):
    return self.layers(x)
X = data_x
y = data_y
dataset = MarketDataset(X, y)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

# Initialize the MLP
mlp = MLP(data_x.shape[1])
# Define the loss function and optimizer
loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
losses=[]
#%%
  # Run the training loop
for epoch in range(0, 12): # 5 epochs at maximum
    
    # Print epoch
    print(f'Starting epoch {epoch+1}')
    
    # Set current loss value
    current_loss = 0.0
    
    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader):
      #print (i)
      
      # Get and prepare inputs
      inputs, targets = data
      inputs, targets = inputs.float(), targets.float()
      targets = targets.reshape((targets.shape[0], 1))
      
      # Zero the gradients
      optimizer.zero_grad()
      
      # Perform forward pass
      outputs = mlp(inputs)
      
      # Compute loss
      loss = loss_function(outputs, targets)
      
      # Perform backward pass
      loss.backward()
      losses.append(loss.item())
      
      # Perform optimization
      optimizer.step()
      
      # Print statistics
      current_loss += loss.item()
      if i % 10 == 0:
          print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 500))
          current_loss = 0.0

  # Process is complete.
print('Training process has finished.')
#%%
plt.plot(losses)
plt.show()
for x in zip(dg.get_y(data_x[0:5],1000000),mlp.forward(dataset.X[0:5].float())):
    print(x[0][0],"<-- mc   mlp --->",x[1].item())
#%% test

bo = BasketOption(np.array([100,80]),a([0,1]),1.1)
bar = BarrierOption(np.array([100,80]),a([0,1]),1.1,.95)
n_sim=1000000
time_grid=np.linspace(0,1,4)
corr=a([[1,-.5],[-.5,1]])

pricing_model = PricingModel(a([100,80]),a([.3,.2]),a([0.03,0.03]),corr,\
    0.03,n_sim,time_grid)

path_manager = PathManager(np.array([100,80]),pricing_model.get_path(),pricing_model.get_discount())
print("basket op",bo.price(path_manager))
print("barrier op",bar.price(path_manager))
#%%

bo = BasketOption(np.array([100,80]),a([0,1]),1.1)
bar = BarrierOption(np.array([100,80]),a([0,1]),1.1,.95)
print("basket op",bo.price(path_manager))
print("barrier op",bar.price(path_manager))


#%%
sss=path_manager.get_s_steps()

print(sss[0,-1,:])
print(path_manager.get_s())


#%%  testing ............

def a(x):
    return np.array(x)

n_sim=1000000
weights1=np.array([0,1])
weights2=np.array([1,0])
time_grid=np.linspace(0,10,1)#a([0,.25,.5,.75,1])
corr=a([[1,-.5],[-.5,1]])
p = PricingModel(a([100,80]),a([.6,.2]),a([0.03,0.03]),corr,\
    0.03,n_sim,time_grid)
path=p.get_path()
print(path.shape)
strike=1.1
discount_factor = p.get_discount()
pay1=np.maximum(np.dot(weights1[None,:],np.exp(path.sum(axis=1)))-strike,0)*discount_factor[-1]
pay2=np.maximum(np.dot(weights2[None,:],np.exp(path.sum(axis=1)))-strike,0)*discount_factor[-1]

print(pay1.mean()*100)
print(pay2.mean()*100)
bs1 = BsOption(100,110,10,0.03,0.6).price()
bs2 = BsOption(100,110,10,0.03,0.2).price()
print("test",bs1,bs2)
#%%
plt.plot(time_grid,np.exp(path[0]),'r')
plt.plot(time_grid,np.exp(path[1]),'g')
# %%
# %%
path.shape
# %%
x=np.random.normal(size=(3,5,7))
# %%
np.moveaxis(x,0,2).shape
# %%


a=torch.autograd.Variable(torch.Tensor([1,2]),requires_grad=True)
opt = torch.optim.Adam([a],lr=0.01)


test_x = np.random.uniform(low=0,high=1,size=[100,1])
noise = np.random.normal(size=(100,1))
test_y = 5*test_x + 7 + 0.1*noise
plt.scatter(test_x,test_y) 
N=2000
test_x_ = torch.from_numpy(test_x)
test_y_ = torch.from_numpy(test_y)
for i in range(N):
    loss=a[0]*test_x_+a[1] - test_y_
    loss=(loss**2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()



print(a)
# %%
dimP = 10

a1 = np.random.normal(size=(dimP,4))
a2 = np.random.normal(size=(4,dimP,4))
a3 = np.random.normal(size=(4,dimP,4))
a4 = np.random.normal(size=(4,dimP))
#%%


#%%
def c(i1,i2,i3,i4):
    b1 = np.tensordot(a3[:,i3,:],a4[:,i4],axes=([1],[0])) # 4 x 1
    b2 = np.tensordot(a2[:,i2,:],b1,axes=([1],[0])) # 4 x 1
    b3 = np.tensordot(a1[i1,:][None,:],b2[:,None],axes=([1],[0])) #
    return b3
#%%
a1_ = torch.from_numpy(a1)
a2_ = torch.from_numpy(a2)
a3_ = torch.from_numpy(a3)
a4_ = torch.from_numpy(a4)

a1_=torch.autograd.Variable(a1_,requires_grad=True)
a2_=torch.autograd.Variable(a2_,requires_grad=True)
a3_=torch.autograd.Variable(a3_,requires_grad=True)
a4_=torch.autograd.Variable(a4_,requires_grad=True)


#%%
def c_(i1,i2,i3,i4):
    b1 = torch.tensordot(a3_[:,i3,:],a4_[:,i4],dims=([1],[0])) # 4 x 1
    b2 = torch.tensordot(a2_[:,i2,:],b1,dims=([1],[0])) # 4 x 1
    b3 = torch.tensordot(a1_[i1,:][None,:],b2[:,None],dims=([1],[0])) #
    return b3

def c_x(x):
    return c_(x[0],x[1],x[2],x[3])

# %%
print(c_(1,1,1,1))
print(c(1,1,1,1))
# %%
def bsmc(r,vol,T,k,N):
    w = np.random.normal(size=[N,1])
    s = np.exp((r-.5*vol**2)*T + np.sqrt(T)*vol*w)
    return np.maximum(s-k,0).mean()*np.exp(-r*T)*100

#%%
possible_r = np.linspace(0.001,0.4,10)
possible_vol = np.linspace(0.001,0.8,10)
possible_T = np.linspace(0.01,5,10)
possible_k = np.linspace(0.1,2,10)

number_of_data = 1000

rs_idx=np.random.choice([x for x in range(10)],size=(number_of_data,1))
vols_idx=np.random.choice([x for x in range(10)],size=(number_of_data,1))
Ts_idx=np.random.choice([x for x in range(10)],size=(number_of_data,1))
ks_idx=np.random.choice([x for x in range(10)],size=(number_of_data,1))

rs=possible_r[rs_idx]
vols=possible_vol[vols_idx]
Ts=possible_T[Ts_idx]
ks=possible_k[ks_idx]

x_data = np.concatenate([rs,vols,Ts,ks],axis=1)
x_data_idx = np.concatenate([rs_idx,vols_idx,Ts_idx,ks_idx],axis=1)
x_data_idx_ = torch.from_numpy(x_data_idx)
#%%
y_data = np.array([bsmc(x[0],x[1],x[2],x[3],30) for x in x_data])
y_data_ = torch.from_numpy(y_data)
#%%
opta = torch.optim.Adam([a1_,a2_,a3_,a4_],lr=1e-1)

old_loss = None
keep_on = True
while keep_on:
    for it in range(10):
        loss = (c_x(x_data_idx_[0]) - y_data_[0])**2
        for i in range(1,x_data_idx_.shape[0]):
            loss+=(c_x(x_data_idx_[i]) - y_data_[i])**2
        loss.backward()
        opta.step()
        opta.zero_grad()
    new_loss = loss.item()
    print("new loss",new_loss)
    if old_loss==None:
        old_loss=new_loss
    else:
        if new_loss<old_loss:
            keep_on=True
            old_loss=new_loss
        else:
            keep_on=False


#%%
# testing

for i in range(0,20):
    i1,i2,i3,i4=x_data_idx[i]# 5,6,5,5
    r,vol,T,k = possible_r[i1],possible_vol[i2],possible_T[i3],possible_k[i4]
    bm_v=bsmc(r,vol,T,k,1000000)
    print(c_(i1,i2,i3,i4).item(),bm_v)
#%%
i1,i2,i3,i4= 5,6,5,5
r,vol,T,k = possible_r[i1],possible_vol[i2],possible_T[i3],possible_k[i4]
bm_v=bsmc(r,vol,T,k,1000000)
print(c_(i1,i2,i3,i4).item(),bm_v)    
#%%
i1,i2,i3,i4= 5,5,5,5
plt.plot(possible_vol,[c_(i1,i,i3,i4) for i in range(10)])
r,vol,T,k = possible_r[i1],possible_vol[i2],possible_T[i3],possible_k[i4]
plt.plot(possible_vol,[bsmc(r,possible_vol[i],T,k,1000000) for i in range(10)])
plt.show()

# %%
print(bsmc(0.03,.3,1,1.1,1000000))
print(BsOption(100,110,1,0.03,0.3).price())
# %%
loss.item()
# %%
