#%%
from __future__ import print_function, absolute_import, division
import numpy as np
import tt
a = tt.rand([3, 5, 7, 11], 4, [1, 4, 6, 5, 1])
b = tt.rand([3, 5, 7, 11], 4, [1, 2, 4, 3, 1])
c = tt.multifuncrs2([a, b], lambda x: np.sum(x, axis=1), eps=1E-6)

print("Relative error norm:", (c - (a + b)).norm() / (a + b).norm())

# %%
a
# %%
a.full().shape
# %%
c
# %%
a = tt.vector(np.array([[[-1,2],[-5,4]],[[-1,2],[-5,4]]]))
def iamafunc(x):
    print("in",x)
    y=np.maximum(x,0)
    print("out",y)
    return y
c = tt.multifuncrs2([a], np.vectorize(iamafunc), eps=1E-6)
# %%
c.full().shape
# %%
def testf(x):
    #print("A",x)
    #print("0",x[:,0])
    #print("1",x[:,1])
    #print(x)
    y1=x[:,0]
    y2=x[:,1]
    #print(y1,y2)
    y=.5*(np.power(y1,.5)+y2)
    return y

b=tt.multifuncrs2([a*a,a],testf,eps=1e-6)
# %%
b.full()
# %%
a.full()
# %%
b.full()
# %%
