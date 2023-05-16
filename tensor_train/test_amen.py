#%%
from __future__ import print_function, absolute_import, division
import sys
sys.path.append('../')
import tt
from tt.amen import amen_solve
""" This program test two subroutines: matrix-by-vector multiplication
    and linear system solution via AMR scheme"""

d = 12
A = tt.qlaplace_dd([d])
x = tt.ones(2,d)
y = amen_solve(A,x,x,1e-6)

#%%

c = tt.multifuncrs2([a, b], lambda x: np.sum(x, axis=1), eps=1E-6)

y = amen_solve(A,x,x,1e-6)


# %%
y
# %%
A
# %%
x
# %%
z=tt.matvec(A,x)
# %%
z
# %%
z.full().reshape(-1)
# %%
