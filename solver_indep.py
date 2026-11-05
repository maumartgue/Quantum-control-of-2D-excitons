"""Independent radial shooting solver for the WSe2/hBN Keldysh problem (natural units,
same conventions as App. D of the paper, Eq. (D1): -1/2(d2/dr2 + 1/r d/dr - m^2/r^2) + V,
V = Keldysh(r0_eff))."""
import numpy as np
from scipy import special
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
E0=269.0; r0=0.84
def V(r):
    r=np.maximum(r,1e-300); x=r/r0
    return -np.pi/(2*r0)*(special.struve(0,x)-special.y0(x))
def rhs(m):
    c=(m*m-0.25)/2.0
    def f(r,y,E): return [y[1], 2.0*(V(r)+c/(r*r)-E)*y[0]]
    return f
def shoot(E,m,R,dense=False,rmin=1e-6):
    a=abs(m)+0.5
    y0=[rmin**a, a*rmin**(a-1)]
    sol=solve_ivp(rhs(m),(rmin,R),y0,args=(E,),method='DOP853',rtol=1e-11,atol=1e-14,dense_output=dense)
    return sol
def bound(m,Eguess,R,width=0.05):
    f=lambda E: shoot(E,m,R).y[0,-1]
    E=brentq(f,Eguess*(1+width),Eguess*(1-width),xtol=1e-15,rtol=1e-14,maxiter=300)
    return E
def bound_wf(m,E,R,rgrid):
    sol=shoot(E,m,R,dense=True)
    u=sol.sol(rgrid)[0]
    n=np.sqrt(np.trapezoid(u*u,rgrid)); return u/n
