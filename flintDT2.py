#!/usr/bin/env python3
#
# Fast 2D D-T2 distribution estimation - numpy version
# Paul Teal, Victoria University of Wellington
# paul.teal@vuw.ac.nz
#
# Let me know of feature requests, and if you find this algorithm does 
# not perform as it should, please send me the data-set, so I can improve it.
# Issued under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3.
# If you distribute this code or a derivative work you are required to make the
# source available under the same terms
#
# Based on flint.py (2 Dec 2020), based on flint.m (30 Oct 2013)
# Versions:
# 0.1 2 December 2020
# 0.2 2 December 2020
# 0.3 3 December 2020
# 0.4 8 December 2020
# 0.5 28 May 2021
# Modified for distribution, Version 1.0, Wednesday, 15 December 2021 

import numpy as np

def flint(K1,K2,ZZ,alpha,SS=1.,
             lambd=0.,K3=1.,tol=1e-5,maxiter=300001,progress=500,
              Strue=1):
  """
  FLINT: Fast Lapace-like INverTer (2D)

  K1 and K2 are the kernel matrices
  ZZ is the NMR data for inversion
  alpha is the (Tikhonov) regularisation multiplier (scalar)
  SS is an optional starting estimate

  K3 is the "cross-kernel" matrix that couples the other two
     (making them dependent)
  lambd is the l1 regularisation multiplier 
  tol is the relative change between successive loss calculations
     for exit
  maxiter is the maximum number of iterations
  progress is the number of iterations between progress display
  (should be several hundred because calculating the loss is much slower than
   the main iterations)
  Strue is the true solution, for tracking the error in simulation studies

  If you use this software, please cite P.D. Teal and E.H Novotny. Improved
  data efficiency for NMR diffusion-relaxation processing. Journal of
  Magnetic Resonance, 2021. http://dx.doi.org/10.1016/j.jmr.2021.107124
  """

  K1 = np.atleast_2d(K1)
  K2 = np.atleast_2d(K2)
  ZZ = np.atleast_2d(ZZ)
  
  lossa = np.full((maxiter), np.nan)

  if not isinstance(SS,np.ndarray):
    SS = np.ones((K1.shape[1],K2.shape[1]))  # initial estimate
  if not isinstance(Strue,np.ndarray):
    Strue = np.ones((K1.shape[1],K2.shape[1]))
  StrueE = 1/np.sum(Strue**2)

  K3present = False  
  if isinstance(K3,np.ndarray):
    K3present = True
    
  KK2 = K2.T @ K2
  if K3present:
    KZ12 = K1.T @ (K3 * (ZZ @ K2) )
  else:
    KK1 = K1.T@K1
    KZ12 = K1.T@ZZ@K2

  # Find the Lipschitz constant: The kernel matrices are poorly
  # conditioned so this will typically converge very fast
  SL = np.copy(SS)
  LL = np.inf
  for ii in range(100):
    lastLL = LL
    LL = np.sqrt(np.sum(SL**2))
    if np.abs(LL-lastLL)/LL < 1e-10:
      break
    SL = SL/LL
    if K3present:
      SL = K1.T @ (K3 * ((K3 * (K1 @ SL)) @ KK2 ))
    else:
      SL = KK1@SL@KK2
  LL = 1.001 * 2 * (LL + alpha)
  print('Lipschitz constant found: ii= % 2d, LL= % 1.3e'%(ii,LL))

  YY = SS
  tt = 1
  fac1 = (LL-2*alpha)/LL
  fac2 = 2/LL
  fac3 = lambd/LL
  lastres = np.inf

  for iter in range(maxiter):
    if K3present:
      term2 = KZ12 - K1.T @ (K3 * ((K3 * (K1 @ YY)) @ KK2 ))
    else:
      term2 = KZ12 - KK1 @ YY @ KK2
    Snew = fac1*YY + fac2*term2 - fac3
    Snew[Snew<0.] = 0.
    ttnew = 0.5*(1 + np.sqrt(1+4*tt**2))
    trat = (tt-1)/ttnew
    YY = Snew + trat * (Snew-SS)
    tt = ttnew
    SS = Snew

    if iter % progress == 0:
      # Don't calculate the residual every iteration -
      # that would slow things down a lot
      normS = alpha * np.sum(SS**2) + lambd * np.sum(np.abs(SS))
      KS = (K3 * (K1 @ SS)) @ K2.T
      resid = np.sum( (ZZ-KS)**2 ) + normS
      lossa[iter] = resid
      resd = (lastres-resid)/resid
      lastres = resid
      trueerr = np.sum((SS-Strue)**2) * StrueE

      # show progress
      print('%7d % 1.2e % 1.2e % 1.3e % 1.3e % 1.3e'
            %(iter,tt,1-trat,resd,resid,trueerr))
      if np.abs(resd)<tol:
        return SS,lossa
  return SS,lossa


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  plt.ion()
  
  # Set up grids in both domains
  Ng = 101
  NT2 = 100
  NE1 = 34  # aiming for 30, but making integer multiples of
            # the echo time results in some duplicates
  Ntp = 5000
  tE = 0.15e-3
  D0 = 2.3e-9            # diffusion coefficinet for water
  gamma = 267.52218744e6 # gyro-magnetic ratio for proton
  sigma = 1e-2           # noise level

  T2grid = np.logspace(-2,1,NT2)
  ggrid = np.logspace(-3,2,Ng)
  tpgrid = np.arange(Ntp)*tE
  tE1grid = np.logspace(np.log10(0.15e-3),np.log10(80e-3),NE1)

  # make tE1grid match integer multiples of the echo time
  tE1grid = tE*np.unique(np.round(tE1grid/tE))
  NE1 = tE1grid.size

  if True:
    # Generate a synthetic D-T2 distribution: a single Gaussian
    gmean = 0.16
    T2mean = 0.18
    gsd = 0.5
    T2sd = 0.1

    T2a = np.exp( -(np.log10(T2grid)-np.log10(T2mean))**2/(2*T2sd**2) )
    ga = np.exp( -(np.log10(ggrid)-np.log10(gmean))**2/(2*gsd**2) )
    ST2g = np.outer(ga,T2a)/100
    Sq2 = np.sum(np.abs(ST2g)**2)
    
    # Contour plot of the distribution
    plt.figure(0)
    plt.clf()
    plt.contour(T2grid,ggrid,ST2g)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relaxation T2 (s)')
    plt.ylabel('Gradient (T/m)')


  KtpT2 = np.exp(-np.outer(tpgrid,1/T2grid))
  KgtE1 = np.exp(-np.outer(tE1grid**3,D0/12*(gamma*ggrid)**2))
  KT2tE1 = np.exp(-np.outer(tE1grid,1/T2grid))
  MM = (KT2tE1 * (KgtE1 @ ST2g)) @ KtpT2.T

  plt.figure(1)
  plt.clf()
  plt.contour(tpgrid*1e3,tE1grid*1e3,MM)
  plt.xlabel('Time t prime (ms)')
  plt.ylabel('1st echo time TE1 (ms)')

  MM1 = MM + np.random.randn(NE1,Ntp)*np.max(np.abs(MM))*sigma

  alpha = 1e-5
  S1,residb = flint(KgtE1,KtpT2,MM1,alpha,K3=KT2tE1)
  flint1SE = np.sum(np.abs(ST2g-S1)**2)

  # Contour plot of the distribution recovered using full data set
  plt.figure(2)
  plt.clf()
  plt.contour(T2grid,ggrid,S1)
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('Relaxation T2 (s)')
  plt.ylabel('Gradient (T/m)')

  if True:
    # Compare the results with those using a separable sub-sample
    # of the data
    
    # Form smaller, separable, grid
    tcut = 58e-3
    tE1grid2 = tE1grid[tE1grid<tcut]
    Ntt = Ntp - int(np.round(tE1grid2[-1]/tE))
    ttgrid = tE1grid2[-1] + np.arange(Ntt)*tE
    
    KttT2 = np.exp(-np.outer(ttgrid,1/T2grid))
    KgtE12 = np.exp(-np.outer(tE1grid2**3,D0/12*(gamma*ggrid)**2))
    MM2 = KgtE12 @ (ST2g @ KttT2.T)
    
    tEndx = np.round(tE1grid2/tE).astype(int)
    tEndx2 = tEndx[-1]-tEndx
    MM3 = np.empty((tE1grid2.size,Ntt))

    # Check data is identical to the larger grid
    for ii in range(tE1grid2.size):
      MM3[ii,:] = MM[ii,tEndx2[ii]:tEndx2[ii]+Ntt]
    MM23err=np.max(np.abs(MM2-MM3))
    
    for ii in range(tE1grid2.size):
      MM3[ii,:] = MM1[ii,tEndx2[ii]:tEndx2[ii]+Ntt]
    # Let this method cheat by initialising with results for full dataset  
    Sflint0,lossa = flint(KgtE12,KttT2,MM3,alpha,SS=S1,tol=1e-7)
    #Sflint0,lossa = fl.flint(KgtE12,KttT2,MM3,alpha,S1,1e-7)
    flint0SE = np.sum(np.abs(ST2g-Sflint0)**2)
    
    # Contour plot of the distribution recovered using separable sub-sample
    plt.figure(3)
    plt.clf()
    plt.contour(T2grid,ggrid,Sflint0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relaxation T2 (log10(s))')
    plt.ylabel('Gradient (log10(T/m))')

    print('MM23err= %1.2e'%MM23err)
    print('Squared error from rectangular sub-sample= %1.2e'%(flint0SE/Sq2))
  
  print('Squared error from full data = %1.2e'%(flint1SE/Sq2))
  
