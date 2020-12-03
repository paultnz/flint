# Fast 2D NMR relaxation distribution estimation - numpy version
# Paul Teal, Victoria University of Wellington
# paul.teal@vuw.ac.nz
#
# Let me know of feature requests, and if you find this algorithm does
# not perform as it should, please send me the data-set, so I can improve it.
# Issued under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3.
# If you distribute this code or a derivative work you are required to make the
# source available under the same terms

# Versions:
# 0.1 30 October 2013 (Matlab/Octave versions)
# 0.2 31 October 2013
# 0.3 20 December 2013
# 0.4  1 May 2014
# 0.5  6 May 2014
# 0.6 11 Aug 2014
# 0.7  2 Aug 2016
# 0.8  6 Sep 2016
# 1.0 15 Sep 2016
# 1.1  4 Dec 2020 Changed calculation of Lipshitz constant, and introduced
#                 some optional arguments and released numpy version

# If you use this software, please cite P.D. Teal and C. Eccles. Adaptive
# truncation of matrix decompositions and efficient estimation of NMR
# relaxation distributions. Inverse Problems, 31(4):045010, April
# 2015. http://dx.doi.org/10.1088/0266-5611/31/4/045010

import numpy as np

def flint(K1,K2,ZZ,alpha,SS=np.asarray([]),tol=1e-5,maxiter=100001,progres=500):
  """
  FLINT: Fast Lapace-like INverTer (2D)

  K1 is the T1 relaxation kernel matrix (size Nechos x NT1)
    (set this to 1 if processing a 1D T2 experiment)
  K2 is the T2 relaxation kernel matrix (size Nechodelays x NT2)
    (set this to 1 if processing a 1D T1 experiment)
  ZZ is the NMR data for inversion  (size Nechos x Nechodelays)
  alpha is the (Tikhonov) regularisation (scalar)
  SS is an optional starting estimate
  tol is the optional relative change between successive calculations
     for exit
  maxiter is the optional maximum number of iterations
  progres is the optional number of iterations between progress displays
    should be at least several hundred, because calculating the error is slow
  """

  resida = np.full((maxiter), np.nan)

  if SS.size==0:
    SS = np.ones((K1.shape[1],K2.shape[1]))  # initial estimate

  KK1 = K1.T@K1
  KK2 = K2.T@K2
  KZ12 = K1.T@ZZ@K2
  tZZ = np.trace(ZZ@ZZ.T)       # used for calculating residual

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
    SL = KK1@SL@KK2
  LL = 1.001 * 2 * (LL + alpha)
  print('Lipschitz constant found: ii= % 2d, LL= % 1.3e'%(ii,LL))

  YY = SS
  tt = 1
  fac1 = (LL-2*alpha)/LL
  fac2 = 2/LL
  lastres = np.inf

  for iter in range(maxiter):
    term2 = KZ12 - KK1 @ YY @ KK2
    Snew = fac1*YY + fac2*term2
    Snew[Snew<0] = 0.
    ttnew = 0.5*(1 + np.sqrt(1+4*tt**2))
    trat = (tt-1)/ttnew
    YY = Snew + trat * (Snew-SS)
    tt = ttnew
    SS = Snew

    if iter % progres == 0:
      # Don't calculate the residual every iteration; it takes much longer
      # than the rest of the algorithm
      normS = alpha * np.sum(SS**2)
      resid = tZZ -2*np.trace(SS.T@KZ12) + np.trace(SS.T@KK1@SS@KK2) + normS
      resida[iter] = resid
      resd = (lastres-resid)/resid
      lastres = resid
      # show progress
      print('%7d % 1.2e % 1.2e % 1.4e % 1.4e'%(iter,tt,1-trat,resid,resd))
      if np.abs(resd)<tol:
        return SS,resida
  return SS,resida

if __name__ == "__main__":
  # Example use of the flint() function

  N1 = 50       # number of data points in each dimension
  N2 = 10000
  Nx = 100      # number of bins in relaxation time grids
  Ny = 101
  tau1min = 1e-4
  tau1max = 10
  deltatau2 = 3.5e-4
  T1 = np.logspace(-2,1,Nx)
  T2 = np.logspace(-2,1,Ny)
  tau1 = np.logspace(np.log10(tau1min),np.log10(tau1max),N1)
  tau2 = (1+np.arange(N2))*deltatau2
  K2 = np.exp(-np.outer(tau2,1/T2))     # simple T2 relaxation data
  K1 = 1-2*np.exp(-np.outer(tau1,1/T1)) # T1 relaxation data
  T2a,T1a = np.meshgrid(np.log10(T2),np.log10(T1))
  if True:
    # Generate some synthetic data
    Ftrue = np.zeros((Nx,Ny))

    centre1 = [-0.5,-1]
    radius1 = [0.35,0.75]
    centre2 = [-1.5,0.3]
    radius2 = 0.2
    centre3 = [0.5,0.3]
    radius3 = 0.2

    dist1 = np.sqrt( (T2a-centre1[0])**2 + (T1a-centre1[1])**2 )
    dist2 = np.sqrt( (T2a-centre2[0])**2 + (T1a-centre2[1])**2 )
    dist3 = np.sqrt( (T2a-centre3[0])**2 + (T1a-centre3[1])**2 )
    ang1 = np.arctan2( T1a-centre1[1], T2a-centre1[0])

    Ftrue[np.logical_and(np.logical_and(radius1[0] < dist1,
                                        dist1<radius1[1]), ang1<=0)] = 0.4
    Ftrue[dist2<radius2] = 0.18
    Ftrue[dist3<radius3] = 0.58

  Mmeas_cln = K1@Ftrue@K2.T   # without noise
  sigma = np.max(Mmeas_cln)*1e-3
  noise = sigma*np.random.randn(N1,N2)
  Mmeas = Mmeas_cln + noise  # with noise

  alpha = 1e-1
  Sflint,resa = flint(K1,K2,Mmeas,alpha)

  plt.figure(1)
  plt.clf()
  plt.semilogy(resa,'+')
  plt.xlabel('Iteration number')
  plt.ylabel('Optimisation metric')

  from mpl_toolkits.mplot3d import Axes3D
  fig1=plt.figure(2)
  plt.clf()
  ax1 = fig1.gca(projection='3d')
  ax1.plot_surface(T2a, T1a, Ftrue)
  plt.xlabel('y (T2)')
  plt.ylabel('x (T1)')
  ax1.view_init(elev=55, azim=240)

  fig2=plt.figure(3)
  plt.clf()
  ax2 = fig2.gca(projection='3d')
  ax2.plot_surface(T2a, T1a, Sflint)
  plt.xlabel('y (T2)')
  plt.ylabel('x (T1)')
  ax2.view_init(elev=55, azim=240)
  plt.show()
