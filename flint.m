function [S,resida] = flint(K1,K2,Z,alpha,S)
% Fast 2D NMR relaxation distribution estimation - Matlab/octave version
% Paul Teal, Victoria University of Wellington
% paul.teal@vuw.ac.nz
% Let me know of feature requests, and if you find this algorithm does 
% not perform as it should, please send me the data-set, so I can improve it.
% Issued under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3.
% If you distribute this code or a derivative work you are required to make the
% source available under the same terms

% Versions:
% 0.1 30 October 2013
% 0.2 31 October 2013
% 0.3 20 December 2013
% 0.4  1 May 2014
% 0.5  6 May 2014
% 0.6 11 Aug 2014
% 0.7  2 Aug 2016
% 0.8  6 Sep 2016  
% 1.0 15 Sep 2016  

% If you use this software, please cite P.D. Teal and C. Eccles. Adaptive
% truncation of matrix decompositions and efficient estimation of NMR
% relaxation distributions. Inverse Problems, 31(4):045010, April
% 2015. http://dx.doi.org/10.1088/0266-5611/31/4/045010 (Section 4: although
% the Lipshitz constant there does not have alpha added as it should have)

% Y is the NMR data for inversion
% alpha is the (Tikhonov) regularisation (scalar) 
% S is an optional starting estimate

% K1 and K2 are the kernel matrices
% They can be created with something like this:
%N1 = 50;       % number of data points in each dimension
%N2 = 10000;
%Nx = 100;      % number of bins in relaxation time grids
%Ny = 101;      
%tau1min = 1e-4;
%tau1max = 10;
%deltatau2 = 3.5e-4;
%T1 = logspace(-2,1,Nx);
%T2 = logspace(-2,1,Ny);
%tau1 = logspace(log10(tau1min),log10(tau1max),N1)';
%tau2 = (1:N2)'*deltatau2;  
%K2 = exp(-tau2 * (1./T2) );     % simple T2 relaxation data
%K1 = 1-2*exp(-tau1 *(1./T1) );  % T1 relaxation data

maxiter = 100000;

if nargin<5
  Nx = size(K1,2);  % N1 x Nx
  Ny = size(K2,2);  % N2 x Ny
  S = ones(Nx,Ny);  % initial estimate
end

if nargout>1
  resida = NaN(maxiter,1);
end

KK1 = K1'*K1;
KK2 = K2'*K2;
KZ12 = K1'*Z*K2;

% Lipschitz constant
L = 2 * (trace(KK1)*trace(KK2) + alpha); % trace will be larger than largest
                                         % eigenvalue, but not much larger
			   
tZZ = trace(Z*Z');       % used for calculating residual

Y = S;
tt = 1;
fac1 = (L-2*alpha)/L;
fac2 = 2/L;
lastres = inf;

for iter=1:maxiter
  term2 = KZ12-KK1*Y*KK2;
  Snew = fac1*Y + fac2*term2;
  Snew = max(0,Snew);
    
  ttnew = 0.5*(1 + sqrt(1+4*tt^2));
  trat = (tt-1)/ttnew;
  Y = Snew + trat * (Snew-S);
  tt = ttnew;
  S = Snew;

  if ~mod(iter,500)
    % Don't calculate the residual every iteration; it takes much longer
    % than the rest of the algorithm
    normS = alpha*norm(S(:))^2;
    resid = tZZ -2*trace(S'*KZ12) + trace(S'*KK1*S*KK2) + normS;
    if nargout>1
      resida(iter) = resid;
    end
    resd = abs(resid-lastres)/resid;
    lastres = resid;
    % show progress
    fprintf('%7i % 1.2e % 1.2e % 1.2e % 1.4e % 1.4e \n',...
	    iter,tt,trat,L,resid,resd);
    if resd<1e-5
      return;
    end
  end
end
