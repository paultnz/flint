function [SS,resida] = flint(K1,K2,ZZ,alpha,SS,tol,maxiter,progress)
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
% 1.1  2 Dec 2020 Changed calculation of Lipshitz constant, and introduced
%                 some optional arguments

% If you use this software, please cite P.D. Teal and C. Eccles. Adaptive
% truncation of matrix decompositions and efficient estimation of NMR
% relaxation distributions. Inverse Problems, 31(4):045010, April
% 2015. http://dx.doi.org/10.1088/0266-5611/31/4/045010 (Section 4: although
% the Lipshitz constant there does not have alpha added as it should have)

% Y is the NMR data for inversion
% alpha is the (Tikhonov) regularisation (scalar) 
% S is an optional starting estimate

% K1 and K2 are the kernel matrices (e.g., T1 relaxation and T2 relaxation)
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

if nargin<8  progress = 500;                         end
if nargin<7  maxiter  = 100000;                      end
if nargin<6  tol      = 1e-5;                        end
if nargin<5  SS       = ones(size(K1,2),size(K2,2)); end

resida = NaN(maxiter,1);

KK1 = K1'*K1;
KK2 = K2'*K2;
KZ12 = K1'*ZZ*K2;
tZZ = trace(ZZ*ZZ');       % used for calculating residual

% Find the Lipschitz constant: The kernel matrices are poorly
% conditioned so this will typically converge very fast
SL = SS;
LL = inf;
for ii=1:100
  lastLL = LL;
  LL = sqrt(sum(SL(:).^2));
  if abs(LL-lastLL)/LL < 1e-10
    break
  end
  SL = SL/LL;
  SL = KK1 * SL * KK2;
end
LL = 1.001 * 2 * (LL + alpha);
fprintf('Lipschitz constant found: ii= % 2d, LL= % 1.3e\n',ii,LL)

YY = SS;
tt = 1;
fac1 = (LL-2*alpha)/LL;
fac2 = 2/LL;
lastres = inf;

for iter=1:maxiter
  term2 = KZ12-KK1*YY*KK2;
  Snew = fac1*YY + fac2*term2;
  Snew = max(0,Snew);
    
  ttnew = 0.5*(1 + sqrt(1+4*tt^2));
  trat = (tt-1)/ttnew;
  YY = Snew + trat * (Snew-SS);
  tt = ttnew;
  SS = Snew;

  if ~mod(iter,progress)
    % Don't calculate the residual every iteration; it takes much longer
    % than the rest of the algorithm
    normS = alpha*norm(SS(:))^2;
    resid = tZZ -2*trace(SS'*KZ12) + trace(SS'*KK1*SS*KK2) + normS;
    if nargout>1
      resida(iter) = resid;
    end
    resd = (lastres-resid)/resid;
    lastres = resid;
    % show progress
    fprintf('%7d % 1.2e % 1.2e % 1.4e % 1.4e \n',...
	    iter,tt,1-trat,resid,resd);
    if abs(resd)<tol
      return;
    end
  end
end
