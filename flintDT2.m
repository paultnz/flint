function [SS,lossa]=flintDT2(K1,K2,ZZ,alpha,SS,lambda,K3,tol,maxiter,progress)
% FLINTDT2: Fast Lapace-like INverTer (2D)
% Fast 2D D-T2 distribution estimation - Matlab/octave version
% Paul Teal, Victoria University of Wellington
% paul.teal@vuw.ac.nz
% Let me know of feature requests, and if you find this algorithm does 
% not perform as it should, please send me the data-set, so I can improve it.
% Issued under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3.
% If you distribute this code or a derivative work you are required to make the
% source available under the same terms
%
% Based on flint.m
% Version 0.4, Friday 4 December 2020
% Modified for distribution, Version 1.0, Wednesday, 15 December 2021 
%
% If you use this software, please cite P.D. Teal and E.H Novotny. Improved
% data efficiency for NMR diffusion-relaxation processing. Journal of
% Magnetic Resonance, 2021. http://dx.doi.org/10.1016/j.jmr.2021.107124
%
%  K1 is the relaxation kernel matrix (size Nechos x NT2)
%  K2 is the diffusion kernel matrix (size Nechodelays x Ndiffusion)
%  ZZ is the NMR data for inversion  (size Nechos x Nechodelays)
%  alpha is the (Tikhonov) regularisation (scalar)
%  SS is an optional starting estimate
%  lambd is the l1 regularisation multiplier (scalar)
%  K3 is the "cross-kernel" matrix that couples the other two
%     (making them dependent)  (size Nechodelays x NT2)
%  tol is the optional relative change between successive calculations
%     for exit
%  maxiter is the optional maximum number of iterations
%  progress is the optional number of iterations between progress displays
%    should be at least several hundred, because calculating the error is slow

K3present = 1;
if nargin<10 progress  = 500;                         end
if nargin<9  maxiter   = 300001;                      end
if nargin<8  tol       = 1e-5;                        end
if nargin<7  K3        = 1; K3present = 0;            end
if nargin<6  lambda    = 0;                           end
if nargin<5  SS        = ones(size(K1,2),size(K2,2)); end
if prod(size(SS))==1 SS= ones(size(K1,2),size(K2,2)); end
  

lossa = NaN(maxiter,1);

KK2 = K2'*K2;
if K3present
  KZ12 = K1' * (K3 .* (ZZ * K2) );
else
  KK1 = K1' * K1;
  KZ12 = K1' * ZZ * K2;
end

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
  if K3present
    SL = K1' * (K3 .* ((K3 .* (K1 * SL)) * KK2 ));
  else
    SL = KK1 * SL * KK2;
  end
end
LL = 1.001 * 2 * (LL + alpha);
fprintf('Lipschitz constant found: ii= % 2d, LL= % 1.3e\n',ii,LL)

YY = SS;
tt = 1;
fac1 = (LL-2*alpha)/LL;
fac2 = 2/LL;
fac3 = lambda/LL;
lastres = inf;

for iter = 1:maxiter
  if K3present
    term2 = KZ12 - K1' * (K3 .* ((K3 .* (K1 * YY)) * KK2 ));
  else
    term2 = KZ12 - KK1 * YY * KK2;
  end
  Snew = fac1*YY + fac2*term2 - fac3;
  Snew = max(0,Snew);
  ttnew = 0.5*(1 + sqrt(1+4*tt^2));
  trat = (tt-1)/ttnew;
  YY = Snew + trat * (Snew-SS);
  tt = ttnew;
  SS = Snew;

  if ~mod(iter,progress)
    % Don't calculate the residual every iteration -
    % that would slow things down a lot
    normS = alpha * sum(SS(:).^2) + lambda * sum(abs(SS(:)));
    KS = (K3 .* (K1 * SS)) * K2';
    resid = sum( (ZZ(:)-KS(:)).^2 ) + normS;
    lossa(iter) = resid;
    resd = (lastres-resid)/resid;
    lastres = resid;
    % show progress
    fprintf('%7d % 1.2e % 1.2e % 1.4e % 1.4e \n',...
      iter,tt,1-trat,resid,resd);
    if abs(resd)<tol
      return
    end
  end
end
