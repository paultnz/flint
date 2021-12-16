% Demonstration of Fast 2D D-T2 distribution estimation
% Paul Teal, Victoria University of Wellington
% paul.teal@vuw.ac.nz
% Friday, 4 December 2020
% Modified for distribution Wednesday, 15 December 2021

clear all

% Set up grids in both domains
Ng = 101;
NT2 = 100;
NE1 = 34;  % was supposed to be 30, but making integer multiples of
           % the echo time results in some duplicates
Ntp = 5000;
tE = 0.15e-3;
D0 = 2.3e-9;            % diffusion coefficinet for water
gamma = 267.52218744e6; % gyro-magnetic ratio for proton
sigma = 1e-2;

T2grid = logspace(-2,1,NT2)';
ggrid = logspace(-3,2,Ng)';
tpgrid = (0:Ntp-1)'*tE;
tE1grid = logspace(log10(0.15e-3),log10(80e-3),NE1)';

% make tE1grid match integer multiples of the echo time
tE1grid = tE*unique(round(tE1grid/tE));
NE1 = length(tE1grid);

if 1==1
  % Generate a synthetic D-T2 distribution: a single Gaussian
  gmean = 0.16;
  T2mean = 0.18;

  gsd = 0.5;
  T2sd = 0.1;

  T2a = exp( -(log10(T2grid)-log10(T2mean)).^2/(2*T2sd^2) );
  ga = exp( -(log10(ggrid)-log10(gmean)).^2/(2*gsd.^2) );
  ST2g = ga * T2a'/100;
  Sq2 = sum(abs(ST2g(:)).^2);
end

% Contour plot of the distribution
figure(1);
contour(T2grid,ggrid,ST2g);
set(gca,'xscale','log');
set(gca,'yscale','log');
xlabel('Relaxation T2 (s)');
ylabel('Gradient (T/m)');

KtpT2 = exp(-tpgrid * (1./T2grid)' );
KgtE1 = exp(-(tE1grid.^3) * (D0/12*(gamma*ggrid).^2)');
KT2tE1 = exp(-tE1grid * (1./T2grid)');
MM = (KT2tE1 .* (KgtE1 * ST2g)) * KtpT2';

figure(2);
contour(tpgrid*1e3,tE1grid*1e3,MM);
xlabel('Time t prime (ms)');
ylabel('1st echo time TE1 (ms)');

MM1 = MM + randn(NE1,Ntp)*max(abs(MM(:)))*sigma;
alpha = 1e-5;
lambda = 1e2;
[S1,residb] = flintDT2(KgtE1,KtpT2,MM1,alpha,1,lambda,KT2tE1);
flint1SE = sum(abs(ST2g(:)-S1(:)).^2);

figure(3);
contour(T2grid,ggrid,S1);
set(gca,'xscale','log');
set(gca,'yscale','log');
xlabel('Relaxation T2 (s)');
ylabel('Gradient (T/m)');



% Now sum along the T2 dimension to estimate the gradient distribution
hg = sum(S1,2);
figure(4);
semilogx(ggrid,hg);

% Esimate D and T2 distribution of water using this gradient distribution
ND = 102;
Dgrid = logspace(-11,-6,ND)';
ST2D = zeros(ND,NT2);
[null,gi] = min(abs(Dgrid-D0));
T20 = 0.2;  % T2 of water
[null,T2i] = min(abs(T2grid-T20));
ST2D(gi,T2i) = 1; 

% Generate diffusion kernel
E3 = tE1grid.^3;
KgtE1DD = zeros(NE1,ND);
for gi = 1:Ng
  KgtE1DD = KgtE1DD + hg(gi) * exp(-E3 * (Dgrid/12*(gamma*ggrid(gi)).^2)');
end

% Generate synthetic data, add noise, and do the estimation
MMD = (KT2tE1 .* (KgtE1DD * ST2D)) * KtpT2';
MM2 = MMD + randn(NE1,Ntp)*max(abs(MMD(:)))*sigma;
[S2,residb] = flintDT2(KgtE1DD,KtpT2,MM2,alpha,1,lambda,KT2tE1);
flint2SE = sum(abs(ST2D(:)-S2(:)).^2);

figure(5);
contour(T2grid,Dgrid,S2);
set(gca,'xscale','log');
set(gca,'yscale','log');
xlabel('Relaxation T2 (s)');
ylabel('Diffusion coefficient (m^2/s)');

fprintf('Squared error of gradient  distribution = %1.2e \n',flint1SE/Sq2);
fprintf('Squared error of diffusion distribution = %1.2e \n',flint2SE);
