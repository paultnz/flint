% Demonstration of Fast 2D D-T2 distribution estimation
% Paul Teal, Victoria University of Wellington
% paul.teal@vuw.ac.nz
% Thursday, 3 December 2020
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
[S1,residb] = flintDT2(KgtE1,KtpT2,MM1,alpha,1,0,KT2tE1);
flint1SE = sum(abs(ST2g(:)-S1(:)).^2);


figure(3);
contour(log10(T2grid),log10(ggrid),S1);
xlabel('Relaxation T2 (log10(s))');
ylabel('Gradient (log10(T/m))');

figure(4);
f1 = find(isfinite(residb));
residb = residb(1:f1(end));
semilogy(residb,'x');
xlabel('Iteration number');
ylabel('Opimisation metric');

if 1==1
  % Form smaller, separable, grid
  tcut = 58e-3;
  tE1grid2 = tE1grid(tE1grid<tcut);
  Ntt = Ntp - round(tE1grid2(end)/tE);
  ttgrid = tE1grid2(end) + (0:Ntt-1)'*tE;
  KttT2 = exp(- ttgrid * (1./T2grid)' );
  KgtE12 = exp(- (tE1grid2.^3) * (D0/12*(gamma*ggrid).^2)' );
  MM2 = (KgtE12 * (ST2g * KttT2'))';

  tEndx = round(tE1grid2/tE);
  tEndx2 = tEndx(end)-tEndx;
  MM3 = zeros(length(tE1grid2),Ntt);
  for ii=1:length(tE1grid2)
    MM3(ii,:) = MM(ii,tEndx2(ii)+1:tEndx2(ii)+Ntt);
  end
  MM23err=max(abs(MM2(:)-MM3(:)));
  
  for ii=1:length(tE1grid2)
    MM3(ii,:) = MM1(ii,tEndx2(ii)+1:tEndx2(ii)+Ntt);
  end
  
  KS1 = KtpT2 * (KT2tE1 .* (KgtE1 * S1))';
  resid1 = sum( (MM1(:)-KS1(:)).^2 );
  KS0 = KttT2 * ( KgtE12 * S1 )';
  resid0 = sum( (MM3(:)-KS0(:)).^2 );
  
  % Let this method cheat by initialising with results for full dataset  
  S1n = S1 + 1e-3*randn(size(S1));
  [Sflint0,lossa] = flintDT2(KgtE12,KttT2,MM3,alpha,S1n,1e-7,500000);
  flint0SE = sum(abs(ST2g(:)-Sflint0(:)).^2);
  
  figure(5);
  contour(T2grid,ggrid,Sflint0);
  set(gca,'xscale','log');
  set(gca,'yscale','log');
  xlabel('Relaxation T2 (s)');
  ylabel('Gradient (T/m)');
 
  fprintf('MM23err= %1.2e \n',MM23err)
  fprintf('Squared error from rectangular sub-sample= %1.2e \n',flint0SE/Sq2);
end
fprintf('Squared error from full data = %1.2e \n',flint1SE/Sq2);
