function demo_proMP_StochasticFourierSampling
addpath('./m_fcts/');

%% Parameters
nbData = 200; %Number of datapoints in a trajectory
nbSamples = 10; %Number of demonstrations
nbVar = 2; %Dimension of position data

%% Load handwriting data
demos=[]
load('data/2Dletters/B.mat');
for n=1:nbSamples
	dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData/2)); %Resampling
	dTmp2 = fliplr(dTmp);
	x(:,n) = [dTmp(:); dTmp2(:)]; 
end

%Simulate phase variations
for n=2:nbSamples
	x(:,n) = circshift(x(:,1),(n-nbSamples/2-1)*4);
end


%% ProMP with Fourier basis functions
k = -8:8;
t = linspace(0,1,nbData);
phi = exp(t' * k * 2 * pi * 1i) ./ nbData;
Psi = kron(phi, eye(nbVar)); 
w = pinv(Psi) * x;
nbFct = size(w,1);

%Distribution in parameter space
Mu_m = mean(abs(w), 2); %Magnitude average
Mu_p = mean_angle(angle(w), 2); %Phase average
Mu_w = Mu_m .* exp(1i * Mu_p); %Reconstruction
Mu = Psi * Mu_w; %Reconstruction
Sigma_m = cov(abs(w')); %Magnitude spread
Sigma_p = cov_angle(angle(w')); %Phase spread
[V_m, D_m] = eig(Sigma_m);
U_m = V_m * D_m.^.5;
[V_p, D_p] = eig(Sigma_p);
U_p = V_p * D_p.^.5;

%% Stochastic reproductions
nbRepros = 5;
%Reproductions with learned variations
xw = (repmat(Mu_m,1,nbRepros) + U_m * randn(nbFct,nbRepros)) .* exp(1i * (Mu_p + U_p * randn(nbFct,nbRepros)));
xr = Psi * xw;

%% Plot 
%Plot signal
figure('position',[10 10 2500 1000],'color',[1,1,1]); 
for i=1:nbVar
	axLim = [1, nbData, min(real(Mu(i:nbVar:end)))-1, max(real(Mu(i:nbVar:end)))+1];
	subplot(2,3,(i-1)*3+[1:2]); hold on;
	for n=1:nbSamples
		h(1) = plot(x(i:nbVar:end,n), '--','lineWidth',3,'color',[0 0.4470 0.7410]);
	end
	for n=1:nbRepros
		h(2) = plot(real(xr(i:nbVar:end,n)), '-','lineWidth',2,'color',[0.9290 0.6940 0.1250]);
		plot(imag(xr(i:nbVar:end,n)), '-','lineWidth',2,'color',[1 .7 0]);
	end
	h(3) = plot(real(Mu(i:nbVar:end)), '-','lineWidth',2,'color',[.8 0 0]);
	h(4) = plot(imag(Mu(i:nbVar:end)), '--','lineWidth',2,'color',[.8 0 0]);
	patch([nbData/2 nbData/2 nbData+2 nbData+2], [axLim([3,4]), axLim([4,3])], [1 1 1],'edgecolor','none','facecolor',[1 1 1],'facealpha',.5);
	plot([nbData/2 nbData/2], axLim(3:4), ':','lineWidth',2,'color',[0 0 0]);
	axis(axLim);
	set(gca,'xtick',[],'ytick',[],'linewidth',2);
	xlabel('$t$','interpreter','latex','fontsize',12); ylabel('$x_1$','interpreter','latex','fontsize',12);
end
legend(h,{'Demonstrations','Stochastically generated samples','Re($\mu^x$)','Im($\mu^x$)'}, 'interpreter','latex','location','north','fontsize',8);


%2D plot
subplot(2,3,[3,6]);
hold on;
axis off;
for n=1:nbSamples
	plot(x(1:nbVar:nbData,n), x(2:nbVar:nbData,n), '-','lineWidth',3,'color',[.7 0 0]);
end
plot(real(Mu(1:nbVar:nbData)), real(Mu(2:nbVar:nbData)), '--','lineWidth',3,'color',[0 .2 .3]);
for n=1:nbRepros
	plot(real(xr(1:nbVar:nbData,n)), real(xr(2:nbVar:nbData,n)), '--','lineWidth',2,'color',[1 .7 .7]);
end
axis equal;
pause;
close all;
end

function Mu = mean_angle(phi, dim)
	if nargin<2
		dim = 1;
	end
	Mu = angle(mean(exp(1i*phi), dim));
end
function Sigma = cov_angle(phi)
	Mu = mean_angle(phi);
	e = phi - repmat(Mu, size(phi,1), 1);
	Sigma = cov(angle(exp(1i*e)));
end