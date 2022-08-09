function demo_proMP01
addpath('./m_fcts/');


%% Parameters
numofbasisfunctions = 8; %Number of basis functions
Posdatadimension = 2; %Dimensions of data
numofdemonstrations = 5; %Number of demonstrations referred to
numofdatapt = 200; %Number of datapoints in a trajectory
numofreproduction = 5; %Number of reproductions


%% Load the mat files with the handwriting data
demos=[];
load('data/2Dletters/A.mat'); 
%load('data/2Dletters/Z.mat');
x=[]; %temporary variable
for n=1:numofdemonstrations
	s(n).x = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),numofdatapt)); %Resampling
	x = [x, s(n).x(:)]; 
end
t = linspace(0,1,numofdatapt);


%% Part 1: ProMP with radial basis functions 

% Compute the value of the basis functions psi and their activation weights w
t_mu = linspace(t(1), t(end), numofbasisfunctions);
m(1).valphi = zeros(numofdatapt,numofbasisfunctions);
for i=1:numofbasisfunctions
	m(1).valphi(:,i) = gaussPDF(t, t_mu(i), 1E-2);
end
m(1).valpsi = kron(m(1).valphi, eye(Posdatadimension));
m(1).w = (m(1).valpsi' * m(1).valpsi + eye(Posdatadimension*numofbasisfunctions).*1E-8) \ m(1).valpsi' * x;
%Distribution in parameter space
m(1).weightMu = mean(m(1).w,2);
m(1).weightSigma = cov(m(1).w') + eye(Posdatadimension*numofbasisfunctions) * 1E0; 
%Trajectory distribution
m(1).Mu = m(1).valpsi * m(1).weightMu;
m(1).Sigma = m(1).valpsi * m(1).weightSigma * m(1).valpsi';


%% ProMP with Bernstein basis functions

%Compute basis functions Psi and activation weights w
m(2).valphi = zeros(numofdatapt,numofbasisfunctions);
for i=0:numofbasisfunctions-1
	m(2).valphi(:,i+1) = factorial(numofbasisfunctions-1) ./ (factorial(i) .* factorial(numofbasisfunctions-1-i)) .* (1-t).^(numofbasisfunctions-1-i) .* t.^i; %Bernstein basis functions
end
m(2).valpsi = kron(m(2).valphi, eye(Posdatadimension));
m(2).w = (m(2).valpsi' * m(2).valpsi + eye(Posdatadimension*numofbasisfunctions).*1E-8) \ m(2).valpsi' * x;
%Distribution in parameter space
m(2).weightMu = mean(m(2).w,2);
m(2).weightSigma = cov(m(2).w') + eye(Posdatadimension*numofbasisfunctions) * 1E0; 
%Trajectory distribution
m(2).Mu = m(2).valpsi * m(2).weightMu;
m(2).Sigma = m(2).valpsi * m(2).weightSigma * m(2).valpsi';


%% Part 3: ProMP with Fourier basis functions (here, only DCT)

%Compute basis functions Psi and activation weights w
m(3).valphi = zeros(numofdatapt,numofbasisfunctions);
for i=1:numofbasisfunctions
	tempoval = zeros(1,numofdatapt);
	tempoval(i) = 1;
	m(3).valphi(:,i) = idct(tempoval);
end	
m(3).valpsi = kron(m(3).valphi, eye(Posdatadimension));
m(3).w = (m(3).valpsi' * m(3).valpsi + eye(Posdatadimension*numofbasisfunctions).*1E-8) \ m(3).valpsi' * x;
%Distribution in parameter space
m(3).weightMu = mean(m(3).w,2);
m(3).weightSigma = cov(m(3).w') + eye(Posdatadimension*numofbasisfunctions) * 1E0; 
%Trajectory distribution
m(3).Mu = m(3).valpsi * m(3).weightMu;
m(3).Sigma = m(3).valpsi * m(3).weightSigma * m(3).valpsi';


%% Conditioning with trajectory distribution

inputindice_time = [1,numofdatapt]; %Time steps input indices
outputindice_time = 2:numofdatapt-1; %Time steps output indices
in = [];
for i=1:length(inputindice_time)
	in = [in, (inputindice_time(i)-1)*Posdatadimension+[1:Posdatadimension]]; %Trajectory distribution input indices
end
out = [];
for i=1:length(outputindice_time)
	out = [out, (outputindice_time(i)-1)*Posdatadimension+[1:Posdatadimension]]; %Trajectory distribution output indices
end

%Reproduction by Gaussian conditioning
for k=1:3
	for n=1:numofreproduction
		m(k).Mu_sec(in,n) = x(in,1) + repmat((rand(Posdatadimension,1)-0.5)*2, length(inputindice_time), 1) ;
		%Efficient computation of conditional distribution by exploiting ProMP structure
		m(k).Mu_sec(out,n) = m(k).valpsi(out,:) * ...
			(m(k).weightMu + m(k).weightSigma * m(k).valpsi(in,:)' / (m(k).valpsi(in,:) * m(k).weightSigma * m(k).valpsi(in,:)') * (m(k).Mu_sec(in,n) - m(k).valpsi(in,:) * m(k).weightMu));
	end
end


%% Plot 

figure('position',[10 10 1800 1300]); 
clrmap = lines(numofbasisfunctions);
methods = {'RBF','BBF','FBF'};


for k=1:3

	%Plot the signals
	subplot(3,3,k); hold on; axis off; title(methods{k},'fontsize',16);
	plot(x(1:2:end,:), x(2:2:end,:), '.','markersize',10,'color',[.7 .7 .7]);
	for n=1:numofreproduction
		plot(m(k).Mu_sec(1:2:end,n), m(k).Mu_sec(2:2:end,n), '*','lineWidth',2,'color',[1 .6 .6]);
	end
	plot(m(k).Mu(1:2:end), m(k).Mu(2:2:end), '--','lineWidth',2,'color',[0 0 0]);
	axis tight; axis equal; 


	%Plot the activation functions
	subplot(3,3,3+k); hold on; axis off; title('\phi_k','fontsize',12);
	for i=1:numofbasisfunctions
		plot(1:numofdatapt, m(k).valphi(:,i),'linewidth',2,'color',clrmap(i,:));
	end
	axis([1, numofdatapt, min(m(k).valphi(:)), max(m(k).valphi(:))]);


	%Plot the matrix for Ψ and Ψ'
	subplot(3,3,6+k); hold on; axis off; title('\psi\psi^T','fontsize',12);
	colormap(flipud(gray));
	imagesc(abs(m(k).valpsi * m(k).valpsi'));
	axis tight; axis square; axis ij;
end
pause;
close all;