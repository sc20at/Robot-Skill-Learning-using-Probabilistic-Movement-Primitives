function GMMvsGMR
addpath('./m_fcts/');

%% initialise the parameters
model.nbStates = 5; %Number of states in the GMM
model.Posdatadimension = 5; %Number of variables time and the 4 axes
model.dt = 0.01; %Time step duration
numofdatapt = 50; %Length of each trajectory
numofdemonstrations = 10; %Number of demonstrations


%% Load handwritten data
demos=[];
load('data/2Dletters/N.mat'); %Load variables for x1,x2 
for n=1:numofdemonstrations
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),numofdatapt)); %Resampling the handwritten data
end
demos=[];
load('data/2Dletters/A.mat'); %Load variables for x3,x4
Data=[];
for n=1:numofdemonstrations
	s(n).Data = [[1:numofdatapt]*model.dt; s(n).Data; spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),numofdatapt))]; %Resampling the handwritten data
	Data = [Data s(n).Data]; 
end


%% Learning and reproduction
model = init_GMM_timeBased(Data, model);
model = EM_GMM(Data, model);
[DataOut, SigmaOut] = GMR(model, [1:numofdatapt]*model.dt, 1, 2:model.Posdatadimension);


%% Graphical representation
disp('GMM and GMR');
figure('position',[10,10,1300,650]); 

%Plot GMM
subplot(1,2,1);
hold on;
box on;
title('GMM');
plotGMM3D(model.Mu(2:4,:), model.Sigma(2:4,2:4,:), [.8 0 0], .3);

for n=1:numofdemonstrations
	dTmp = [Data(2:4,(n-1)*numofdatapt+1:n*numofdatapt) fliplr(Data(2:4,(n-1)*numofdatapt+1:n*numofdatapt))];
	patch(dTmp(1,:),dTmp(2,:),dTmp(3,:), [.5,.5,.5],'facealpha',0,'linewidth',2,'edgecolor',[.5,.5,.5],'edgealpha',.5);
end

view(3);
axis equal;
axis vis3d;
set(gca,'Xtick',[]);
set(gca,'Ytick',[]);
set(gca,'Ztick',[]);
xlabel('x_1');
ylabel('x_2');
zlabel('x_3');

%Plot GMR
subplot(1,2,2);
hold on;
box on;
title('GMR');
plotGMM3D(DataOut(1:3,1:2:end), SigmaOut(1:3,1:3,1:2:end), [0 .8 0], .2, 2);

for n=1:numofdemonstrations
	dTmp = [Data(2:4,(n-1)*numofdatapt+1:n*numofdatapt) fliplr(Data(2:4,(n-1)*numofdatapt+1:n*numofdatapt))];
	patch(dTmp(1,:),dTmp(2,:),dTmp(3,:), [.5,.5,.5],'facealpha',0,'linewidth',2,'edgecolor',[.5,.5,.5],'edgealpha',.5);
end
plot3(DataOut(1,:),DataOut(2,:),DataOut(3,:),'-','linewidth',4,'color',[0 .4 0]);
view(3);
axis equal;
axis vis3d;
set(gca,'Xtick',[]);
set(gca,'Ytick',[]);
set(gca,'Ztick',[]);
xlabel('x_1');
ylabel('x_2');
zlabel('x_3');
pause;
close all;