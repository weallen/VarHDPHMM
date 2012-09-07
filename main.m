%% Setup
addpath(genpath('HMMall'));
addpath(genpath('vbhmm'));
addpath(genpath('~/Documents/MATLAB/Lightspeed'));
%addpath('/course/cs195f/asgn/pmtk3-28feb11')
addpath('~/Documents/MATLAB/pmtk3/');
initPmtk3;
cd('~/Documents/CS/CSCI2950P/project/code');

%% Use PMTK3 to generate some test data
nstates = 9;
%pi = [0.25 0.25 0.25 0.25];
pi = ones(9,1)+ rand(9,1);
pi = pi ./ sum(pi);
%pi = ones(nstates,1) ./ nstates;
%A = [.985 .005 .005 .005;
%     .005 .985 0.005 0.005;
%     .005 0.005 .985 0.005; 
%     .005 0.005 0.005 .985];
%
A = [0.60 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05;
     0.20 0.70 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143;
     0.20 0.0143 0.70 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143;
     0.20 0.0143 0.0143 0.70 0.0143 0.0143 0.0143 0.0143 0.0143;
     0.20 0.0143 0.0143 0.0143 0.70 0.0143 0.0143 0.0143 0.0143;
     0.20 0.0143 0.0143 0.0143 0.0143 0.70 0.0143 0.0143 0.0143;
     0.20 0.0143 0.0143 0.0143 0.0143 0.0143 0.70 0.0143 0.0143;
     0.20 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143 0.70 0.0143;
     0.20 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143 0.70];
%A = A+normrnd(0.1,0.5,9)/10; %rand(9,9)/10.0;
A=mkStochastic(A);
 %A = [0 0.5 0 0.5;
%     0.5 0 0.5 0;
%     0.5 0 0 0.5;
%     0 0.5 0.5 0];
 %A = normalize(rand(5,5));



for k=1:nstates
    Sigma(:, :, k) = randpd(1) + 2*eye(1);
end

%emission = condGaussCpdCreate([0.0 10.0 20.0 30.0],Sigma);
T = zeros(nstates,20);
for i=1:nstates
    T(i,:) = randdirichlet(i*ones(20,1));
end
emission = tabularCpdCreate(T);

%hmm = hmmCreate('discrete', pi, A, emission);
gaussEmission = condGaussCpdCreate(5*randn(1, nstates), Sigma);
gaussHmm =  hmmCreate('gauss', pi, A, gaussEmission);
hmm = hmmCreate('discrete', pi, A, emission);

T = 1000;
%hmm = mkRndGaussHmm(4, 1);
[gaussData, gaussHidden] = hmmSample(gaussHmm, 1000, 1);

[data, hidden] = hmmSample(hmm, T, 1);
[test, hiddenTest] = hmmSample(hmm, 1000, 1);

%%
save 'data.mat',data, hidden;

%%
load data;

%%
rep = 10;

transmats = cell(rep,1);
obsmats = cell(rep,1);  
LLs = cell(rep,1);
for i=1:1
    i
    prior1 = normalise(rand(nstates,1));
    transmat1 = mk_stochastic(rand(nstates,nstates));
    obsmat1 = mk_stochastic(rand(nstates,20));
    %[prior1, transmat1, obsmat1] = randseginit(data,nstates, 20);
% improve guess of parameters using EM
    [LLs{i}, priors{i}, transmats{i}, obsmats{i}] = dhmm_em(data, prior1, transmat1, obsmat1, 'max_iter', 1000, 'thresh',1e-6,'verbose',true);
end
%%
maxll = -Inf;
bestModel = 1;

for i=1:10
    if max(LLs{i}) > maxll
        bestModel = i;
    end
end

emstates = viterbi_path(priors{bestModel}, transmats{bestModel},...
    obsmats{bestModel});
%% Fit with regular HMM
lls = {};
models = {};
for i=1:1
    i
    [model, hmmllhist] = hmmFit(data, 9, 'discrete');
    lls{i} = hmmllhist;
    models{i} = model;
end
maxll = -Inf;
bestModel = models{1};
for i=1:30
    if max(lls{i}) > maxll
        bestModel = models{i};
    end
end
hmmStates = hmmMap(bestModel, data);

%% 
blah = hmmFitEm(gaussData',4,'gauss');
imagesc(blah.A);

%% Fit with HDP-HMM

[model, llhist] = hdphmmFitVar(gaussData', 10, 'gauss');

%imagesc(model.A);
states = hmmMap(model, data);

%% Gaussian variational
[Wa, mus, lambdas, Wpi, F,softev] = dovaremGauss(gaussData,4,1000,1e-6);
states = viterbi_path(Wpi, Wa, softev);

%% EM algorithm
% when init with cheat: F:-3073.513  	
% when init with EM: F:-3120.595
% when init with rand seg: F:-3166.257

% randomly intialize variables
%[Wa, Wb, Wpi, F] = dovarem(data);
Fs = {};
N=100;
j = 0;
K = 20;
best = -1E14;
bestidx = 0;
llhist = zeros(1,1,1);
 for kappa=1:5:100
    for gamma=1:5:100
        for i=1:N            
            fprintf('(%d, %d)\n', kappa, gamma);
            %net = vbhmm({data}, 1:20, 10,1000,1e-6);            
            %kappa = 1;
            %gamma = 1; 
            [Wa, Wb, Wpi, F, logliks] = dovarem(data(1:1000),20,K,1000,1e-6, kappa, gamma,1.5,hidden);
            j = j+1;  
            llhist(j) = max(logliks);
            Fs{j} = {Wa, Wb, Wpi, F, kappa, gamma};   
        end
    end
 end
    %Fs{i} = {net.Wa, net.Wb, net.Wpi, net.F};
        %end

%% Show local optima

kappas = [1 10 20 30 40 50 60 70 80 90 100];
gammas = [1 10 20 30 40 50 60 70 80 90 100];
nK = length(kappas);
LLs = zeros(nK, nK);%*(-1E14);
for i=1:nK
    for j=1:nK
        fprintf('(%d, %d)\n', i, j);
        for k=1:3
            [Wa, Wb, Wpi, F, logliks] = dovarem(data,20,K,1000,1e-6, kappas(i), gammas(j),1.5,hidden);            
            LLs(i,j) = LLs(i,j) + max(logliks);             
        end        
        LLs(i,j) = LLs(i,j) / 3.0;
    end
end

%%
hist(LLs);
ylabel('Count')
xlabel('Loglikelihood');
xlim([-3000 -2800]);

 %%
 K=15;
kappa = 10;
gamma = 10;
Wa = [];
Wb = [];
Wpi = [];
bestll = [-1E14];
for i=1:10
    i
    [tWa, tWb, tWpi, ~, logliks] = dovarem(data,20,K,1000,1e-6, kappa, gamma,1.0,hidden);
    if max(logliks) > max(bestll)
        bestll = logliks;
        Wa = tWa;
        Wpi = tWpi;
        Wb = tWb;
    end
end
logliks = bestll;

%% plot ll bounds
for i=1:N
    plot(Fs{i}{4},'b-'); hold on;
end
xlim([0 400]);

maxval = -Inf;
maxidx = 1;
for i=1:N
    temp = Fs{i}{4};
    if max(temp) > maxval
        maxval = max(temp);
        maxidx = i;
    end
end
xlim([0 100]);

%% decode
%Wa, Wb, Wpi, F = Fs{maxidx};
%kappa = 30;
%gamma = 1;
%[Wa, Wb, Wpi, F] = dovarem(data,20,10,1000,1e-6, kappa, gamma,1.0);
Waprime = mkStochastic(Wa);
Wpiprime = mkStochastic(Wpi);
Wbprime = mkStochastic(Wb);
softev = Wbprime(:,data); 
%softev = Wb(:,data);


states = viterbi_path(Wpiprime, Waprime, Wbprime(:,data));
%states = viterbi_path(Wpi, Wa, softev);
plot(mapSequence2Truth(hidden, states)); hold on; plot(hidden,'r');
%imagesc(Waprime);

%%


%% get ride of unnecessary states
activeStates = zeros(20,1);
for i=1:20
    if sum(states == i) <= 5
        activeStates(i) = logical(0);
    else
        activeStates(i) = logical(1);
    end
end
activeStates = logical(activeStates);
softev = Wbprime(activeStates,data); 
states = viterbi_path(Wpiprime(activeStates), Waprime(activeStates,activeStates), softev);
plot(mapSequence2Truth(hidden, states)); hold on; plot(hidden,'r');
%[ri, gce, vi] = compare_segmentations(hidden, states);

%% Make figure
figure;
subplot(2,3,1);
imagesc(A);

subplot(2,3,2);
imagesc(badWa(2:5,2:5));

subplot(2,3,3);
imagesc(goodWa(1:4,1:4));

subplot(2,3,4);
plot(hidden,'r');

subplot(2,3,5);
plot(mapSequence2Truth(hidden, badSeq)); hold on; plot(hidden,'r');

subplot(2,3,6);
plot(mapSequence2Truth(hidden, goodSeq)); %hold on; plot(hidden,'r');

%% Plot hyperparam matrix

N=5;
n = 1;
kappas = [1 5 10 25 50 100];
gammas = [1 5 10 25 50 100];
llhist = zeros(1,1,1);
for i=1:5
    for j=1:5
        n
        kappa = kappas(i);
        gamma = gammas(j);
        fprintf('(%d, %d)', kappa, gamma);
         %net = vbhmm({data}, 1:20, 10,1000,1e-6);
         Fs = {};
         maxval = -Inf;
         currWa = zeros(10,10);
         for k=1:100
            [Wa, Wb, Wpi, F, logliks] = dovarem(data,20,10,1000,1e-6, kappa, gamma,1.0);
            llhist(i,j,k) = max(logliks);
         end
        %subplot(5,5,n);
        %imagesc(mkStochastic(currWa));
        n = n + 1;
    end        
end

%%
k = 0;
for i=1:size(llhist,1)
    for j=1:size(llhist,2)
        k = k + 1;
        subplot(size(llhist,1), size(llhist,2), k);
        hist(reshape(llhist(i,j,:),100,1));
        
    end
end

%% Alice in Wonderland example
trainText = readText('aliceTrain.txt');
aliceData = zeros(size(trainText));
chars = ['a':'z','_'];
nums = 1:27;
% convert to 1...27

for j=1:27
    aliceData(1,trainText == chars(j)) = j;
end
%%
Was = cell(20,1);
Wbs = cell(20,1);
Wpis = cell(20,1);
Fs = cell(20,1);
maxval = -Inf;

for i=1:20
    i 
    [Was{i}, Wbs{i}, Wpis{i}, Fs{i}] = dovarem(aliceData,27,60,1000,1e-6,0.1,10,10);
end
    

%[aliceModel, hmmllhist] = hmmFit(aliceData, 30, 'discrete');

%% test
testText = readText('aliceTest.txt');
aliceTest = zeros(size(testText));
chars = ['a':'z','_'];
nums = 1:27;
% convert to 1...27

for j=1:27
    aliceTest(1,trainText == chars(j)) = j;
end
%%
% best was -2.0397e+04
ll = zeros(20,1);
for i=1:20
    ll(i) = dhmm_logprob(aliceTest, mkStochastic(Wpis{i}),...
                      mkStochastic(Was{i}), mkStochastic(Wbs{i}));
    %[~, ~, ~, ~, Fv] = forwback(Was{i}, Wbs{i}, Wpis{i}, {aliceTest})
end
%%
[~, idx] = max(ll);
Wbprime = mkStochastic(Wbs{idx});
states = viterbi_path(mkStochastic(Wpis{idx}), mkStochastic(Was{idx}), Wbprime(:,aliceTest));
textsamp = numToText(dhmm_sample(mkStochastic(Wpis{idx}), mkStochastic(Was{idx}), Wbprime, 1, 500));
blah = obsProb{8};
textsampHmm = numToText(dhmm_sample(priorProb{8}, ...
    transProb{8}, obsProb{8}, 1, 500));
hmmStates = viterbi_path(priorProb{8}, transProb{8}, blah(:, aliceTest));
activeStates = zeros(60,1);
for i=1:60
    sum(states==i)
    if sum(states == i) <= 0.01*length(aliceData)        
        activeStates(i) = logical(0);
    else
        activeStates(i) = logical(1);
    end
end
activeStates = logical(activeStates);
Waprime = mkStochastic(Was{idx});
Wpprime = mkStochastic(Wpis{idx});
%imagesc(Waprime(activeStates,activeStates));
Wa2 = Waprime(activeStates,activeStates);
ll = dhmm_logprob(aliceTest, Wpprime(activeStates), Waprime(activeStates,activeStates), Wbprime(activeStates, aliceData));
%% Variational HDP-HMM
%minConf_SPG(funObj,x,funProj,options):

funObj = @(x) truncgemlikelihood(x,1,W);
funProj = @(w) projectSimplex(w);
betaWeights = randtruncgem(15,3);
W = rand(15,15)*20;
[x,f] = minConf_SPG(funObj, betaWeights, funProj);

%% TEST RAND INIT OR EM INIT
global X;
global Y;
global Z;
global betaWeights;
%[X, Y, Z, ~] = randseginit(data, 10, 20);
%betaWeights = randtruncgem(10,1);
[Wa, Wb, Wpi, F] = dovarem(data,20,10,1000,1e-6,30,1,1);
dhmm_logprob(test, mkStochastic(Wpi), mkStochastic(Wa), mkStochastic(Wb))

%% Run blocked Gibbs HDP-HMM
T = 2000;
d = 2;
clear settings;

settings.Niter = 655;
settings.resample_kappa = 1;  % Whether or not to use sticky model
settings.seqSampleEvery = 100; % How often to run sequential z sampling
settings.saveEvery = 100;  % How often to save Gibbs sample stats
settings.storeEvery = 1;
settings.storeStateSeqEvery = 100;
settings.ploton = 0;  % Whether or not to plot the mode sequence while running sampler
settings.plotEvery = 20;
settings.plotpause = 0;  % Length of time to pause on the plot
settings.saveDir = 'hdphmm_samps';  % Directory to which to save files
settings.Kz = 20;
settings.Ks = 1;
model.HMMmodel.params.a_sigma = 1;
model.HMMmodel.params.b_sigma = 0.01;

model.HMMmodel.params.a_alpha=10.0;  % affects \pi_z
model.HMMmodel.params.b_alpha=0.01;
model.HMMmodel.params.a_gamma=10;  % global expected # of HMM states (affects \beta)
model.HMMmodel.params.b_gamma=0.01;
model.HMMmodel.params.c=10;  % self trans
model.HMMmodel.params.d=1;
model.HMMmodel.type = 'HDP';

model.obsModel.type = 'Multinomial';
model.obsModel.params.nu = 1000;
model.obsModel.params.nu_delta = (model.obsModel.params.nu-d-1);
model.obsModel.params.alpha = ones(1,20);
model.obsModel.mixtureType = 'finite';

settings.trial = 1;
data_struct.obs = data;
data_struct.true_labels = hidden;

loglik = HDPHMMDPinference(data_struct, model, settings);


%% Make plot of all the stuff
hmmllhist = LLs{1};
plot(loglik);
hold on;
plot(logliks,'r');
hold on;
plot(length(logliks):655,logliks(end),'r');
hold on;
plot(hmmllhist,'g');
hold on; plot(length(hmmllhist):655,hmmllhist(end),'g');
xlabel('Iteration');
ylabel('Loglikelihood');
%ylim([-2.96E4 -2.92E4]);
%xlim([1 824]);