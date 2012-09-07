%% Setup
cd('~/Documents/CS/CSCI2950P/project/code/chrom');
addpath(genpath('../HMMall'));
addpath(genpath('../vbhmm'));
addpath(genpath('~/Documents/MATLAB/Lightspeed'));
addpath(genpath('../HDPHMM_HDPSLDS_toolbox/relabeler/'));
%addpath('/course/cs195f/asgn/pmtk3-28feb11')
addpath('~/Documents/MATLAB/pmtk3/');
initPmtk3;
cd('~/Documents/CS/CSCI2950P/project/code/chrom');

%%
chromdata = importdata('K562_chr11_binary.txt');

%testdata = chromdata.data(10000:100000,:);

%% Generate test data


pi = [0.25 0.25 0.25 0.25];
%pi = ones(9,1)+ rand(9,1);
%pi = pi ./ sum(pi);
%pi = ones(nstates,1) ./ nstates;
A = [.985 .005 .005 .005;
     .005 .985 0.005 0.005;
     .005 0.005 .985 0.005; 
     .005 0.005 0.005 .985];
%
%A = [0.60 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05;
%     0.20 0.70 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143;
%     0.20 0.0143 0.70 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143;
%     0.20 0.0143 0.0143 0.70 0.0143 0.0143 0.0143 0.0143 0.0143;
%     0.20 0.0143 0.0143 0.0143 0.70 0.0143 0.0143 0.0143 0.0143;
%     0.20 0.0143 0.0143 0.0143 0.0143 0.70 0.0143 0.0143 0.0143;
%     0.20 0.0143 0.0143 0.0143 0.0143 0.0143 0.70 0.0143 0.0143;
%     0.20 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143 0.70 0.0143;
%     0.20 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143 0.0143 0.70];
%A = A+normrnd(0.1,0.5,9)/10; %rand(9,9)/10.0;
A=mkStochastic(A);

K = 9;
L = 5;
T = 1000;
emit = rand(K,L);
% super easy emission distribution


hidden = zeros(T, 1);
currstate = find(multirnd(pi) == 1);

testdata = zeros(T,L);

for i=1:T
    hidden(i) = currstate;
    for j=1:L
        testdata(i,j) = binornd(1, emit(currstate, j));
    end
    currstate = find(multirnd(A(currstate,:)) == 1);        
end
%rasterplot(testdata);

temp = zeros(K,L,2);
temp(:,:,1) = emit;
temp(:,:,2) = 1-emit;

%% Try on test data
%blah = importdata('~/src/ChromVarHDPHMM/test.txt');
%testdata = blah.data;
K = 10;
L = 5;
%kappa = 31.0;
%gamma = 21.0;
%[~, ~, ~, logprob, kappa, gamma] = chromsethyperparams(testdata, K, L, 100, 100, 1.0);
kappa = 20;
gamma = 40;
[Wa, Wb, Wpi, F, logliks] = chromem(testdata, L, K, 300, 1e-6, kappa, gamma, 1.5);
states = decodeactive(Wa, Wpi, Wb, testdata);
chromlogprobCPP(mk_stochastic(Wa), mkemitstochastic(Wb),mk_stochastic(Wpi), testdata)

activeStates = zeros(K,1);
for i=1:K
    if sum(states == i) > 1
        activeStates(i) = 1;
    else
        activeStates(i) = 0;
    end
end
activeStates = logical(activeStates);
imagesc(mk_stochastic(Wa(activeStates, activeStates)));

%% Show local optima
LLs = [];
kappa = 20;
gamma = 40;
for i=1:500
    i
    [Wa, Wb, Wpi, F, logliks] = chromem(testdata, L, K, 300, 1e-6, kappa, gamma, 1.5);  
    LLs(i) = max(logliks);
end

%%
hist(LLs);
ylabel('Count')
xlabel('Loglikelihood');
xlim([-3600 -3100]);

%% decode hidden sequence
plot(mapSequence2Truth(hidden', states));
%plot(states,'b');
hold on;
plot(hidden,'r');

%% Try on real data

%%
K = 40;
L = size(chromdata.data, 2);

subrange = chromdata.data(1:10000,:);
[~, ~, ~, logprob, kappa, gamma] = chromsethyperparams(subrange, K, L, 100, 100, 2.0);

%%
realdata = chromdata.data(1:10000,:);
%realdata = chromdata.data(1:	, :);
% best gamma = 96, best kappa = 41 by empirical Bayes
kappa = 41;
gamma = 96;
K = 40;
L = 10;
[Wa, Wb, Wpi, F, loglik] = chromem(realdata, L,K,1000,1e-6, kappa, gamma,5.0);
chromlogprobCPP(mk_stochastic(Wa), mkemitstochastic(Wb), mk_stochastic(Wpi), realdata)
states = decodeactive(Wa, Wpi, Wb, realdata);
%%
subplot(2,1,1)
states = map2smallestIntegers(states,K);
plot(states,'.','MarkerSize',1.0);
ylim([min(states)-1 max(states)+1]);
hresp=gca;
set(hresp,'ytick',[],'xtick',[]);
subplot(2,1,2);
rasterplot(realdata);



%% REGULAR HMM SECTION
%% Test regular HMM

K = 4;
L = 5;

[ LL, prior, transmat, emit1, emit0 ] = chromhmm( testdata, K, L );
obslik = zeros(K, 1000);
for t=1:T
    currvals = logical(testdata(t,:));        
    for kk=1:K
        obslik(kk,t) = prod(emit1(kk,currvals)) .* prod(emit0(kk,~currvals));
    end
end

states = viterbi_path(prior, transmat, obslik);
states = map2smallestIntegers(states,K);
plot(path);
hold on;
plot(hidden,'r');

%% Try multiple Models
realdata = chromdata.data(1:10000,:);

lls = [];
i = 1;
L = 10;
Ks = [1 5 10 15 20 30 40 50 60 70 80];
nK = length(Ks);
models = cell(1,nK);
for i=1:length(Ks);
    kk = Ks(i)
    [ LL, prior, transmat, emit1, emit0 ] = chromhmm( realdata, kk, L );
    models{i} = {LL, prior, transmat, emit1, emit0};
    lls(i) = max(LL);
    i = i+1;
end

%% Evaluate BIC
%BIC = -2log(L) + klog(T)
%k = K*L + K*(K-1) + (K-1)

BIC = [];
T = length(realdata);
for i=1:nK
    K = Ks(i);
    k = K*L + K*(K-1) + (K-1);
    BIC(i) = 2*lls(i) - k*log(T);
end
[~, idx] = max(BIC);
plot(Ks, BIC);
xlabel('Number of states');
ylabel('BIC score');

%% Evaluate test
realtest = chromdata.data(:,:);
T = size(realtest,1);
testhmmlls2 = [];
for i=1:nK    
    i
    K = Ks(i);
    transmat = models{i}{3};        
    init = models{i}{2};
    emit = zeros(K,10,2);
    emit(:,:,1) = models{i}{4};
    emit(:,:,2) = models{i}{5};
    %obslik = ones(K, T);    
    %for t=1:T
    %    currvals = logical(realtest(t,:));        
    %    for kk=1:K
    %        obslik(kk,t) = prod(emit1(kk,currvals)) .* prod(emit0(kk,~currvals));
    %    end
    %end
    %[~,~,~, testhmmlls2(i)] = fwdback(init, transmat, obslik, 'fwd_only', 1);
    testhmmlls2(i) = chromlogprobCPP(transmat, emit, init', realtest);
end


%%
%kappa = 41;
%gamma = 96;
kappa = 100;
%gamma = 50;
gamma = 100;
K = 40;
L = 10;
Wa = [];
Wb = [];
Wpi = [];
bestll = -1E14;
for i=1:10
    i
    [tWa, tWb, tWpi, F, loglik] = chromem(realdata, L,K,1000,1e-6, kappa, gamma,10.0);
    ll = chromlogprobCPP(mk_stochastic(tWa), mkemitstochastic(tWb), mk_stochastic(tWpi), realtest);
    if ll > bestll
        Wa = tWa;
        Wb = tWb;
        Wpi = tWpi;
        bestll = ll;
    end
end

%%
plot(Ks, testhmmlls2);
hold on;
plot(Ks, repmat(bestll, [1 nK]),'r');
ylim([-9.5317e+05 -3.0e+05]);
xlabel('Number of HMM states');
ylabel('Loglikelihood');

%% segment data with best params
states = decodeactive(Wa, Wpi, Wb, realtest);
states = map2smallestIntegers(states,40);
%%
transmat = models{3}{3};        
init = models{3}{2};
emit = zeros(10,10,2);
emit(:,:,1) = models{3}{4};
emit(:,:,2) = models{3}{5};
hmmstates = decodeactive(transmat, init, emit, realtest);
hmmstates = map2smallestIntegers(hmmstates,40);

%%
subplot(2,1,1);
rasterplot(realdata);
ylabel('Tracks');
subplot(2,1,2);
plot(states,'.','MarkerSize',10.0);
xlabel('Window number');
ylabel('State number');

%% plot just RNH1
subplot(2,1,1)

plot(states(2462:2549),'.','MarkerSize',10.0);
ylim([min(states)-1 max(states)+1]);
xlabel('Window number');
ylabel('State number');

hresp=gca;
set(hresp,'ytick',[],'xtick',[]);
subplot(2,1,2);
rasterplot(realdata(2462:2549,:));
ylabel('Tracks');

%% Do whole data learning
kappa = 100;
gamma = 500;
[tWa, tWb, tWpi, F, loglik] = chromem(realtest, L,K,1000,1e-6, kappa, gamma,10.0);
ll = chromlogprobCPP(mk_stochastic(tWa), mkemitstochastic(tWb), mk_stochastic(tWpi), realtest);
%states = decodeactive(Wa, Wpi, Wb, realtest);
%states = map2smallestIntegers(states,40);
