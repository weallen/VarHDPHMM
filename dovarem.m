function [Wa, Wb, Wpi, F, logliks] = dovarem(data,L,K,its,tol, kappa, gamma, eta, hidden)
%K = 10;
T = length(data);

%L = 20;
%data = {data};
%gamma = 40.0;
alpha = 1.0;
alphaprime = 1;
%kappa = 50;

global betaWeights;
betaWeights = randtruncgem(K,eta);
ua = betaWeights'*(gamma/K);
ub = ones(1,L)*(alpha/L);
upi = ones(1,K)*(alphaprime/K);

softev = zeros(K, T);

wa = []; wb = [];
for k=1:K, % loop over hidden states
  wa(k,:) = dirrnd(ua)*T;
  wb(k,:) = dirrnd(ub)*T;
end
wpi = dirrnd(upi);


% cheat
%wpi = [0.25 0.25 0.25 0.25];
%wa = [.985 .005 .005 .005;
%     .01 .98 0.005 0.005;
%     .01 0.005 .98 0.005; 
%     .01 0.005 0.005 .98];

%% OPT 0: INITIALIZE TOTALLY RANDOMLY
%prior0 = normalise(rand(K,1));
%trans0 = mk_stochastic(rand(K,K));
%obs0   = mk_stochastic(rand(K,L));

%% OPT 1: INITIALIZE FROM RANDOM SEGMENTATION
[wpi, wa, wb, ~] = randseginit(data, K, L);
%global X;
%global Y;
%global Z;
%wpi = X;
%wa = Y;
%wb = Z;

%% OPT 2: INITIALIZE FROM EM
%disp 'Setting params by EM';
%[~, wpi, wa, wb] = dhmm_em(data, ...
%    prior0, trans0, obs0, 'max_iter', 500, 'thresh', 1e-6, 'verbose', 1);
%[~, wpi, wa, wb] = dhmm_em(data, ...
%    mkStochastic(wpi), mkStochastic(wa), mkStochastic(wb), 'max_iter', 500, 'thresh', 1e-6, 'verbose', 1);


%% OPT 3: Cheat
%[wpi, wa, wb] = initGroundTruth(data,hidden,K,L);

%%
wpi = wpi';
disp 'Done setting params';
%model = hmmFit(data, K, 'discrete');
%wpi = model.pi;
%wa = model.A;
%wb = model.emission.T;

%% Init vals
[ Wa, Wb, Wpi ] = updateparams( betaWeights, wa, wb, wpi, upi, ub, gamma, kappa, K, L);

%% Run algo
p = ones(K,1)./sum(1:K);
loglik = zeros(1,1);
logliks = [];
F = [];
Fa = [];
Fb = [];
Fpi = [];
lnZ = [];
Fold = -Inf; 
for i=1:its 

    
    %% E-step    
   
    astar = exp(digamma(Wa) - repmat(digamma(sum(Wa,2)), [1 K]));
    bstar = exp(digamma(Wb) - repmat(digamma(sum(Wb,2)), [1 L]));
    pistar = exp(digamma(Wpi) - digamma(sum(Wpi,2)));

    [wa, wb, wpi, lnZ(i), lnZv] = forwback(astar,bstar,pistar,{data});            
    
    Fa(i)=0; Fb(i)=0; Fpi(i)=0;
    for kk = 1:K,
        Fa(i) = Fa(i) - kldirichlet(Wa(kk,:),ua);
        Fb(i) = Fb(i) - kldirichlet(Wb(kk,:),ub);
    end
    Fpi(i) = - kldirichlet(Wpi,upi);
  
    F(i) = Fa(i)+Fb(i)+Fpi(i)+lnZ(i);
    Fold = F(i);
    %fprintf('It:%3i \tFa:%3.3f \tFb:%3.3f \tFpi:%3.3f \tFy:%3.3f \tF:%3.3f \tdF:%3.3f\n',i,Fa(i),Fb(i),Fpi(i),lnZ(i),F(i),F(i)-Fold);     
    if (i > 2)
        if (F(i) - F(i-1)) < tol
            return;
        end
    end
    
    logliks(i) = dhmm_logprob(data, mkStochastic(Wpi),mkStochastic(Wa), mkStochastic(Wb));
    
    %% M-step           
    [ Wa, Wb, Wpi ] = updateparams( betaWeights, wa, wb, wpi, upi, ub, gamma, kappa, K, L);    
    
    % beta step
    %funObj = @(x) truncgemlikelihood(x,1,log(astar));
    %funProj = @(w) projectSimplex(w);
	%betaWeights = minConf_SPG(funObj, betaWeights, funProj);

    %% Attempt merge move
    %attemptmerge(data, Wa, Wb, Wpi, F(i));
end


%% Final param settings
[ Wa, Wb, Wpi ] = updateparams( betaWeights, wa, wb, upi, wpi, ub, gamma, kappa, K, L);
end

function entropy = computeEntropy(Wz,Wx)
end