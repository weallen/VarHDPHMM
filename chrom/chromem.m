function [Wa, Wb, Wpi, F, logliks] = chromem(data,L,K,its,tol, kappa, gamma, eta)
T = length(data);


A = 0.1;
B = 0.9;
alphaprime = 1;

betaWeights = randtruncgem(K,eta);
ua = betaWeights'*(gamma/K);
ub = [A B];
upi = ones(1,K)*(alphaprime/K);

softev = zeros(K, T);

logliks = [];
% wa : expected transition counts
wa = []; 

% wb : expected emission counts
wb = zeros(K,L,2);

for k=1:K, % loop over hidden states
  wa(k,:) = dirrnd(ua)*T;  
  wb(k,:,1) = betarnd(ub(1), ub(2), L, 1)*T;
  wb(k,:,2) = T - wb(k,:,1);
end

% wpi : expected init counts
wpi = dirrnd(upi);


%% OPT 0: INITIALIZE TOTALLY RANDOMLY
prior0 = normalise(rand(K,1));
trans0 = mk_stochastic(rand(K,K));
obs0   = rand(K,L);

%% OPT 1: INITIALIZE FROM RANDOM SEGMENTATION
[wpi, wa, wb, ~] = chromrandseginit(data, K, L, kappa);


%% OPT 2: INITIALIZE FROM EM
%disp 'Setting params by EM';
%[~, wpi, wa, wb] = dhmm_em(data, ...
%    prior0, trans0, obs0, 'max_iter', 500, 'thresh', 1e-6, 'verbose', 1);
%[~, wpi, wa, wb] = dhmm_em(data, ...
%    mkStochastic(wpi), mkStochastic(wa), mkStochastic(wb), 'max_iter', 500, 'thresh', 1e-6, 'verbose', 1);

%%
%wpi = wpi';
%disp 'Done setting params';

%% Init vals
[ Wa, Wb, Wpi ] = chromupdateparams( betaWeights, wa, wb, wpi, upi, A, B, gamma, kappa, K, L);

%% Run algo
p = ones(K,1)./sum(1:K);
loglik = zeros(1,1);

F = [];
Fa = [];
Fb = [];
Fpi = [];
lnZ = [];
Fold = -Inf; 
for i=1:its 

    astar = exp(digamma(Wa) - repmat(digamma(sum(Wa,2)), [1 K]));
    pistar = exp(  digamma(Wpi) - digamma(sum(Wpi,2))  );
    bstar = zeros(K, L, 2);

    for kk=1:K
        total = Wb(kk,:,1) + Wb(kk,:,2);
        bstar(kk,:,1) = exp(digamma(Wb(kk,:,1)))./exp(digamma(total));
        bstar(kk,:,2) = exp(digamma(Wb(kk,:,2)))./exp(digamma(total));
    end
    
    %% E-step    
    
   
    %[wa, wb, wpi, lnZ(i)] = chromfwdback(astar,bstar,pistar,{data});            
    [wa, wb, wpi, lnZ(i)] = chromfwdbackCPP(astar,bstar,pistar,data);            

    Fa(i)=0; Fb(i)=0; Fpi(i)=0;
    for kk = 1:K,
        Fa(i) = Fa(i) - kldirichlet(Wa(kk,:),ua);
        for j = 1:L            
            Fb(i) = Fb(i) - klbeta(Wb(kk,j,:),ub);
        end
    end
    Fpi(i) = - kldirichlet(Wpi,upi);
  
    F(i) = Fa(i)+Fb(i)+Fpi(i)+lnZ(i);
    Fold = F(i);
    %fprintf('It:%3i \tFa:%3.3f \tFb:%3.3f \tFpi:%3.3f \tFy:%3.3f \tF:%3.3f \tdF:%3.3f\n',i,Fa(i),Fb(i),Fpi(i),lnZ(i),F(i),F(i)-Fold);     
    if (isnan(Fold))
        return;
    end
    if (i > 2)
        if (F(i) - F(i-1)) < tol
            return;
        end
    end
    logliks(i) = lnZ(i);
       
    %% M-step           
    [ Wa, Wb, Wpi ] = chromupdateparams( betaWeights, wa, wb, wpi, upi, A, B, gamma, kappa, K, L);
    
    % beta step
    %funObj = @(x) truncgemlikelihood(x,1,log(astar));
    %funProj = @(w) projectSimplex(w);
	%betaWeights = minConf_SPG(funObj, betaWeights, funProj);

    %% Attempt merge move
%    attemptmerge(data, Wa, Wb, Wpi, F(i));
end


%% Final param settings
[ Wa, Wb, Wpi ] = chromupdateparams( betaWeights, wa, wb, wpi, upi, A, B, gamma, kappa, K, L);
end

