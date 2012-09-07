function [ L, gradient ] = truncgemlikelihood( beta, alpha, W)
%BETALIKELIHOOD Used for optimizing degenerate GEM distribution
% Returns likelihood and derivatives
W = log(W);
K = length(beta);
T = zeros(K,1);
prior_gradient = zeros(K,1);
trans_gradient = zeros(K,1);

%% Compute likelihood
for k=1:K
    T(k) = 1-sum(beta(1:(k-1)));
end

Lprior = (alpha - 1)*log(T(K)) - sum(log(T(1:(K-1)))) + (K-1)*log(alpha);

Ltrans = 0;
for z=1:K
    Ltrans = Ltrans + log(alpha) - sum(log(gamma(alpha * beta)));
    for i=1:K
        Ltrans = Ltrans + (alpha * beta(i))*W(z,i);
    end
end

L = Lprior + Ltrans;

%% Compute gradient
% for prior
for k=1:K
    prior_gradient(k) = -(alpha - 1)/T(K) + sum(1./T((k+1):(K-1)));
end

for k=1:K
    trans_gradient(k) = 0;
    sum1 = 0;
    for z=1:K        
        sum1 = sum1 + (W(z,k) - digamma(alpha * beta(k)));                                
    end
    trans_gradient(k) = trans_gradient(k) + alpha*sum1;
    sum2 = 0;    
    for z=1:K
        sum2 = sum2 + (W(z,K) - digamma(alpha * beta(K)));
    end
    trans_gradient(k) =  trans_gradient(k) - alpha * sum2;
end

% for transitions
gradient = prior_gradient + trans_gradient;
%gradient = prior_gradient;
end

