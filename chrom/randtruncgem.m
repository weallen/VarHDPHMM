function betaWeights = randtruncgem( K, alpha )
betaWeights = zeros(K,1);
betaWeights(1) = randbeta(1,alpha);
for k=2:(K-1)    
    zs = randbeta(1,alpha);
    betaWeights(k) = zs*(1-sum(betaWeights(1:(k-1))));
end
betaWeights(K) = 1-sum(betaWeights(1:(K-1)));
end

