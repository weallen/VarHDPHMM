function cpd  = fitDiscreteEss( cpd, ess )
%FITDISCRETEESS Summary of this function goes here
%   Detailed explanation goes here
prior = cpd.prior;
if isempty(prior)
    alpha = 1;
else
    alpha = prior.alpha;
end

for i=1:cpd.nstates
    for j=1:cpd.nObsStates
        cpd.T(i,j) = exp(digamma(alpha + ess.counts(i,j)))/exp(digamma(sum(alpha + ess.counts(i,:))));
    end
end
end

