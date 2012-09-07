function [ Wa, Wb, Wpi, logprob, kappa, gamma ] = chromsethyperparams( data, K, L, maxkappa, maxgamma,eta)
bestlogprob = -1E14;
i = 0;
Wa = [];
Wb = [];
Wpi = [];
kappa = 0;
gamma = 0;
for kk=1:5:maxkappa
    for gg=1:5:maxgamma
        i = i + 1
        [a, b, p, F] = chromem(data, L, K, 300, 1e-6, kk, gg, eta);
        temp = chromlogprobCPP(mk_stochastic(a), mkemitstochastic(b),mk_stochastic(p), data);        
        if temp > bestlogprob
            Wa = a;
            Wb = b;
            Wpi = p;
            bestlogprob = temp;
            kappa = kk;
            gamma = gg;
        end
    end
end
logprob = bestlogprob;
end

