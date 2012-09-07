function [ Wa, Wb, Wpi ] = updateparams( betaWeights, wa, wb, wpi, upi, ub, gamma, kappa, K, L)
% update params in M-step of variational algorithm

ua = betaWeights' * (gamma/K);    

Wa = wa + repmat(ua, [K 1]);
for h=1:K
   for k=1:K
       if h==k
            Wa(h,k) = Wa(h,k) + kappa;
       end
    end
end

Wb = wb + repmat(ub, [K 1]);
Wpi = wpi + upi;    

end

