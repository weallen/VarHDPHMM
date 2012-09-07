function [ Wa, Wb, Wpi ] = chromupdateparams(betaWeights, wa, wb, wpi, upi, A, B, gamma, kappa, K, L)
% update params in M-step of variational algorithm

% Wb : [K x L] 
% Wa : [K x K]
% Wpi : [1 x K]
ua = betaWeights' * (gamma/K);    

Wa = wa + repmat(ua, [K 1]);
for h=1:K
   for k=1:K
       if h==k
            Wa(h,k) = Wa(h,k) + kappa;
       end
    end
end

% p(B_ij|y) = Beta(y + alpha, 1 - y + beta)
%Wb = wb + repmat(ub, [K 1]);

Wpi = wpi + upi;    
Wb = zeros(K, L, 2);
for i=1:K   
    Wb(i,:,1) = A + wb(i,:,1);
    Wb(i,:,2) = B + wb(i,:,2);
end

end

