function  [wa, Gamma, wpi, lnZ, lnZv] = forwbackGauss(A,Pi,Obs)

%Performs forward-backward message passing for HMMs
%
%[wa, wb, wpi, lnZ, lnZv] = forwback(A,B,Pi,data);
%
% A (K by K) - state transition probabilities
% B (P by K) - observation emission probabilities
% Pi (1 by K) - initial state prior
% Note these probabilities can be sub-normalised.
%
%The E step returns:
%
% wa (K by K) - state transition counts (from all data)
% wb (P by K) - observation emission counts (from all data)
% wpi (1 by K) - initial state prior counts (from all data)
% lnZ (1 by 1) - log likelihood of all data
% lnZv (N by 1) - log likelihood of each data string
%
%M J Beal 13/04/02

[N K] = size(Obs);

Xi = zeros(K,K);
Gammak = zeros(K,1);
GammaInit = zeros(1,K);
lnZv = zeros(N,1);


Gamma = zeros(N,K);
alpha = zeros(N,K);
beta = zeros(N,K);
scale = zeros(1,N);
  
% Pick out the likelihood of each symbol in the sequence  
  
% Forward pass (with scaling)
alpha(1,:) = Pi.*Obs(1,:);
scale(1) = sum(alpha(1,:));
alpha(1,:) = alpha(1,:)/scale(1);
for t=2:N
  alpha(t,:) = (alpha(t-1,:)*A).*Obs(t,:);
  scale(t) = sum(alpha(t,:));
  alpha(t,:) = alpha(t,:)/scale(t);
end;
  
sum(isnan(alpha(:)))
  % Backward pass (with scaling)
beta(N,:) = ones(1,K)/scale(N);
for t=N-1:-1:1
  beta(t,:) = (beta(t+1,:).*Obs(t+1,:))*A'/scale(t); 
end;
  
  % Another pass gives us the joint probabilities
for t=1:N-1
  Xi=Xi+A.*(alpha(t,:)'*(beta(t+1,:).*Obs(t+1,:)));
end;
  
  % Compute Gamma
Gamma = alpha.*beta;
Gamma = Gamma./repmat(sum(Gamma,2),1,K);
  
  % Compute the sums of Gamma conditioned on k
%for t = 1:N
%  Gammak(:,data{n}(t)) = Gammak(:,data(t)) + Gamma(t,:)';
%end;
  
GammaInit = GammaInit + Gamma(1,:); 

lnZv = sum(log(scale));
  


lnZ = sum(lnZv,1);
wa = Xi;
wpi = GammaInit;

end
