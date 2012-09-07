function [Wa, mus, lambdas, Wpi, F,softev] = dovaremGauss(data,K,its,tol)
%K = 10;
T = length(data);
N = length(data);

%L = 20;
%data = {data};
gamma = 10.0;
alpha = 1.0;
alphaprime = 1;
kappa = 1;

betaWeights = randtruncgem(K,1.0);

% hyperparams for 
ua = betaWeights'*(gamma/K);
upi = ones(1,K)*(alphaprime/K);

% hyperparams for Normal-Gamma distribution
a0 = 0;
b0 = 0;
k0 = 0;
mu0 = 0;
kappa0 = 0;
[~,mus] = kmeans(data,K);
lambdas = randnorm(K,1,1);

softev = zeros(K, T);

wa = []; wb = [];
for k=1:K, % loop over hidden states
  wa(k,:) = dirrnd(ua)*T;  
end
wpi = dirrnd(upi);

p = ones(K,1)./sum(1:K);
loglik = zeros(1,1);
softev = zeros(K, T);

F = [];
Fa = [];
Fb = [];
Fpi = [];
lnZ = [];
Fold = -Inf; 

for i=1:its

    % update soft evidence       
    for k=1:K
        softevff(k, :) = normalLogPdf(data,mus(k), 1.0);
    end
    softev = normalizeLogspace(softev');
    softev = exp(softev');
    
    % M-step           
    ua = betaWeights' * (gamma/K);    

    Wa = wa + repmat(ua, [K 1]);
    for h=1:K
       for k=1:K
            if h==k
                Wa(h,k) = Wa(h,k) + kappa;
            end
        end
    end

    Wpi = wpi + upi;
    
    astar = exp(digamma(Wa) - repmat(digamma(sum(Wa,2)), [1 K]));    
    pistar = exp(digamma(Wpi) - digamma(sum(Wpi,2)));
    
    % beta step
    %funObj = @(x) truncgemlikelihood(x,1,log(astar));
    %funProj = @(w) projectSimplex(w);
	%betaWeights = minConf_SPG(funObj, betaWeights, funProj);

    % E-step    

    %[alpha, beta, gamma, current_ll] = hmmFwdBack(p, Wz, obslik);        
    %xi_summed = hmmComputeTwoSlice(alpha, beta, Wz, obslik);
   [wa, Gamma, wpi, lnZ(i), lnZv] = forwbackGauss(astar,pistar,softev');            
    
    % update q(mu)
    for k=1:K
        mus(k) = sum(Gamma(:,k)' .* data) / sum(Gamma(:,k));
        %elambda = aN / bN;
        %muN = (kappa0 * mu0 + N * m) / (kappa0 + N);
        %kappaN = (kappa0 + N) * elambda;     
    end
  
    %for k=1:K
    %    lambdas(k) = sum(Gamma(:,k)' .* ((data - mus(k)).^2)) / sum(Gamma(:,k));
    %end
    
    
    % update q(lambda)
    %emu = muN;
    %emuSquare = 1/kappaN + muN^2;
    %aN = a0 + (N+1)/2;
    %bN = b0 + 1/2*((sSq + kappa0 * mu0^2) - 2*emu*(s + kappa0 * mu0) + emuSquare*(kappa0+N));
    
    % do lower bound computataion
    Fa(i)=0; Fb(i)=0; Fpi(i)=0;
    for kk = 1:K,
        Fa(i) = Fa(i) - kldirichlet(Wa(kk,:),ua);
        Fb(i) = 0;
        %Fb(i) = Fb(i) - kldirichlet(Wb(kk,:),ub);
    %    Fb(i) = 1/2 * log(1/kappaN) + log(gamma(aN)) - aN*log(bN);
    end
    
    Fpi(i) = - kldirichlet(Wpi,upi);
  
    F(i) = Fa(i)+Fb(i)+Fpi(i)+lnZ(i);
    Fold = F(i);
%    fprintf('It:%3i \tFa:%3.3f \tFb:%3.3f \tFpi:%3.3f \tFy:%3.3f \tF:%3.3f \tdF:%3.3f\n',i,Fa(i),Fb(i),Fpi(i),lnZ(i),F(i),F(i)-Fold);     
    if (i > 2)
        if (F(i) - F(i-1)) < tol
            return;
        end
    end
end

end

