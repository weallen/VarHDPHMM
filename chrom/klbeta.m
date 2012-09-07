%res = kldirichlet(vecP,vecQ)
%
%Calculates KL(P||Q) where P and Q are Beta distributions with
%parameters 'vecP' and 'vecQ', which are row vectors, not
%necessarily normalised.
%
% KL(P||Q) = \int d\pi P(\pi) ln { P(\pi) / Q(\pi) }.
%


function [res] = klbeta(Ps,Qs)
alphaP = Ps(1)+eps;
betaP = Ps(2)+eps;
alphaQ = Qs(1)+eps;
betaQ = Qs(2)+eps;
%alphaP = sum(vecP,2);
%alphaQ = sum(vecQ,2);

%res = gammaln(alphaP)-gammaln(alphaQ) ...
%    - sum(gammaln(vecP)-gammaln(vecQ),2) ...
%    + sum( (vecP-vecQ).*(digamma(vecP)-digamma(alphaP)) ,2);
res = betaln(alphaQ, betaQ) - betaln(alphaP,betaP) -...
   (alphaQ - alphaP)*digamma(alphaP) -...
    (betaQ - betaP)*digamma(betaP) +...
    (alphaQ - alphaP + betaQ - betaP)*digamma(alphaP+betaP);
