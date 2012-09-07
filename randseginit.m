function [ wpi, wa, wb, segs ] = randseginit(data,K,L)
% RANDSEGINIT Initialize parameters for HMM from random 
% segmentation of data.
segs = zeros(size(data,2),1);
N = length(segs)/2;

segval = 0;

kappa = 1.0;

% make random transition matrix
wpi = normalise(rand(K,1));
wa = mk_stochastic(rand(K,K) + kappa * eye(K));

wb = zeros(K,L);

%% Random seg from random matrix
% generate random segmentation
segs(1) = find(mnrnd(1,wpi) == 1);
for i=2:length(segs)
    currtrans = wa(segs(i-1),:);
    segs(i) = find(mnrnd(1, currtrans) == 1);
end

%% K-block random init
%{
n = 0;
j = 1;
%states = randperm(K);
states = randperm(K);
chunklen = floor(length(data)/K)
currstate = states(1);
for i=1:K
    n = n + 1;
    currstate = states(n);        

    for k=1:chunklen
       segs(j) = currstate;   
       j = j + 1;
    end  
end

% count transitions
wa = ones(K,K); %+ kappa*eye(K);
for i=1:(length(segs)-1)
    wa(segs(i), segs(i+1)) = wa(segs(i), segs(i+1)) + 1;
end
wa = mk_stochastic(wa);
%}
%% Compute emission params
for i=1:K
    currdata = data(segs == i);
    for j=1:L
        wb(i,j) = sum(currdata == j) + 1.0;
    end
end

wb = mk_stochastic(wb);
end

