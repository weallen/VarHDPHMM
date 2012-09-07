function [ wpi, wa, wb, segs ] = chromrandseginit(data,K,L,kappa)
% RANDSEGINIT Initialize parameters for HMM from random 
% segmentation of data.
segs = zeros(size(data,1),1);
N = length(segs)/2;

segval = 0;

T = size(data,1);

% make random transition matrix
wpi = normalise(rand(1,K));
wa = mk_stochastic(rand(K,K) + kappa * eye(K));

wb = zeros(K,L,2);

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
chunklen = floor(length(data)/K);
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
for t=1:T
    curr = logical(data(t,:));    
    k = segs(t);
    wb(k,curr,1) = wb(k,curr,1) + 1.0;
    wb(k,~curr,2) = wb(k,~curr,2) + 1.0;    
end

wb = mk_stochastic(wb);
end

