function [ wpi, wa, wb  ] = initGroundTruth(data, segs,K,L)
%INITGROUNDTRUTH Summary of this function goes here
%   Detailed explanation goes here

wpi = zeros(K,1);
wpi(segs(1)) = 1;

wb = zeros(K,L);

wa = zeros(K,K); 

for i=1:(length(segs)-1)    
    wa(segs(i), segs(i+1)) = wa(segs(i), segs(i+1)) + 1;
end

wa = mk_stochastic(wa);

for i=1:K
    currdata = data(segs == i);
    for j=1:L
        wb(i,j) = sum(currdata == j) + 1.0;
    end
end

wb = mk_stochastic(wb);

end

