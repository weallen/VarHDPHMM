function [ newemit ] = mkemitstochastic( emit )
[K L ~] = size(emit);
newemit = zeros(size(emit));

for k=1:K
    for j=1:L
        total = emit(k,j,1) + emit(k,j,2);
        newemit(k,j,1) = emit(k,j,1) / total;
        newemit(k,j,2) = emit(k,j,2) / total;
    end
end
end

