function softev = chrommksoftev(Wpi, Wa, Wb, data)

K = length(Wpi);
[T L] = size(data);

softev = zeros(K,T);
for t=1:T
    for k=1:K
        curr = logical(data(t,:));
        softev(k,t) = prod(Wb(k,curr,1))*prod(Wb(k,~curr,2));
    end
end

end

