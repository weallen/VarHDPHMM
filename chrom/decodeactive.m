function [ states ] = decodeactive( Wa, Wpi, Wb, data )
K = size(Wa, 1);

Waprime = mk_stochastic(Wa);
Wpiprime = mk_stochastic(Wpi);
Wbprime = mkemitstochastic(Wb);
temp = chrommksoftev(Wpiprime, Waprime, Wbprime, data);
disp 'made soft evidence';
states = viterbi_path(Wpiprime, Waprime, temp);
disp 'made path';
activeStates = zeros(K,1);
for i=1:20
    if sum(states == i) <= 1
        activeStates(i) = 0;
    else
        activeStates(i) = 1;
    end
end
activeStates = logical(activeStates);
%temp = chrommksoftev(Wpiprime(activeStates), Waprime(activeStates,activeStates), Wbprime(activeStates,:,:), data);

%states = viterbi_path(Wpiprime(activeStates), Waprime(activeStates,activeStates), temp);

end

