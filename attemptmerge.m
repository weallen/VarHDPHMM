function [ output_args ] = attemptmerge( data, Wa, Wb, Wpi, varbound )
%ATTEMPTMERGE Summary of this function goes here
% randomly select two states to attempt to merge
% if the merge increases the variational bound, then accept 
% the merge; otherwise, reject it.

% instantiate the state labels
states = viterbi_path(mkStochastic(Wpi), mkStochastic(Wa), mkStochastic(Wb));

% randomly select two labels to merge

% re-estimate transition and emission distributions

% check if imporves bound
end

