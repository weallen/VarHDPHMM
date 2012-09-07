function path = varviterbi_path( prior, transmat, obslik )

K = length(prior(:));
T = size(obslik,2);


T
path = zeros(1,T);
psi = zeros(T,K);
delta = zeros(T,K);
scale = zeros(1,T);
    
  % Forward pass (with scaling)
  delta(1,:) = prior'.*obslik(:,1);
  scale(1) = sum(delta(1,:));
  delta(1,:) = delta(1,:)/scale(1);
  psi(1,:) = 0;
  for t=2:T
    for j=1:K
        [delta(t,j), psi(t,j)] = max(delta(t-1,:).*transmat(j,:));
        delta(t,j) = delta(t,j) * obslik(j,t);
    end
    scale(t) = sum(delta(t,:));
    delta(t,:) = delta(t,:)/scale(t);
  end;
  [p, path(T)] = max(delta(T,:));
  % Backward pass (with scaling)  
  for t=T-1:-1:1
    path(t) = psi(t+1,path(t+1));
  end;
end

