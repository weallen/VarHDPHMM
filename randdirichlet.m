function x = randdirichlet( a )

x = randgamma(a);
Z = sum(x,1);
x = x./Z(ones(size(a,1),1),:);

end

