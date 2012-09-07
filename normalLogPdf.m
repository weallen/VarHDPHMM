function p = normalLogPdf(x, mu, s)
X = x - mu;
p = -(X.^2)./(2*s) - 0.5 * log(2*3.1415*s);
end

