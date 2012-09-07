function [ LL, prior, transmat, emit1, emit0 ] = chromhmm( testdata, K, L )
[ prior0, trans0, wb, segs ] = chromrandseginit(testdata,K,L,0.0);
emit1 = reshape(wb(:,:,1),K,L);
emit0 = reshape(wb(:,:,2),K,L);
for k=1:K
    for l=1:L
        total = emit1(k,l) + emit0(k,l);
        emit1(k,l) = emit1(k,l)/total;
        emit0(k,l) = emit0(k,l)/total;
    end
end
[LL, prior, transmat, emit1, emit0] = bhmm_em(testdata, prior0, trans0, emit1, emit0);


end

