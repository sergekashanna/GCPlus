function [x,n,N,K,q] = GC_Encode_loc(u,l,c,w)
    k=numel(u);
    K=ceil(k/l);
    N=K+c;
    q=2^l;
    u=[u(1:(K-1)*l),zeros(1,K*l-numel(u)),u((K-1)*l+1:end)];
    A=reshape(u,l,K);
    U=bi2de(fliplr(A'));
    rsEncoder = comm.RSEncoder(N,K,'BitInput',false);
    primPolyDegree = l;
    rsEncoder.PrimitivePolynomialSource = 'Property';
    rsEncoder.PrimitivePolynomial = de2bi(primpoly(primPolyDegree,'nodisplay'),'left-msb');
    X=rsEncoder(U);
    x=reshape(transpose(fliplr(de2bi(X,l))),1,numel(X)*l);
    if(mod(k,l)~=0)
        x((K-1)*l+1:K*l-mod(k,l))=[];
    end
    x=[x(1:k),ones(1,w+1),zeros(1,w+1),ones(1,w+1),x(k+1:end)];
    n=numel(x);
end