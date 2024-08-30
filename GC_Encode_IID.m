function [x,n,N,K,q,U,X] = GC_Encode_IID(u,l,c1,c2,t)
    %Add markers
    k=numel(u);
    K=ceil(k/l);
    N=K+c1+c2;
    q=2^l;
    u=[u(1:(K-1)*l),zeros(1,K*l-numel(u)),u((K-1)*l+1:end)];
    A=reshape(u,l,K);
    U=bi2de(fliplr(A'));
    flag = mod(c1+c2,2)~=0;
    rsEncoder = comm.RSEncoder(N+flag,K,'BitInput',false);
    primPolyDegree = l;
    rsEncoder.PrimitivePolynomialSource = 'Property';
    rsEncoder.PrimitivePolynomial = de2bi(primpoly(primPolyDegree,'nodisplay'),'left-msb');
    X=rsEncoder(U);
    if(flag)
        X(end)=[];
    end
    x=reshape(transpose(fliplr(de2bi(X,l))),1,numel(X)*l);
    if(mod(k,l)~=0)
        x((K-1)*l+1:K*l-mod(k,l))=[];
    end
    check_par=x(end-c2*l+1:end);
    x(end-c2*l+1:end)=[];
    x=[x,repelem(check_par,t)];
    n=numel(x);
end