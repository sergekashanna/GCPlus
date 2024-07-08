function [uhat, Uhat] = GC_Decode_IID(y,n,k,l,N,K,c,c1,c2,q,secondary,len_last,lim,P1,P2,t)
    
    %Initilaizations
    uhat=[];
    d=numel(y)-n;
    
    %RS encoder setup
    rsEncoder = comm.RSEncoder(N,K,'BitInput',false);
    primPolyDegree = l;
    rsEncoder.PrimitivePolynomialSource = 'Property';
    rsEncoder.PrimitivePolynomial = de2bi(primpoly(primPolyDegree,'nodisplay'),'left-msb');

    %RS decoder setup
    rsDecoder=comm.RSDecoder(N,K,'BitInput',false);
    rsDecoder.ErasuresInputPort=true;
    rsDecoder.NumCorrectedErrorsOutputPort=true;
    rsDecoder.PrimitivePolynomialSource = 'Property';
    rsDecoder.PrimitivePolynomial = de2bi(primpoly(l,'nodisplay'),'left-msb');

    %Decode check parities
    seg=y(end-c2*l*t+1:end);
    p=rep_decode_gamma(seg,t,c2*l);
    A=reshape(p,l,c2);
    Par=bi2de(fliplr(A'));
    
    %Fast check
    if(d==0)
        yE=y(1:k+c1*l);
        lengths=ones(1,K+c1)*l;
        lengths(K)=len_last;
        Y=divide_vector(yE,lengths);
        Y=[Y';Par];
        erasure_pattern=[zeros(1,N-c+c1),ones(1,c-c1)];
        [Uhat, errs]=step(rsDecoder,Y,erasure_pattern');
        Xhat=rsEncoder(Uhat);
        if(isequal(Xhat(K+c1+1:K+c1+c2),Par(1:c2)) && errs~=-1)
            uhat=reshape(transpose(fliplr(de2bi(Uhat,l))),1,K*l);
            if(len_last<l)
                uhat((K-1)*l+1:K*l-mod(k,l))=[];
            end
            return;
        end
    end
    
    D=abs(d);
    for z=0:D
        if(d<0)
            d=d+z;
            yE=y(1:k+c1*l+d);
        else
            d=d-z;
            yE=y(1:k+c1*l+d);
        end
        %Primary check
        for i=1:size(P1,1)
            lengths=ones(1,K+c1)*l;
            lengths(K)=len_last;
            erasures=P1(i,:);
            for j=1:abs(d)
                m=mod(j-1,c1)+1;
                lengths(erasures(m))=lengths(erasures(m))+sign(d);
            end
            if(~isempty(lengths(lengths<0)))
                continue;
            end
            Y=divide_vector(yE,lengths);
            Y=[Y';Par];
            erasure_pattern=[zeros(1,N-c+c1),ones(1,c-c1)];
            erasure_pattern(erasures)=1;
            Y(Y>q-1)=0;
            [Uhat,errs]=step(rsDecoder,Y,erasure_pattern');
            if(errs==-1)
                continue;
            end
            Xhat=rsEncoder(Uhat);
            if(isequal(Xhat(K+c1+1:K+c1+c2),Par(1:c2)))
                uhat=reshape(transpose(fliplr(de2bi(Uhat,l))),1,K*l);
                if(len_last<l)
                    uhat((K-1)*l+1:K*l-mod(k,l))=[];
                end
                return;
            end
        end
    
        %Secondary check
        if(secondary && abs(d)<lim)
            if(abs(d)<=c1)
                P=sign(d+eps)*P2{abs(d)+1};
            else
                P=sign(d+eps)*patterns_beta(abs(d),K,0,len_last,c1);
            end
            for i=1:size(P,1)
                lengths=ones(1,K+c1)*l;
                lengths(K)=len_last;
                lengths=lengths+P(i,:);
                Y=divide_vector(yE,lengths);
                Y=[Y';Par];
                erasure_pattern=[zeros(1,N-c+c1),ones(1,c-c1)];
                erasure_pattern(P(i,:)~=0)=1;
                Y(Y>q-1)=0;
                [Uhat,errs]=step(rsDecoder,Y,erasure_pattern');
                if(errs==-1)
                    continue;
                end
                Xhat=rsEncoder(Uhat);
                if(isequal(Xhat(K+c1+1:K+c1+c2),Par(1:c2)))
                    uhat=reshape(transpose(fliplr(de2bi(Uhat,l))),1,K*l);
                    if(len_last<l)
                        uhat((K-1)*l+1:K*l-mod(k,l))=[];
                    end
                    return;
                end
            end
        end
    end
end

function [uhat] = rep_decode_gamma(y,repetitions,k)
    uhat=[];
    while(numel(uhat)<k && ~isempty(y))
        t=min(repetitions,length(y));
        seg=y(1:t);
        bit=mode(seg);
        uhat=[uhat, bit];
        y(1:t)=[];
    end
    if(length(uhat)<k)
        uhat=[uhat, randi([0,1],[1, k-length(uhat)])];
    elseif(length(uhat)>k)
        uhat(k+1:end)=[];
    end
end