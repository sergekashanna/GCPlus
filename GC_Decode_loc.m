function uhat = GC_Decode_loc(y,n,k,l,N,K,c,c1,c2,q,len_last,P1,w,W)
    
    %Initilaizations
    uhat=[];
    d=numel(y)-n;
    buff=[ones(1,w+1),zeros(1,w+1),ones(1,w+1)];
    
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

 
    %Detect & correct edit errors using buffer
    if(d==0)
        yE=y;
        yE(k+1:k+W)=[];
        lengths=ones(1,K+c)*l;
        lengths(K)=len_last;
        Y=divide_vector(yE,lengths);
        Y=Y';
        erasure_pattern=[zeros(1,N-c+c1+c2),ones(1,c-c1-c2)];
        [Uhat, errs]=step(rsDecoder,Y,erasure_pattern');
        if(errs~=-1)
            uhat=reshape(transpose(fliplr(de2bi(Uhat,l))),1,K*l);
            if(len_last<l)
                uhat((K-1)*l+1:K*l-mod(k,l))=[];
            end
            return;
        end
    else
        if(isequal(y(k+w+2+d:k+3*w+3+d),buff(w+2:end)))
            yE=y(1:k+d);
            p=y(end-c*l+1:end);
            A=reshape(p,l,c);
            Par=bi2de(fliplr(A'));
        else
            uhat=y(1:k);
            return;
        end
    end
    
    %Primary check
    if(d~=0)
        for i=1:size(P1,1)
            lengths=ones(1,K)*l;
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
            if(isequal(Xhat(K+c1+1:K+c1+c2),Par(c1+1:c1+c2)))
                uhat=reshape(transpose(fliplr(de2bi(Uhat,l))),1,K*l);
                if(len_last<l)
                    uhat((K-1)*l+1:K*l-mod(k,l))=[];
                end
                return;
            end
        end
    end
end