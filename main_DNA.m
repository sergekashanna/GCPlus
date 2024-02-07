%Setup and file generation
fragments=10000;
k=168; 
File=randi([0,1],fragments,k);
Pe=0.001:0.001:0.01;
%Pd=2*Pe/7; Pi=Pe/7; Ps=4*Pe/7;
Pd=Pe/3; Pi=Pe/3; Ps=Pe/3;

%Outer code - encoding
outer_redundancy=0.1*fragments; %upper bound on redundancy
q=2^14;
[File_binary_encoded, File_qary_encoded]=Encode_outerRS(File,outer_redundancy,q);
disp("Done outer code");
 
%Inner encoder parameters
c=10; c1=8; c2=2; 
l=8;
K=ceil(size(File_binary_encoded,2)/l);
len_last=mod(size(File_binary_encoded,2)-1,l)+1;
t=5;

%Inner decoder parameters
lim=5;
depth=zeros(1,lim); depth(1)=1; depth(2)=1; depth(3)=0;
secondary=1;

% Pre-computing
Patterns_primary=burst_patterns(K+c1,c1); 
Patterns_secondary=preCompute_Patterns(depth,K,len_last,lim,c1);
disp("Done precomputing patterns");

%Initalizations
redundancy=zeros(1,length(Pe));

%Main
for j=1:length(Pe)
    counter=0;
    erasure_pattern=ones(1,size(File_binary_encoded,1));
    F=[];
    for f=1:size(File_binary_encoded,1)

        %Inner code
        u=File_binary_encoded(f,:);
        [x,n,N,K,q1]=GC_Encode_IID(u,l,c1,c2,t);
        y=DNA_channel(x,Pd(j),Pi(j),Ps(j));
        uhat=GC_Decode_DNA(y,n,length(u),l,N,K,c,c1,c2,q1,secondary,len_last, lim, Patterns_primary, Patterns_secondary,t);
        
        %File to be decoded by the outer code
        if(isempty(uhat))
            F=[F; zeros(1,length(u))];
        else
            erasure_pattern(f)=0;
            F=[F; uhat];
        end

        %Tracking decodability to evaluate maximum achievable rate 
        if(isequal(uhat,u))
            counter=counter+1;
        elseif(isempty(uhat))
            counter=counter-1;
        else
            counter=counter-2;
        end

        %Track progress
        if(mod(f,1000)==0 || counter==fragments)
            disp(['Pe: ' num2str(Pe(j)) ', Oligo#: ' num2str(f) ', Counter: ' num2str(counter)]);
        end

        if(counter==fragments) %file can be now decoded
            F=[F; zeros(fragments+outer_redundancy-f,length(u))];

            %Test decodability to verify that rate is achievable
            Fhat=Decode_outerRS(F, fragments, erasure_pattern, q);
            if(~isequal(Fhat,File_qary_encoded(1:fragments,:)))
                disp('Decoding error, aborting the simulation');
            else
                redundancy(j)=f-fragments;
                disp(['Edit Error Probability: ' num2str(Pe(j))  ', Maximum achievable rate: ' num2str(1-redundancy(j)/fragments)]);
            end
            
            break;
            
        end
    end
end

figure(1)
Rates=1-redundancy/fragments;
semilogy(Pe,Rates)


function [F_encoded, Fq_encoded] = Encode_outerRS(F,r,q)

    l=log2(q);
    Fq=binary_to_qary_File(F,l);

    %Outer RS encoder setup
    rsEncoder = comm.RSEncoder(size(F,1)+r,size(F,1),'BitInput',false);
    primPolyDegree = l;
    rsEncoder.PrimitivePolynomialSource = 'Property';
    rsEncoder.PrimitivePolynomial = de2bi(primpoly(primPolyDegree,'nodisplay'),'left-msb');
    
    %Outer encoding
    Fq_encoded=[];
    for i=1:size(Fq,2)
        Fq_encoded=[Fq_encoded,rsEncoder(Fq(:,i))];
    end

    F_encoded=qary_to_binary_File(Fq_encoded,l);
end

function [Fhat] = Decode_outerRS(F,frags,erasure_pattern,q)

    Fq=binary_to_qary_File(F,log2(q));

    %Outer RS decoder setup
    rsDecoder=comm.RSDecoder(size(Fq,1),frags,'BitInput',false);
    rsDecoder.ErasuresInputPort=true;
    rsDecoder.NumCorrectedErrorsOutputPort=true;
    rsDecoder.PrimitivePolynomialSource = 'Property';
    rsDecoder.PrimitivePolynomial = de2bi(primpoly(log2(q),'nodisplay'),'left-msb');

    %Outer decoding
    Fhat=[];
    for i=1:size(Fq,2)
        Fhat=[Fhat,step(rsDecoder,Fq(:,i),erasure_pattern')];
    end
end

function [Fq] = binary_to_qary_File(F,l)
    K=ceil(size(F,2)/l);
    Fq=[];
    for i=1:size(F,1)
        u=F(i,:);
        u=[u(1:(K-1)*l),zeros(1,K*l-numel(u)),u((K-1)*l+1:end)];
        A=reshape(u,l,K);
        U=bi2de(fliplr(A'));
        Fq=[Fq;U'];
    end
end

function [F] = qary_to_binary_File(Fq,l)
    F=[];
    for i=1:size(Fq,1)
        X=Fq(i,:);
        x=reshape(transpose(fliplr(de2bi(X,l))),1,numel(X)*l);
        F=[F;x];
    end
end