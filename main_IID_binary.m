%Code parameters
k=133;
c1=8; c2=2; 
l=7;
len_last=mod(k-1,l)+1;
K=ceil(k/l);
t=5;

%Decoder setting
lim=5;
depth=zeros(1,lim);
depth(1)=2; depth(2)=1; depth(3)=0;
secondary=1;


%Simulation setup
total_iter=1e3;
Pe=0.001:0.001:0.01;
Pd=Pe/3; Pi=Pe/3; Ps=Pe/3; 
%Pi=Pe/7; Pd=2*Pe/7; Ps=4*Pe/7; 

%Pre-computing some patterns
Patterns_primary=burst_patterns(K+c1,c1); 
Patterns_secondary=preCompute_Patterns(depth,K,len_last,lim,c1);
disp("Done precomputing patterns");

%Initalizations
failure=zeros(1,length(Pe));
error=zeros(1,length(Pe));
total_errors=zeros(1,length(Pe));

%Main
for j=1:length(Pe)

    %Initalize counters
    countS=0; %succesful decoding
    countF=0; %decoding failure uhat empty
    countE=0; %decoding error uhat ~= u

    parfor iter=1:total_iter
        u=randi([0,1],1,k);
        [x,n,N,K,q] = GC_Encode_IID(u,l,c1,c2,t);
        y=binary_channel(n,x,Pd(j),Pi(j),Ps(j));
        uhat = GC_Decode_IID(y,n,k,l,N,K,c1,c2,q,secondary,len_last, lim, Patterns_primary, Patterns_secondary,t);
        
        if(isequal(uhat,u))
            countS=countS+1;
        elseif(isempty(uhat))
            countF=countF+1;
        else
            countE=countE+1;
        end
    end

    failure(j)=countF/total_iter;
    error(j)=countE/total_iter;
    total_errors(j)=1-(countS/total_iter);
    disp(['Edit Prob: ' num2str(Pe(j))  ', Failure rate: ' num2str(failure(j)) ', Error rate: ' num2str(error(j)) ', Total: ' num2str(total_errors(j))]);
end
figure(1)
semilogy(Pe,total_errors)
