%Code parameters
k=168; 
l=8;
len_last=mod(k-1,l)+1;
K=ceil(k/l);

%Simulation setup
total_iter=1e5;
w=2:5; %window size in NTs
Pe=0.99;
Pd=Pe/3; Pi=Pe/3; Ps=Pe/3;

%Initalizations
errors=zeros(1,length(w));
code_rate=zeros(1,length(w));
error_rate=zeros(1,length(w));

%Main
for j=1:length(w)
    c=4;
    c1=2;
    c2=2;
    P=burst_patterns(K,c1);
    %Initalize counters
    countS=0; %succesful decoding
    countF=0; %decoding failure uhat empty
    W=3*(2*w(j)+1);
    if(mod(W,2)~=0)
        W=W+1;
    end

    parfor iter=1:total_iter
        u=randi([0,1],1,k);
        [x,n,N,K,q] = GC_Encode_loc_DNA(u,l,c,2*w(j));
        y=DNA_channel_loc(w(j),x,Pd,Pi,Ps);
        uhat = GC_Decode_loc(y,n,k,l,N,K,c,c1,c2,q,len_last,P,2*w(j),W);
        if(isequal(uhat,u))
            countS=countS+1;
        end
    end
    errors(j)=1-countS/total_iter;
    code_rate(j)=k/(k+W+(c1+c2)*l);
    error_rate(j)=(Pe*w(j))/((k+W+(c1+c2)*l)/2);
    disp(['Window length: ' num2str(w(j)) ', Avg. Edit Rate: ' num2str(error_rate(j)) ', Code Rate: ' num2str(code_rate(j)) ', Total Decoding Error Rate: ' num2str(errors(j))]);
end
