%Code parameters
k=133; 
l=floor(log2(k));
len_last=mod(k-1,l)+1;
K=ceil(k/l);

%Simulation setup
total_iter=1e3;
w=l+1:l:4*l+1;
Pe=0.99;
Pd=Pe/3; Pi=Pe/3; Ps=Pe/3;

%Initalizations
errors=zeros(1,length(w));
code_rate=zeros(1,length(w));
error_rate=zeros(1,length(w));

%Main
for j=1:length(w)
    c=2*((w(j)-1)/l+1);
    c1=(w(j)-1)/l+1;
    c2=c-c1;
    P=burst_patterns(K,c1);
    %Initalize counters
    countS=0; %succesful decoding
    countF=0; %decoding failure uhat empty
    parfor iter=1:total_iter
        u=randi([0,1],1,k);
        [x,n,N,K,q] = GC_Encode_loc(u,l,c,w(j));
        y=binary_channel(w(j),x,Pd,Pi,Ps);
        uhat = GC_Decode_loc(y,n,k,l,N,K,c,c1,c2,q,len_last,P,w(j));
        if(isequal(uhat,u))
            countS=countS+1;
        end
    end
    errors(j)=1-countS/total_iter;
    code_rate(j)=k/(k+3*(w(j)+1)+(c1+c2)*l);
    error_rate(j)=(Pe*w(j))/(k+3*(w(j)+1)+(c1+c2)*l);
    disp(['Window length: ' num2str(w(j)) ', Avg. Edit Rate: ' num2str(error_rate(j)) ', Code Rate: ' num2str(code_rate(j)) ', Decoding Error  Rate: ' num2str(errors(j))]);
end
