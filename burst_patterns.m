function [P] = burst_patterns(K,d)
    P=[];
    for i=1:K-d+1
        P=[P;i:i+d-1];
    end
end