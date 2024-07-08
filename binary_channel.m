function [y] = binary_channel(w,x,Pd,Pi,Ps)
%Random edit channel - binary case
        n=length(x);
        pos=randi([1,n-w+1],1,1);
        y=x(1:pos-1);
        for i=pos:pos+w-1
            r=randsample([0,1,2,3],1,true,[1-Pd-Pi-Ps, Pd, Pi, Ps]);
            if(r==0)
                    y=[y,x(i)];
            elseif(r==2)
                    y=[y,randi([0,1],1,1),x(i)];
            elseif(r==3)
                    y=[y,~x(i)];
            end
        end
        y=[y,x(pos+w:end)];
end