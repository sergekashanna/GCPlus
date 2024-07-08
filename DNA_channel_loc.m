function [y] = DNA_channel_loc(w,x,Pd,Pi,Ps)
%Random edit channel - DNA localized case
%Assuming x has even length
        pos=randi([1,length(x)/2-w+1],1,1);
        posb=2*(pos-1)+1;
        wb=2*w;
        y=x(1:posb-1);
        for i=posb:2:posb+wb-1
            r=randsample([0,1,2,3],1,true,[1-Pd-Pi-Ps, Pd, Pi, Ps]);
            if(r==0)
                y=[y,x(i:i+1)];
            elseif(r==2)
                y=[y,randi([0,1],1,2),x(i:i+1)];
            elseif(r==3)
                y=[y,sub_array(x(i:i+1))];
            end
        end
        y=[y,x(posb+wb:end)];
end

function random_binary_array = sub_array(input_array)
    % Check if input_array is binary
    if ~all(ismember(input_array, [0, 1]))
        error('Input array must contain only binary values (0s and 1s).');
    end
    
    % Generate all possible binary arrays of length n
    possible_arrays = dec2bin(0:2^length(input_array)-1) - '0';
    possible_arrays = possible_arrays(~ismember(possible_arrays, input_array, 'rows'), :);

    % Randomly select one array
    idx = randi(size(possible_arrays, 1));
    random_binary_array = possible_arrays(idx, :);
end