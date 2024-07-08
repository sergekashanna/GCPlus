function [Y,parts] = divide_vector(x, y)
    Y=[];
    n = length(y); % number of parts
    m = sum(y); % total length of x
    if m ~= length(x)
        error('The length of x must equal the sum of elements in y.')
    end
    
    parts = cell(n,1); % initialize cell array to store parts
    start_idx = 1; % initialize starting index
    for i = 1:n
        end_idx = start_idx + y(i) - 1; % calculate ending index
        parts{i} = x(start_idx:end_idx); % extract part from x
        if(isempty(parts{i}))
            Y=[Y,0];
        else
            Y=[Y,bi2de(fliplr(parts{i}))];
        end
        start_idx = end_idx + 1; % update starting index for next part
    end
end