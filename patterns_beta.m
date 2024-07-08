function [P] = patterns_beta(d,K,depth,len_last, c1)
    P=[];
    M=min(c1,abs(d)+2*depth);
    minValue=-depth;
    maxValue=abs(d)+depth;
    for i=1:M
        C=generateIntegerSolutions(abs(d), i, minValue, maxValue, depth);
        pos=nchoosek(1:K+c1,i);
        for q=1:size(pos,1)
            for j=1:size(C,1)
                temp=zeros(1,K+c1);
                temp(pos(q,:))=C(j,:);
                if(temp(K)<= len_last)
                    P=[P; temp];
                end
            end
        end
    end
    P=sortRowsByNonZerosAndL1Norm(P);
end

function solutionsMatrix = generateIntegerSolutions(n, k, minValue, maxValue, depth)
    % Create vectors of values within the specified ranges
    ranges = arrayfun(@(minVal, maxVal) minVal:maxVal, minValue, maxValue, 'UniformOutput', false);
    
    % Generate all combinations of values within the ranges
    [X{1:k}] = ndgrid(ranges{:});
    
    % Concatenate the values into a matrix
    solutionsMatrix = cell2mat(cellfun(@(x) x(:), X, 'UniformOutput', false));
    
    % Filter solutions that satisfy the equation x1 + x2 + ... + xk = n
    validSolutions = sum(solutionsMatrix, 2) == n;

    % Filter solutions where all variables have non-zero values
    validSolutions = validSolutions & all(solutionsMatrix ~= 0, 2);

    % Filter solutions based on the L1 norm condition
    validSolutions = validSolutions & vecnorm(solutionsMatrix, 1, 2) - n <= 2*depth;
    
    % Extract only the valid solutions
    solutionsMatrix = solutionsMatrix(validSolutions, :);
end

function sortedMatrix = sortRowsByNonZerosAndL1Norm(inputMatrix)
    % Count the number of non-zero elements in each row
    nonZeroCounts = sum(inputMatrix ~= 0, 2);

    % Calculate the L1 norm of each row
    l1Norms = sum(abs(inputMatrix), 2);

    % Create a table with row indices, non-zero counts, and L1 norms
    tableData = table((1:size(inputMatrix, 1))', nonZeroCounts, l1Norms, 'VariableNames', {'Index', 'NonZeroCount', 'L1Norm'});

    % Sort the table based on non-zero counts and L1 norms
    sortedTable = sortrows(tableData, {'NonZeroCount', 'L1Norm'});

    % Extract the sorted matrix based on the sorted indices
    sortedIndices = sortedTable.Index;
    sortedMatrix = inputMatrix(sortedIndices, :);
end
