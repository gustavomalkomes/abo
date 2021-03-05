function Si = greedy_furthest(dist,k)
    n     = size(dist,1);
    Si    = zeros(k,1);
    %Si(1) = randperm(n,1);
    Si(1) = 1;
    for i=2:k,
        [~,c] = max(min(dist(:,Si(1:i-1)),[],2));
        Si(i) = c;
    end
end