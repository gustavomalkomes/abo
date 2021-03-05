function partitions = random_set_partition(n,s)
% sample s random partitions from a set of size n
% n - number of elements in the set
% s - number of samples
% we generate random partitions according to the urn model
% [1] - Stam, Generation of a random partition of a finite set 
% by an urn model. J. Combin. Theory Ser. A 35 (1983), no. 2, 231?240;
% [2] - Generating uniform random partitions,
% http://djalil.chafai.net/blog/2012/05/03/generating-uniform-random-partitions/

Bn = bell_number(n);

cum = 0;
i = 1;
cum_total = 1 - 10^-7;
max_iter = 500;

while cum < cum_total && i < max_iter
    a(i) = probability(i,n,Bn);
    cum = cum + a(i);
    i = i + 1;    
end
cdf = cumsum(a);

u = rand(s,1);

samples = zeros(s,1);
for i=1:numel(cdf)
    selection = u <= cdf(i);
    if any(selection)
        samples(selection) = i;
        u(selection) = 2; % 'deleting' those samples
    end
       
   if all(samples)
       break;
   end
end

for i = 1:size(samples,1)
   k = samples(i);
   colors = randi([1,k],n,1);
   set = [];
   for j = 1:n
       if ~ismember(j,set(:))
        curr_set = find(colors == colors(j));
        set = [set, curr_set', 0];
       end
   end
   partitions{i} = set(1:end-1);
end

end

function p = probability(k, n, Bn)
    p = k^n / (factorial(k)*exp(1)*Bn);
end