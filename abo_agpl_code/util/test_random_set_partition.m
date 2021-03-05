n = 3;
num_samples = 100000;
partition = random_set_partition(n,num_samples);

total_counter = [];
for s = 1:size(partition,2)
    counter = 0;
    curr_set = partition{s};
    if ~isempty(curr_set)
        for i=s:size(partition,2)
            set = partition{i};
            if numel(set) == numel(curr_set) && all(set == curr_set)
                counter = counter + 1;
                partition{i} = [];
            end
        end
        total_counter = [total_counter, counter];
    end
end
total_counter./sum(total_counter)
