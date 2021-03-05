function num_hyps=get_num_hyps(names,base_names,base_hyps)
num_hyps = zeros(size(names));
for i=1:numel(names)
    count = 0;
    name = names{i};
    for k=1:length(base_names)
        count = count + base_hyps(k) * length(strfind(name,base_names{k}));
    end
    num_hyps(i) = count;
end
end