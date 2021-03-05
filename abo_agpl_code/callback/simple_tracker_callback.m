function simple_tracker_callback(problem, models, x, y, i, context)

y_first = min(problem.initial_y);
y_best  = min([problem.initial_y; y]);
gap     = (y_first - y_best)/(y_first - problem.optimum);

%acq     = context.acq;
max_acq = context.max_acq;

if problem.verbose
    fprintf('%s %s iter: %3d; f_last: %6.5f; f_min: %6.5f; acq: %6.6f; gap: %1.5f\n', ...
        datestr(now,'yy-mmm-dd-HH:MM'), problem.name, i, y(end), y_best, max_acq, gap);
    num_param = size(models{end}.parameters.L,1);
    fprintf('num_models %d; num_param: %d \n', size(models,2), num_param);
    
end


end

