function context = save_tracker_callback(output_file, problem, models, ~, ...
    y, i, context)

y_first = min(problem.initial_y);
y_best  = min([problem.initial_y; y]);

gap     = NaN;
if isfield(problem, 'optimum') && ~isnan(problem.optimum)
    gap = (y_first - y_best)/(y_first - problem.optimum);
end

max_acq     = context.max_acq;
context.gap = gap;

if problem.verbose
    fprintf('%s %s iter: %3d; f_last: %6.5f; f_min: %6.5f; acq: %6.6f; gap: %1.5f\n', ...
        datestr(now,'yy-mmm-dd-HH:MM'), problem.name, i, y(end), y_best, max_acq, gap);
end


time_acquisition = context.time_acquisition;
time_model_search = context.time_model_search;
time_update = context.time_update;
time_oracle = context.time_oracle;

fileID = fopen(output_file,'a');
fprintf(fileID, '%15s, %3.2i, %10.5f, %8.5f, %3.2d, %8.3f, %8.3f, %8.3f, %8.3f \n', ...
    problem.name, i, y_best, gap, numel(models), ...
    time_update, time_model_search, time_acquisition, time_oracle);
fclose(fileID);


end

