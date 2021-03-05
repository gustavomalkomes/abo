function create_jobs(exp_name)

folder     = './jobs_robotics/';
folder_log = [folder, 'logs/'];

if ~exist(folder,'dir')
    mkdir(folder);
end

if ~exist(folder_log,'dir')
    mkdir(folder_log);
end

algorithms = {'abo', 'SEard', 'bom', 'mcmc'};

switch exp_name
    case 'grid'
        datasets = {'svm_on_grid', 'lda_on_grid', 'logreg_on_grid'};
        
    case 'opt'
        datasets = {'ackley_2', 'branin', 'drop', 'camel6', ...
            'griewank_2','rosen', 'egg', ...
            'hart3', 'levy_3', 'ackley_5', 'griewank_5', ...
            'hart6'};
        
    case 'opt2'
        datasets = {'beale', 'rastr', 'rastr_d4', 'goldpr', 'shubert'};
        
    case 'robot'
        datasets = {'robot_pushing_3_on_grid1', ...
            'robot_pushing_4_on_grid1'};
        
    case 'cosmo'
        datasets = {'cosmological_on_grid'};
        
    otherwise
        error('Unknown configuration')
end

% Use 2017a
cmd = '/cluster/cloud/matlab-2017a/bin/matlab -nodisplay -singleCompThread -r "cd ..;';

execution_list = cell(numel(datasets)*numel(algorithms),1);
n_jobs = 0;
for i = 1:numel(datasets)
    for seed = 0:1:19
        for j = 1:numel(algorithms)
            fileName = [datasets{i}, '_', algorithms{j}, '_', num2str(seed)];
            n_jobs = n_jobs + 1;
            execution_list{n_jobs} = fileName;
            
            fileID = fopen([folder,fileName, '.sh'],'w');
            
            fprintf(fileID,'#$ -cwd\n#$ -soft -pe smp 1\n\n');
            
            exp_name_str = sprintf('''%s''',exp_name);
            algo_str = sprintf('''%s''',algorithms{j});
            dataset_str = sprintf('''%s''',datasets{i});
            
            str = sprintf('%s run_experiment(%s, %s, %s, %d); exit"', ...
                cmd, exp_name_str, algo_str, dataset_str, seed);
            
            cmd_full = sprintf('%s',str);

            %cmd_full = sprintf('%s >> %s%s_%s_%s_d_s%d',str, folder_log, ...
            %    exp_name, datasets{i}, algorithms{j}, seed);
            
            fprintf(fileID,'%s\n', cmd_full);
            
            fclose(fileID);
        end
        
    end
end

jobs_per_batch = 180;
num_jobs = numel(execution_list);
num_full_batches = floor(num_jobs/jobs_per_batch);
num_jobs_per_file = repmat(jobs_per_batch, 1, num_full_batches);
remaining_jobs = mod(num_jobs,jobs_per_batch);
if remaining_jobs > 0
    num_jobs_per_file = [num_jobs_per_file, remaining_jobs];
end
num_files = size(num_jobs_per_file,2);

c = 1;
for j = 1:num_files
    fileID = fopen([folder,'fire_all_', num2str(j),'.sh'],'w');
    for i = 1:num_jobs_per_file(j)
        fprintf(fileID,'qsub %s.sh -q all.q\n', execution_list{c});
        fprintf(fileID,'sleep 10\n');
        c = c + 1;
    end
    fclose(fileID);
end

end
