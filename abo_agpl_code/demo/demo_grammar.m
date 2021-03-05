
addpath(genpath('../'))

covariances_names = {'SE','M1'};
level             = 0;
data_noise        = 0.01;

% loading hyper-parameters
hyp = gpr_hyperparameters(data_noise);
% creating a initial set of covariances based on strings 
covariances = ther(covariances_names,hyp);

% expanding the set of covariances on the tree untill a certain level
[covariances,covariances_names] = covariance_builder_grammar(covariances,level);

% some handfull functions
fprintf('Array of covariances names:\n'); disp(covariances_names');
fprintf('Converting covariance function handler to string --> %s \n',covariance2str(covariances{1}.fun));
fprintf('Looking for a covariance index --> %d \n',find_covariance('(SE+M1)',covariances_names));
% note on find_covariances
% M1+SE will not be found because in covariances_names SE appear before M1 
% then, SE+M1 is the correct name; Additionaly, the parenthesis are
% important, so don't forget them.

%[base_covs_masked,base_names] = mask_kernels(base_covs, num_features);