clear all;
clc;

addpath(genpath('../'));

filename = './graphs/samples_from_priors2';

covariances_root_names = {'SE','PER','RQ','LIN'};
level                    = 1;
true_covariance          = 1;
num_lines                = 3;
data_noise               = 0.0001;

param = gpr_hyperparameters();

figure(2);
clf;

subplot(5,1,1);
x_plot     = linspace(0,1,200)';
x_plot     = sort(x_plot);
y_plot = lognpdf(x_plot,param.length_scale_mean,sqrt(param.length_scale_var));
plot(x_plot,y_plot);
title('lengthScale');

subplot(5,1,2);
x_plot = 0.1:0.01:2;
y_plot = lognpdf(x_plot,param.output_scale_mean,sqrt(param.output_scale_var));
plot(x_plot,y_plot);
title('outputScale');

subplot(5,1,3);
x_plot = 0:0.001:5;
y_plot = lognpdf(x_plot,param.p_length_scale_mean,sqrt(param.p_length_scale_var));
plot(x_plot,y_plot);
title('periodicLengthScale');

subplot(5,1,4);
x_plot = 0:0.001:0.6;
y_plot = lognpdf(x_plot,param.p_mean,sqrt(param.p_var));
plot(x_plot,y_plot);
title('period');

subplot(5,1,5);
x_plot = 0:0.01:0.3;
y_plot = lognpdf(x_plot,param.alpha_mean,sqrt(param.alpha_var));
plot(x_plot,y_plot);
title('alpha');

subplot(6,1,6);
x_plot = 0:0.01:0.15;
y_plot = lognpdf(x_plot,param.lik_noise_std,sqrt(param.lik_noise_std_var));
plot(x_plot,y_plot);
title('noise');


param.options.num_restarts    = 3;
param.options.minimize_method = 'minimize';

covariances_root               = covariance_grammar_started(covariances_root_names,param);
[covariances,names]            = covariance_builder_grammar(covariances_root,level);
%names'

number_points   = 200;
xx              = linspace(0,1,number_points)';

colors = [37,52,148; ...
    0,136,55; ...
    65,182,196; ...
    44,127,184; ...
    166,219,160]/255;

%figure(1)
figure('units','normalized','outerposition',[0 0 0.8 0.8]);
clf;
for i=1:numel(covariances),
    subplot(4,numel(covariances)/4,i);
    hold on;
    for j=1:num_lines,
        true_gp = gpr_model_builder(covariances{i},data_noise);
        y = generate_synthetic_data(xx,true_gp);
        plot(xx,y,'color', colors(j,:),'LineWidth',2)
        %plot(xx,y)
    end
%     axis([0 1 0 1])
    title(sprintf('%s',covariance2str(covariances{i}.fun)));
end

set(gcf, 'Color', 'w');
%export_fig([filename,'.pdf'],'-painters');
