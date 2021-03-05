function plot_progress(problem, models, x, y, i, context)

figure(1)

for m = 1:numel(models)
    
    gp_model = models{m}.parameters;
    name = covariance2str(models{m}.parameters.covariance_function);
    x_train = x(1:end-1,:);
    y_train = y(1:end-1,:);
    x_last = x(end,:);
    y_last = y(end,:);
    
    % visualization
    if size(x,2) == 1
        figure(1);
        
        % gp predictions
        z = sort(problem.x_pool);
        [~, ~, mu,cov] = models{m}.prediction(gp_model.theta, ...
            gp_model.inference_method, ...
            gp_model.mean_function, gp_model.covariance_function, ...
            gp_model.likelihood, x_train, y_train, z);
        
        cov(cov < 0) = 0;

        % ei
        [x_pool, x_pool_order] = sort(context.x_pool);
        ei = context.ei(x_pool_order);
        
        % predictions
        ax1 = subplot(4,1,1); % top subplot
        hold off;
        f = [mu+2*sqrt(cov); flip(mu-2*sqrt(cov),1)];
        fill([z; flipdim(z,1)], f,  [0.8, 0.8, 1],'edgecolor', 'none')
        hold on;
        plot(z, mu,'Color',[0,0,200]/255); % predictions
        plot(x_train,y_train, 'b.', 'MarkerSize', 30); % training points
        plot(x_last, y_last, 'r.','MarkerSize',30); % last chosen point
        title(name);
        hold off;
        
        title(ax1,['Predictions for ', name])
        
        ax2 = subplot(4,1,2); % bottom subplot
        plot(ax2,x_pool,ei)
        title(ax2,['EI for ', name])
        
        ax3 = subplot(4,1,3);
        plot(ax3,z,mu)
        title(ax3,['MU for ', name])
        
        ax4 = subplot(4,1,4);
        plot(ax4,z,cov)
        title(ax4,['Cov for ', name])
        
    elseif size(x,2) == 2
        
        
        figure(1)
        set(gcf, 'units','normalized','outerposition',[0 0 1 1])
        clf;
        
        [xx ,yy] = meshgrid(0:0.05:1, 0:0.05:1);
        % gp predictions
        z = [xx(:), yy(:)];
        
        [~, ~, mu,cov] = models{m}.prediction(gp_model.theta, ...
            gp_model.inference_method, ...
            gp_model.mean_function, gp_model.covariance_function, ...
            gp_model.likelihood, x_train, y_train, z);
        
        cov(cov < 0) = 0;
        
        mu_plot = reshape(mu, size(xx));
        
        ax1 = subplot(2,2,1); % top subplot
        %surf(ax1, xx,yy,mu_plot, 'EdgeColor', 'none')
        contourf(ax1,xx,yy,mu_plot)
        %view(ax1, 0, 90);
        colorbar EastOutside
        hold on;
        plot(x(:,1), x(:,2), 'k.', 'MarkerSize', 20);
        plot(x_last(:,1), x_last(:,2), 'ro', 'MarkerSize', 12, ...
            'LineWidth', 2, 'MarkerEdgeColor','k', 'MarkerFaceColor','r');
        
        title(ax1,['Predictions for ', name])
        
        ax2 = subplot(2,2,2); % bottom subplot
        ei = ei_objective(z,models{m},x_train,y_train,min(y_train));
        ei_plot = reshape(ei, size(xx));
        
        hold on;
        %surf(ax2, xx,yy,ei_plot, 'EdgeColor', 'none')
        contourf(ax2,xx,yy,ei_plot);
        colorbar EastOutside
        plot(x(:,1), x(:,2), 'k.', 'MarkerSize', 20);
        plot(x_last(:,1), x_last(:,2), 'ro', 'MarkerSize', 12, ...
            'LineWidth', 2, 'MarkerEdgeColor','k', 'MarkerFaceColor','r');
        
        title(ax2,['EI for ', name])
        
        ax3 = subplot(2,2,3);
        mu_plot = reshape(mu, size(xx));
        contourf(ax3,xx,yy,mu_plot);
        hold on;
        plot(x(:,1), x(:,2), 'k.', 'MarkerSize', 20);
        plot(x_last(:,1), x_last(:,2), 'ro', 'MarkerSize', 12, ...
            'LineWidth', 2, 'MarkerEdgeColor','k', 'MarkerFaceColor','r');
        
        colorbar EastOutside
        title(ax3,['MU for ', name])
        
        ax4 = subplot(2,2,4);
        cov_plot = reshape(cov, size(xx));
        contourf(ax4,xx,yy,cov_plot);
        
        hold on;
        plot(x(:,1), x(:,2), 'k.', 'MarkerSize', 20);
        plot(x_last(:,1), x_last(:,2), 'ro', 'MarkerSize', 12, ...
            'LineWidth', 2, 'MarkerEdgeColor','k', 'MarkerFaceColor','r');
        
        colorbar EastOutside
        title(ax4,['Cov for ', name])
        
        plot(x(:,1), x(:,2), 'k.', 'MarkerSize', 20);
        plot(x_last(:,1), x_last(:,2), 'r.', 'MarkerSize', 30);
        
        
    end
    
    fprintf('Iteration %d.  max(ei) = %f; y_min: %f\n', i, max(ei), min(y));
    
    if ~exist('./tmp/', 'dir')
        mkdir('./tmp/');
    end
    print(['./tmp/progress_', name, '_', num2str(i)], '-dpng');
    
    
end



end

% expected improvement as a objective function
function ei = ei_objective(x_star, gp_model, ...
    x_train, y_train, y_min)

% load gp parameters
model = gp_model.parameters;

% gp predictions
[~, ~, mu,cov] = gp_model.prediction(model.theta, ...
    model.inference_method, ...
    model.mean_function, model.covariance_function, ...
    model.likelihood, x_train, y_train, x_star);

% make sure that cov > 0
cov((cov<0)) = 0;
sigma = sqrt(cov);

% compute expected improvement
delta = (y_min - mu);
u     = delta./sigma;
u_pdf = normpdf(u);
u_cdf = normcdf(u);

ei  = delta .* u_cdf + sigma .* u_pdf;
end