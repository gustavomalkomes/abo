% expected improvement as a objective function
function [ei, ei_grad] = ei_objective(x_star, gp_model, ...
    x_train, y_train, y_min, exclude)

% load gp parameters
model = gp_model.parameters;

% gp predictions
if size(model.posterior.alpha,1) ~= size(y_train,1)
    x_train = x_train(~exclude,:);
end

[~, ~, mu,cov] = gp_model.prediction(model.theta, ...
    model.inference_method, ...
    model.mean_function, model.covariance_function, ...
    model.likelihood, x_train, model.posterior, x_star);

% make sure that cov > 0
cov((cov<0)) = 0;
sigma = sqrt(cov);

% compute expected improvement
delta = (y_min - mu);
u     = delta./sigma;
u_pdf = normpdf(u);
u_cdf = normcdf(u);

ei        = delta .* u_cdf + sigma .* u_pdf;
ei(ei<0) = 0; 

if nargout == 2
     error('TODO')
%     [n,d]  = size(x_star);
% 
%     dmu_dx = zeros(d,n);
%     dK_dx  = zeros(d,n);
%     e      = 1e-12;  
%     
%     for j = 1:d
%         dx = zeros(1,d);
%         dx(j) = dx(j) + e;                            
%         
%         dx_plus  = x_star + dx/2;
%         dx_minus = x_star - dx/2;
% 
%         [~, ~, mu,cov] = gp_model.prediction(model.theta, ...
%             model.inference_method, ...
%             model.mean_function, model.covariance_function, ...
%             model.likelihood, x_train, model.posterior, [dx_minus; dx_plus]);
% 
%         if n == 1
%             dmu_dx(j, :) = diff(mu)/(e);
%             dK_dx(j, :)  = diff(cov)/(e);        
%         else
%             diff_mu  = mu(n+1:end,:) - mu(1:n,:);
%             diff_cov = cov(n+1:end,:) - cov(1:n,:);
%             dmu_dx(j, :) = diff_mu/(e);
%             dK_dx(j, :)  = diff_cov/(e);                    
%         end
%     end
% 
%     dei_dmu    = -u_cdf;
%     dei_dsigma = u_pdf;
%     
%     if n == 1
%         ei_grad = dei_dmu.*dmu_dx + (dei_dsigma./(2*sigma)).*dK_dx;
%     else
%         dei_dK  = dei_dsigma./(2*sigma);
%         ei_grad = repmat(dei_dmu',d,1).*dmu_dx + ...
%             repmat(dei_dK', d,1).*dK_dx;
%     end
end

end