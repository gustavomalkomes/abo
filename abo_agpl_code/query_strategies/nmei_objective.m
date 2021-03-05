% expected improvement as a objective function
function [ei, ei_grad] = nmei_objective(x_star, gp_model, ...
    x_train, y_train, y_min, exclude, model_posterior)

if nargout > 1 
    [ei, ei_grad] = mei_objective(x_star, gp_model, x_train, y_train, y_min, exclude, model_posterior);
    ei = -ei;
    ei_grad = -ei_grad;
else
    ei = mei_objective(x_star, gp_model, x_train, y_train, y_min, exclude, model_posterior);
    ei = -ei;
end

if size(x_star,2) ~= size(x_train,2)
    ei = ei';
end

% if any(x_star > 1) || any(x_star < 0)
%      ei = ei + x_star.^2;
%      ei_grad = ei_grad + 2*x_star'; 
% end


end