% expected improvement as a objective function
function [ei, ei_grad] = nei_objective(x_star, gp_model, ...
    x_train, y_train, y_min, exclude)

if nargout > 1 
    [ei, ei_grad] = ei_objective(x_star, gp_model, x_train, y_train, y_min, exclude);
    ei = -ei;
    ei_grad = -ei_grad;
else
    ei = ei_objective(x_star, gp_model, x_train, y_train, y_min, exclude);
    ei = -ei;
end

end