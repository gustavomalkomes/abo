function problem = initilization_random_points(problem, d, ...
    num_points, num_init_points) 

p                 = sobolset(d);
x_pool            = net(p, num_points);
p                 = sobolset(d,'Skip',1e2,'Leap',1e5);
p                 = scramble(p,'MatousekAffineOwen');
x_pool            = [x_pool; net(p, num_points)];

num_grid_points   = size(x_pool,1);

used           = false(num_grid_points,1);
init_idx       = randperm(num_grid_points,num_init_points);
used(init_idx) = true;

x              = x_pool(init_idx,:);
y              = zeros(num_init_points,1);

for k = 1:num_init_points
    y(k) = problem.label_oracle([], [], [], x(k,:));
end

problem.x_pool               = x_pool;
problem.used                 = used;
problem.initial_x            = x;
problem.initial_y            = y;