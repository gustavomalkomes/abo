%select the next test point using expected improvement
function [next_x, scores] = select_best_candidate(best_y, x_star, mu_star, s2_star, times)
best_ei = -inf;
scores = zeros(1,length(x_star));
for i = 1:size(x_star,1)
    scores(i) = compute_ei(best_y, mu_star(i), sqrt(s2_star(i))) / times(i);
    if (scores(i) >= best_ei)
        best_ei = scores(i);
        next_x = x_star(i);
    end
end
end