function h = hellinger_distance(p_mu, p_K, q_mu, q_K)
% Squared Hellinger distance for two multivariate Gaussian distributions


% p_K and q_K are the same, just return 0
if (max(max(abs(p_K-q_K))) < eps)
    h = 0;
    return;
end

% % compute P det
% det_p = det(p_K);
% if det_p < 0
%     det_p = 0;
% end
%
% % compute Q det
% det_q = det(q_K);
% if det_q < 0
%     det_q = 0;
% end
%
% % compute the numerator
% numerator = (det_p*det_q)^(1/4);
% if numerator > 0
%     sum_pq = (1/2)*(p_K + q_K);
%
%     % compute the det of sum
%     det_K_sum   = det( sum_pq );
%     if det_K_sum < 0
%         det_K_sum = eps;
%     end
%
%     denominator = sqrt(det_K_sum);
%     base = numerator/denominator;
% else
%     base = 0;
% end
%
% % double check if base is nan or greater than one
% if isnan(base) || base > 1,
%     % if numerator and denominator are the same,
%     % base must be one
%     if abs(numerator-denominator) < 1e-6, % relatively loose tolerance
%         base = 1;
%     else
%         % this is a little bit arbitraty
%         warning('NaN found. Automatically setting NaN to 0');
%         h = 0;
%         return;
%     end
% end
%
%
% % check if we need to compute the exponent
% mu = p_mu - q_mu;
% if all(mu < eps)
%     h = 1 - base;
% else
%
%     if ~exist('sum_pq', 'var'),
%         sum_pq = (1/2)*(p_K + q_K);
%     end
%
%     exponent = mu'*(sum_pq \ mu);
%     h = 1 - (base)*(exp(-(1/8)*exponent));
% end

try
    chol_p = chol(p_K);
    chol_q = chol(q_K);
catch
    try
    p_K = chol(fix_pd_matrix(p_K));
    q_K = chol(fix_pd_matrix(q_K));
    catch
    chol_p = fix_pd_matrix(smoothing_matrix(p_K));
    chol_q = fix_pd_matrix(smoothing_matrix(q_K));        
    end
end


% compute hellinger distance using logdet

% compute P logdet
%chol_p = chol(p_K);
logdet_p = 2*sum(log(diag(chol_p)));

% compute Q logdet
%chol_q = chol(q_K);
logdet_q = 2*sum(log(diag(chol_q)));

% compute .5(P+Q) logdet
chol_pq  = chol(.5 * (p_K + q_K));
logdet_pq = 2*sum(log(diag(chol_pq)));

% compute log distance
log_base = .25 * (logdet_p + logdet_q) - .5 * (logdet_pq);
mu = p_mu - q_mu;
if all(mu < eps)
    log_h = log_base;
else
    log_h = log_base  - (1/8) * mu' * solve_chol(pq_chol, mu);
end

% exponentiate
h = 1 - exp(log_h);

end

function As = smoothing_matrix(A)

% First we compute the squared Frobenius norm of our matrix
nA = sum(sum(A.^2));
% Then we make this norm be meaningful for element wise comparison
nA = nA / numel(A);
% Finally, we smooth our matrix
As = A;
As( As.^2 < 1e-10*nA ) = 0;

end