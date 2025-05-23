function [alpha_map, theta_map] = Map_model_fit(x, a0, b0, c0, d0)
% MAP_GAMMA_SHAPE_SCALE estimates the MAP of Gamma(shape, scale)
% Inputs:
%   x  - observed data vector (positive values)
%   a0, b0 - Gamma prior parameters for alpha (shape): alpha ~ Gamma(a0, b0)
%   c0, d0 - Gamma prior parameters for theta (scale): theta ~ Gamma(c0, d0)
% Outputs:
%   alpha_map, theta_map - MAP estimates of shape and scale parameters

    % Ensure column vector
    x = x(:);
    N = length(x);
    S = sum(x);
    L = sum(log(x));

    % Negative log-posterior function to minimize
    function nlp = neg_log_post(params)
        alpha = params(1);
        theta = params(2);
        if alpha <= 0 || theta <= 0
            nlp = Inf;
            return;
        end
        log_post = ...
            -N * gammaln(alpha) ...
            - N * alpha * log(theta) ...
            + (alpha - 1) * L ...
            - S / theta ...
            + (a0 - 1) * log(alpha) - alpha / b0 ...
            + (c0 - 1) * log(theta) - theta / d0;
        nlp = -log_post;
    end

    % Initial guess
    init_alpha = 1;
    init_theta = mean(x);

    % Optimization options
    opts = optimset('fmincon');
    opts.Display = 'off';

    % Lower bounds to ensure positivity
    lb = [1e-5, 1e-5];

    % Perform optimization using fmincon
    [opt_params, ~] = fmincon(@neg_log_post, [init_alpha, init_theta], ...
        [], [], [], [], lb, [], [], opts);

    alpha_map = opt_params(1);
    theta_map = opt_params(2);
end
