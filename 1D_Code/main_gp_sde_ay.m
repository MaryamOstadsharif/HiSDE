%% Track inducing-point counts and average waiting time during SSM inference
clear all
close all

% Parameters
in_loop   = 2;
num_iters = 6;
in_start  = 1;

% Preallocate array for inducing-point counts
inducing_counts = zeros(1, num_iters);

% Initialize model parameters and data
[init_param, cur_param] = gp_sde_init_ay();
Z = model_z_ay(3, init_param);

% Inference loop: run SMC + EM and record event count each iteration
for iter = 1:num_iters
    for l = 1:in_loop
        % Sequential Monte Carlo step on growing data window
        data = gp_sde_smc_ay( ...
            in_start, ...                         % current start index
            init_param, ...                       % model parameters
            Z(1:min(100 + iter*100, init_param.K)), ...  % observed signal window
            cur_param ...                         % current EM parameters
        );

        % Expectation-Maximization update of model parameters
        cur_param = gp_sde_em_ay( ...
            Z, ...                                % full signal for EM
            data, ...                             % SMC output
            init_param, ...                       % initial parameters
            cur_param, ...                        % current parameters
            1 ...                                 % verbosity flag
        );

        % Advance time index
        in_start = in_start + 1;
    end

    % Record number of inducing points (event_cnt) after EM
    inducing_counts(iter) = cur_param.event_cnt;

    % Optional: pause for visualization speed control
    pause(0.1);
end

% Compute waiting times from final parameters
gam_alpha = cur_param.gam_alpha(1:cur_param.event_cnt);
gam_beta  = cur_param.gam_beta(1:cur_param.event_cnt);
mt        = gam_alpha .* gam_beta;            % vector of waiting times
avg_wait  = mean(mt);                          % average waiting time

% Display initial and final inducing-point counts and average wait
fprintf('Initial inducing-point count: %d\n', inducing_counts(1));
fprintf('Final inducing-point count:   %d\n', inducing_counts(end));
fprintf('Average waiting time:         %.3f time-steps\n', avg_wait);

% Generate final fit plots
gp_sde_gpfit_pr(Z, data, init_param, cur_param, mt);
