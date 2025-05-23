%% Track inducing-point counts and average waiting time during SSM inference
clear all
close all

% Parameters controlling the inference loop
in_loop   = 2;   % Number of SMC+EM updates per iteration
num_iters = 6;   % Total number of EM iterations
in_start  = 1;   % Starting index for the data window

% Preallocate array to record how many inducing points (events) are found
inducing_counts = zeros(1, num_iters);

%% Initialization of model parameters and synthetic data
% [init_param, cur_param] = HiSDE_init_1d()
%   Initializes the HiSDE (Hierarchical SDE) model for 1D time series.
%   Outputs:
%     init_param – struct of fixed model settings (e.g., time-step dt, K, priors)
%     cur_param  – struct of current EM parameters (e.g., event_cnt, gam_alpha, gam_beta)
[init_param, cur_param] = HiSDE_init_1d();

% Z = HiSDE_model_z_1d(n, init_param)
%   Generates a synthetic 1D observed signal Z using the HiSDE generative model.
%   Inputs:
%     n          – random seed or scenario index (here: 3)
%     init_param – struct returned by HiSDE_init_1d
%   Output:
%     Z          – 1×K vector of simulated observations
Z = HiSDE_model_z_1d(3, init_param);

%% Inference: alternate SMC and EM to learn event times and parameters
for iter = 1:num_iters
    for l = 1:in_loop
        % data = HiSDE_smc_1d(start_idx, init_param, Z_window, cur_param)
        %   Runs the Sequential Monte Carlo (particle) filter over a growing window of Z.
        %   Inputs:
        %     start_idx  – index where this window begins
        %     init_param – fixed model settings
        %     Z_window   – observed data from indices 1 to min(100+iter*100, K)
        %     cur_param  – current estimate of EM parameters
        %   Output:
        %     data       – struct containing SMC outputs, including:
        %                    Ts (event times), Ms (marks), weights, and Y trajectories
        data = HiSDE_smc_1d( ...
            in_start, ...
            init_param, ...
            Z(1:min(100 + iter*100, init_param.K)), ...
            cur_param ...
        );

        % cur_param = HiSDE_em_1d(Z_full, data, init_param, cur_param, verbose)
        %   Performs one Expectation-Maximization update of the model parameters.
        %   Inputs:
        %     Z_full     – the full observed signal (for complete-data likelihood)
        %     data       – SMC output struct
        %     init_param – fixed model settings
        %     cur_param  – current EM parameters (to be updated)
        %     verbose    – flag (1 to print progress)
        %   Output:
        %     cur_param  – updated EM parameters including new event count,
        %                  updated gam_alpha and gam_beta arrays
        cur_param = HiSDE_em_1d( ...
            Z, ...
            data, ...
            init_param, ...
            cur_param, ...
            1 ...
        );

        % Move the window start forward by one time-step
        in_start = in_start + 1;
    end

    % Record how many inducing points (events) the EM estimated this iteration
    inducing_counts(iter) = cur_param.event_cnt;

    % Optional pause to slow down visualization during the loop
    pause(0.1);
end

%% Compute final waiting times from the learned gamma parameters
% gam_alpha and gam_beta store shape and scale for each event’s waiting time
gam_alpha = cur_param.gam_alpha(1:cur_param.event_cnt);
gam_beta  = cur_param.gam_beta(1:cur_param.event_cnt);

% mt = gam_alpha .* gam_beta computes the mean waiting time for each event
mt        = gam_alpha .* gam_beta;            

% Average waiting time across all inferred events
avg_wait  = mean(mt);                          

%% Display summary of results
fprintf('Initial inducing-point count: %d\n', inducing_counts(1));
fprintf('Final inducing-point count:   %d\n', inducing_counts(end));
fprintf('Average waiting time:         %.3f time-steps\n', avg_wait);

%% Generate comparison plots of the final fit
% GP_fit_compare(Z, data, init_param, cur_param, mt)
%   Creates figures comparing:
%     • the original signal Z
%     • the inferred latent trajectories (Y)
%     • the inducing points (event times)
%     • the Gaussian Process fit to inducing points
%   Inputs:
%     Z          – observed signal
%     data       – SMC struct from the last call (contains Y, Ts, etc.)
%     init_param – fixed model parameters
%     cur_param  – final EM parameters
%     mt         – vector of mean waiting times (for annotations)
GP_fit_compare(Z, data, init_param, cur_param, mt);
