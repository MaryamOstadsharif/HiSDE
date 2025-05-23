clear all
close all

%% Parameters for the 3D inference loop
in_loop  = 2;    % Number of SMC+EM updates per iteration
in_start = 1;    % Starting index for the rolling observation window

%% 1) Initialize 3D HiSDE model parameters
% [init_param, cur_param] = HiSDE_init_3d()
%   Sets up hyperparameters and priors for the 3D SDE model:
%     init_param – struct with fields:
%                     • max_events, gam_a0/b0/c0/d0 (Gamma hyperpriors)
%                     • norm_mu0/k0/alpha/beta (Normal‐IG hyperpriors)
%                     • K, dt, sample (time‐steps, dt, #particles)
%                     • W (observation matrix for 3D → Z), z_var, etc.
%     cur_param  – struct of current EM parameters, initialized to zeros:
%                     • X/Y prior means & variances, mark_mean/var,
%                       gam_alpha/beta, event_cnt
[init_param, cur_param] = HiSDE_init_3d();

%% 2) Generate synthetic 3D data
% [Z, Y] = HiSDE_model_z_3d(init_param)
%   Simulates:
%     • Y – true 3D latent trajectories (3×K)
%     • Z – noisy scalar observations (1×K) via Z = W·Y + noise
%   Inputs: init_param struct (contains W, dt, z_var, K)
%   Outputs:
%     Z – observed time series (1×K)
%     Y – latent state array (3×K)
[Z, Y] = HiSDE_model_z_3d(init_param);

% For testing, copy fixed observation parameters into cur_param
cur_param.W     = init_param.W;     % measurement mapping
cur_param.z_var = init_param.z_var; % observation noise variance

%% 3) Inference loop: alternate SMC and EM to learn inducing points
for iter = 1:20
    for l = 1:in_loop
        % data = HiSDE_smc_3d(start_idx, init_param, Z_window, cur_param)
        %   Runs a 3D Sequential Monte Carlo filter over a sliding window of Z.
        %   Inputs:
        %     start_idx  – index at which to begin this window
        %     init_param – model hyperparameters and priors
        %     Z_window   – Z(:,1:min(100+iter*100, K)): growing subset of observations
        %     cur_param  – current EM parameters (for proposal distributions)
        %   Output:
        %     data       – struct containing:
        %                    • Xs   – [sample×3×(window+1)] particle increments
        %                    • Ys   – [sample×3×(window+1)] integrated trajectories
        %                    • Ts   – [sample×max_events] event times
        %                    • Ms   – [sample×max_events] event marks
        %                    • Cntr – [sample×1] event count per particle
        data = HiSDE_smc_3d( ...
            in_start, ...
            init_param, ...
            Z(:,1:min(100 + iter*100, init_param.K)), ...
            cur_param ...
        );

        % cur_param = HiSDE_em_3d(Z_full, data, init_param, cur_param)
        %   Performs one EM update of the 3D model parameters.
        %   Inputs:
        %     Z_full     – full observation sequence (1×K)
        %     data       – SMC output struct
        %     init_param – fixed hyperparameters
        %     cur_param  – previous EM parameters to be updated
        %   Output:
        %     cur_param  – updated struct with:
        %                    • updated gam_alpha/beta, mark_mean/var
        %                    • updated priors for X(0), Y(0)
        %                    • event_cnt (number of inducing points)
        cur_param = HiSDE_em_3d(Z, data, init_param, cur_param);

        % Advance the window start index by one time‐step
        in_start = in_start + 1;

        % Reset observation parameters to true values
        cur_param.W     = init_param.W;
        cur_param.z_var = init_param.z_var;

        % Display inner loop counter (for debugging/monitoring)
        disp(['  inner loop l = ' num2str(l)]);
    end
end

%% 4) Extract and reshape the final trajectories
% Xs is [sample×3×(K+1)]; squeeze to [sample×3×(K+1)] → [sample×(3×(K+1))]
Xs = squeeze(data.Xs);

%% 5) Compute inducing‐point indices for plotting
% raw_tau = data.Ts(1,1:event_cnt) gives the event times (continuous)
raw_tau = data.Ts(1, 1:cur_param.event_cnt);
% Convert to discrete indices in 1…K
tau_idx = round(raw_tau ./ init_param.dt) + 1;

%% 6) Visualization of inferred vs. true trajectories
% HiSDE_plot_3d(Z, Ys, Y_true, Xs, dt, tau_idx)
%   Plots:
%     • Observations Z
%     • Inferred integrator trajectories data.Ys (sample×3×(K+1))
%     • True latent Y
%     • Latent increments Xs
%     • Inducing points (tau_idx) as markers/arrows
%   Inputs:
%     Z        – observed 1×K
%     data.Ys  – particle‐averaged integrator [sample×3×K]
%     Y        – true latent 3×K
%     Xs       – latent increments [sample×3×(K+1)]
%     dt       – time‐step size
%     tau_idx  – 1×event_cnt vector of time indices for events
HiSDE_plot_3d(Z, data.Ys, Y, Xs, init_param.dt, tau_idx);
