function [init_param, cur_param] = HiSDE_init_md(Ns, Ds)
% HiSDE_init_md  Initialize parameters for the multi-dimensional HiSDE model
%
%   [init_param, cur_param] = HiSDE_init_md(Ns, Ds)
%
%   Sets up priors and fixed settings for inference on D-dimensional latent
%   state with N-dimensional observations, using a hierarchical SDE framework.
%
%   Inputs:
%     Ns – number of observed channels / sensors (observation dimensionality)
%     Ds – dimensionality of the latent state (excluding potential DC term)
%
%   Outputs:
%     init_param – struct of fixed hyperparameters and model settings:
%       • max_events – maximum latent events to consider
%       • gam_a0,b0,c0,d0 – Gamma‐hyperpriors for inter‐event waiting times
%       • N           – observation dimension (Ns)
%       • D           – latent state dimension (Ds)
%       • norm_mu0    – prior mean vector (D×1) for initial state and marks
%       • norm_k0     – prior strength on mean
%       • norm_phi    – prior scale matrix (D×D) for covariance (NIW prior)
%       • norm_v0     – degrees of freedom for NIW prior (≥ D+1)
%       • K           – number of time‐steps per trajectory
%       • dt          – time‐step size
%       • sample      – number of particles for SMC
%       • x_var       – process noise variance for latent increments
%       • y_var       – integrator noise variance
%       • z_var       – observation noise covariance (N×N diagonal)
%       • W           – measurement matrix (N×(D+1)), including DC term
%       • scale_adj   – scale adjustment factor on integrator update
%       • train_sub   – fraction or count of samples used for training (subsampling)
%
%     cur_param  – struct of current EM parameters, initialized to zeros:
%       • mark_mean  – [max_events×D] posterior mean of event marks (for each latent dim)
%       • mark_var   – [max_events×D×D] posterior covariance of event marks
%       • gam_alpha  – [max_events×1] Gamma shape parameters for waiting times
%       • gam_beta   – [max_events×1] Gamma scale parameters for waiting times
%       • event_cnt  – current inferred number of events (starts at 0)

%% 1) Latent‐event count and Gamma prior hyperparameters
init_param.max_events = 200;   % max number of events allowed
init_param.gam_a0     = 50;    % prior shape for inter‐event α
init_param.gam_b0     = 2;     % prior rate for α
init_param.gam_c0     = 5;     % prior shape for scale θ
init_param.gam_d0     = 6;     % prior rate for θ

%% 2) Observation and latent dimensions
init_param.N = Ns;   % number of observed channels
init_param.D = Ds;   % dimension of latent state

%% 3) NIW prior hyperparameters for latent state and marks
init_param.norm_mu0 = zeros(Ds,1);        % prior mean vector
init_param.norm_k0  = 1;                  % strength on prior mean
init_param.norm_phi = 4 * eye(Ds);        % scale matrix for covariance
init_param.norm_v0  = Ds + 2;             % degrees of freedom (≥ D+1)

%% 4) Simulation/inference settings
init_param.K      = 1000;    % time‐series length
init_param.dt     = 0.5;     % time‐step size
init_param.sample = 5000;    % number of particles per SMC run

%% 5) Noise variances and measurement mapping
init_param.x_var = 10 * 1e-2;            % latent increment noise variance
init_param.y_var = 1e-3;                % integrator noise variance
% Random diagonal observation noise covariance
diag_elements    = 0.1 * abs(randn(Ns,1));
init_param.z_var = diag(diag_elements);
% Measurement matrix maps [latent; DC] → observations
init_param.W     = 0.15 * randn(Ns, Ds+1);

%% 6) Initialize EM parameter struct
cur_param.mark_mean = zeros(init_param.max_events, Ds);
cur_param.mark_var  = zeros(init_param.max_events, Ds, Ds);
cur_param.gam_alpha = zeros(init_param.max_events, 1);
cur_param.gam_beta  = zeros(init_param.max_events, 1);
cur_param.event_cnt = 0;  % no events inferred initially

%% 7) Additional settings
init_param.scale_adj = 1;   % integrator scaling factor
init_param.train_sub = 5;   % subsample size or fraction for training
end
