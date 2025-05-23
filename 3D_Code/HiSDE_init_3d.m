function [init_param, cur_param] = HiSDE_init_3d()
% HiSDE_init_3d  Initialize parameters for the 3D Hierarchical SDE model
%
%   [init_param, cur_param] = HiSDE_init_3d()
%
%   Outputs:
%     init_param – struct containing fixed hyperparameters and model settings:
%       • max_events – maximum number of latent events to consider
%       • gam_a0,b0,c0,d0 – Gamma‐hyperpriors for inter‐event waiting times
%       • N           – observation dimensionality (number of sensors)
%       • norm_mu0    – prior mean vector (3×1) for initial state and marks
%       • norm_k0     – prior strength on mean
%       • norm_phi    – prior scale matrix (3×3) for covariance (NIW prior)
%       • norm_v0     – prior degrees of freedom for covariance (NIW prior)
%       • K           – number of time‐steps per trajectory
%       • dt          – time‐step size
%       • sample      – number of particles/trajectories to simulate
%       • x_var       – process noise variance for latent increments
%       • y_var       – integrator noise variance
%       • z_var       – observation noise covariance (N×N diagonal)
%       • W           – measurement matrix mapping 3D state to N‐dim observations
%       • scale_adj   – scale adjustment factor on integrator update
%
%     cur_param  – struct of current EM‐estimated parameters, initialized to zeros:
%       • mark_mean  – [max_events×3] posterior mean of event‐marks for each event
%       • mark_var   – [max_events×3×3] posterior covariance of event‐marks
%       • gam_alpha  – [max_events×1] Gamma shape parameters for waiting times
%       • gam_beta   – [max_events×1] Gamma scale parameters for waiting times
%       • event_cnt  – current inferred number of events (initially 0)

%% 1) Event‐count and Gamma prior hyperparameters
init_param.max_events = 100;
init_param.gam_a0     = 30;    % prior shape for waiting‐time α
init_param.gam_b0     = 2;     % prior rate for α
init_param.gam_c0     = 5;     % prior shape for scale θ
init_param.gam_d0     = 6;     % prior rate for θ

%% 2) Observation dimension and NIW hyperparameters for 3D state
init_param.N        = 10;              % number of observed channels
init_param.norm_mu0 = zeros(3,1);      % prior mean for X(0), Y(0), and marks
init_param.norm_k0  = 1;               % strength of prior mean
init_param.norm_phi = 4 * eye(3);      % prior scale matrix for covariance
init_param.norm_v0  = 3 + 2;           % degrees of freedom (≥ dimension+1)

%% 3) Simulation settings
init_param.K      = 500;      % time‐series length
init_param.dt     = 0.5;      % time‐step size
init_param.sample = 20000;    % number of particles per SMC run

%% 4) Noise variances and measurement mapping
init_param.x_var = 10 * 1e-1;            % latent increment noise variance
init_param.y_var = 1e-3;                % integrator noise variance
% Observation noise covariance: random diagonal entries
diag_elements    = 0.1 * abs(randn(init_param.N,1));
init_param.z_var = diag(diag_elements);
% Random measurement matrix (N×3) mapping 3D latent to observations
init_param.W     = 0.1 * randn(init_param.N, 3);

%% 5) Initialize EM parameter struct
cur_param.mark_mean  = zeros(init_param.max_events, 3);
cur_param.mark_var   = zeros(init_param.max_events, 3, 3);
cur_param.gam_alpha  = zeros(init_param.max_events, 1);
cur_param.gam_beta   = zeros(init_param.max_events, 1);
cur_param.event_cnt  = 0;            % no events inferred at start

%% 6) Scale adjustment for integrator update
init_param.scale_adj = 1;

end
