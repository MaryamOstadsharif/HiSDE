function [init_param, cur_param] = HiSDE_init_1d()
% HiSDE_init_1d  Initialize parameters for the 1D Hierarchical SDE model
%
%   [init_param, cur_param] = HiSDE_init_1d()
%
%   Outputs:
%     init_param – struct containing fixed model settings and priors:
%       • max_events  – maximum allowable number of latent events
%       • gam_a0,b0,c0,d0 – hyperparameters for Gamma prior on waiting times
%       • norm_mu0,k0,alpha,beta – hyperparameters for Normal–Inverse‐Gamma prior on marks
%       • K           – total length of the time series (number of steps)
%       • dt          – time-step size
%       • sample      – number of particles for SMC
%       • x_var, y_var, z_var – process, observation, and noise variances
%       • scale_adj   – optionally adjust scale of mark distribution
%
%     cur_param  – struct containing current EM parameters (initialized to zeros):
%       • mark_mean  – per-event posterior mean of mark m_i
%       • mark_var   – per-event posterior variance of mark m_i
%       • gam_alpha  – per-event shape parameters of Gamma posterior on waiting times
%       • gam_beta   – per-event scale parameters of Gamma posterior on waiting times
%       • event_cnt  – current estimate of the number of events
%
%   No inputs.

%% Maximum number of latent events to consider
init_param.max_events = 100;

%% Gamma‐prior hyperparameters for waiting‐time distribution
% We assume τ_i ∼ Gamma(shape=gam_alpha, scale=gam_beta)
% and place a conjugate hyperprior with parameters (a0,b0,c0,d0)
init_param.gam_a0 = 20;    % prior count‐shape parameter
init_param.gam_b0 = 2;     % prior count‐rate parameter
init_param.gam_c0 = 5;     % prior scale‐shape parameter
init_param.gam_d0 = 6;     % prior scale‐rate parameter

%% Normal‐Inverse‐Gamma prior hyperparameters for mark distribution
% Mark m_i ∼ Normal(μ,m_var) with conjugate N–IG prior
init_param.norm_mu0    = 0;   % prior mean of marks
init_param.norm_k0     = 1;   % prior relative precision on mean
init_param.norm_alpha  = 3;   % prior shape for variance
init_param.norm_beta   = 100; % prior scale for variance

%% Fixed model parameters for simulation and inference
init_param.K      = 500;     % length of the time series
init_param.dt     = 0.5;     % time‐step size
init_param.sample = 10000;   % number of particles for SMC

%% Process and observation noise variances
init_param.x_var = 1e-1;     % latent SDE process variance
init_param.y_var = 1e-4;     % measurement‐function variance
init_param.z_var = 0.1;      % observation noise variance

%% Initialize EM parameters (to be updated in HiSDE_em_1d)
% All vectors sized to max_events; initially zero until events are inferred
cur_param.mark_mean = zeros(init_param.max_events, 1);
cur_param.mark_var  = zeros(init_param.max_events, 1);
cur_param.gam_alpha = zeros(init_param.max_events, 1);
cur_param.gam_beta  = zeros(init_param.max_events, 1);
cur_param.event_cnt = 0;     % start with zero inferred events

%% Optional scale adjustment for marks
init_param.scale_adj = 1;
end
