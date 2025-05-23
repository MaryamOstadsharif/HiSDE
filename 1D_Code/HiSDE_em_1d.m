function new_param = HiSDE_em_1d(Z, data, init_param, cur_param, mode)
% HiSDE_em_1d  Perform one Expectation–Maximization update for the 1D HiSDE model
%
%   new_param = HiSDE_em_1d(Z, data, init_param, cur_param, mode)
%
%   Inputs:
%     Z          – 1×K (or K×1) vector of observed data
%     data       – struct output from HiSDE_smc_1d with fields:
%                    • Xs   – [sample×(K+1)] latent increment trajectories
%                    • Ys   – [sample×(K+1)] integrator trajectories
%                    • Ms   – [sample×max_events] sampled marks
%                    • Ts   – [sample×max_events] sampled event times
%                    • Cntr – [sample×1] event count per particle
%     init_param – struct of fixed hyperparameters (as in HiSDE_init_1d):
%                    • gam_a0,b0,c0,d0 – Gamma‐prior hyperparameters
%                    • norm_mu0,k0,alpha,beta – Normal‐Inverse‐Gamma hyperparameters
%                    • dt, sample – time‐step and particle count
%     cur_param  – struct of current EM parameters with fields:
%                    • mark_mean, mark_var – current posterior estimates for marks
%                    • gam_alpha, gam_beta – current posterior estimates for waiting‐time
%                    • x_mu0, x_var0       – current initial state prior for X
%                    • y_mu0, y_var0       – current initial state prior for Y
%                    • event_cnt           – number of events inferred so far
%     mode       – integer flag controlling deduplication of repeated samples:
%                    • mode=1: use all samples
%                    • mode=2: remove duplicates before updating
%
%   Output:
%     new_param  – struct containing updated EM parameters:
%                    • x_mu0, x_var0       – updated prior for X(0)
%                    • y_mu0, y_var0       – updated prior for Y(0)
%                    • mark_mean, mark_var – updated posterior mean/var for each event’s mark
%                    • gam_alpha, gam_beta – updated posterior shape/scale for each waiting‐time
%                    • event_cnt           – updated number of events

%% Unpack SMC outputs
Xs   = data.Xs;    % latent increments
Ys   = data.Ys;    % integrator values
Ms   = data.Ms;    % sampled marks per event
Ts   = data.Ts;    % sampled event times
Cntr = data.Cntr;  % event counts per particle

%% Unpack model settings
K      = length(Z);        % number of time‐steps
dt     = init_param.dt;    
sample = init_param.sample; % number of particles

%% Unpack hyperparameters for the priors
gam_a0 = init_param.gam_a0;
gam_b0 = init_param.gam_b0;
gam_c0 = init_param.gam_c0;
gam_d0 = init_param.gam_d0;

norm_mu0    = init_param.norm_mu0;
norm_k0     = init_param.norm_k0;
norm_alpha  = init_param.norm_alpha;
norm_beta   = init_param.norm_beta;

%% Prepare new_param structure, initialize from cur_param
new_param.mark_mean  = cur_param.mark_mean;
new_param.mark_var   = cur_param.mark_var;
new_param.gam_alpha  = cur_param.gam_alpha;
new_param.gam_beta   = cur_param.gam_beta;

%% 1) Update initial‐state priors for X and Y at time k=0
xt = Xs(:,1);  % all particles’ initial X
if mode == 2
    [xt, idx_unique] = unique(xt);
    % optionally track how many unique remain
end
mx = mean(xt);
% Posterior mean: weighted combination of prior and sample mean
new_param.x_mu0 = (norm_k0*norm_mu0 + sample*mx) / (norm_k0 + sample);
% Posterior variance (N–IG update)
sx = sum((xt - new_param.x_mu0).^2);
new_param.x_var0 = (2*norm_beta + sx + norm_k0*(norm_mu0 - new_param.x_mu0)^2) ...
                   / (2*norm_alpha + sample + 3);

yt = Ys(:,1);  % all particles’ initial Y
if mode == 2
    yt = yt(idx_unique);
end
my = mean(yt);
new_param.y_mu0 = (norm_k0*norm_mu0 + sample*my) / (norm_k0 + sample);
sy = sum((yt - new_param.y_mu0).^2);
new_param.y_var0 = (2*norm_beta + sy + norm_k0*(norm_mu0 - new_param.y_mu0)^2) ...
                   / (2*norm_alpha + sample + 3);

%% 2) Update per‐event mark and waiting‐time parameters
% Loop over each possible event index (from 2 to max observed)
for cntr = 2:max(Cntr)
    % Select particles where at least cntr events occurred
    inds = find(Cntr >= cntr);
    
    %% a) Mark update (Normal‐Inverse‐Gamma posterior)
    ms = Ms(inds, cntr);  % mark samples for event #cntr
    if mode == 2
        [ms, idx_unique_m] = unique(ms);
    end
    m_mean = mean(ms);
    % Posterior mean of mark
    new_param.mark_mean(cntr-1) = (norm_k0*norm_mu0 + numel(ms)*m_mean) ...
                                  / (norm_k0 + numel(ms));
    % Posterior variance of mark
    s2 = sum((ms - new_param.mark_mean(cntr-1)).^2);
    new_param.mark_var(cntr-1) = (2*norm_beta + s2 + norm_k0*(norm_mu0 - new_param.mark_mean(cntr-1))^2) ...
                                 / (2*norm_alpha + numel(ms) + 3);

    %% b) Waiting‐time update (Gamma posterior via MAP)
    ts = Ts(inds, cntr) - Ts(inds, cntr-1);  % inter‐event intervals
    if mode == 2
        ts = ts(idx_unique_m);
    end
    % Compute MAP estimates for Gamma(shape, scale)
    [alpha_map, theta_map] = Map_model_fit(ts, gam_a0, gam_b0, gam_c0, gam_d0);
    new_param.gam_alpha(cntr-1) = alpha_map;
    new_param.gam_beta(cntr-1)  = theta_map;
end

%% 3) Update event count
new_param.event_cnt = max(Cntr) - 1;

end
