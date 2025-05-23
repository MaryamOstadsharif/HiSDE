function data = HiSDE_smc_md(iter, init_param, Z, cur_param)
% HiSDE_smc_md  Run Sequential Monte Carlo for multi-dimensional HiSDE model
%
%   data = HiSDE_smc_md(iter, init_param, Z, cur_param)
%
%   Performs one pass of the particle filter over a sliding window of
%   observations Z for the D-dimensional latent SDE model, sampling events,
%   latent increments, and integrator trajectories, then resampling by weight.
%
%   Inputs:
%     iter       – iteration index (1 for initialization; >1 to use updated cur_param)
%     init_param – struct of fixed settings and priors:
%                    • max_events  – maximum latent events per trajectory
%                    • gam_a0,b0,c0,d0 – Gamma‐prior hyperparameters for waiting times
%                    • norm_mu0,k0,phi,v0 – NIW prior hyperparameters for marks & state
%                    • K, dt, sample – time‐steps, time increment, and number of particles
%                    • x_var, y_var – noise variances for latent increments and integrator
%                    • scale_adj    – integrator scaling factor
%     Z          – [N×K] observed data matrix (N channels × time‐steps)
%     cur_param  – struct of current EM parameters:
%                    • W(:,1:D)   – measurement matrix mapping latent to log‐rate
%                    • W(:,D+1)   – bias (DC) term for rate
%                    • z_var      – not used here (Poisson obs model)
%                    • x_mu0, x_var0, y_mu0, y_var0 – priors for initial state
%                    • mark_mean, mark_var            – per‐event posterior mark params
%                    • gam_alpha, gam_beta            – per‐event posterior waiting‐time params
%                    • event_cnt                      – inferred number of events so far
%
%   Output:
%     data – struct containing:
%              • Xs   – [D×sample×(K+1)] latent increments per particle
%              • Ys   – [D×sample×(K+1)] integrator trajectories per particle
%              • Ms   – [D×sample×max_events] sampled marks per event
%              • Ts   – [sample×max_events] sampled event times per particle
%              • Cntr – [sample×1] event count index per particle

%% Unpack model settings and priors
scale_adj = init_param.scale_adj;    % integrator scale factor
max_events = init_param.max_events;  % max events allowed

% Gamma‐prior hyperparameters
gam_a0 = init_param.gam_a0;  
gam_b0 = init_param.gam_b0;
gam_c0 = init_param.gam_c0;
gam_d0 = init_param.gam_d0;
gam_alpha = gam_a0 / gam_b0;   % prior shape
gam_beta  = gam_c0 / gam_d0;   % prior scale

% NIW prior hyperparameters
norm_mu0 = init_param.norm_mu0;  % D×1 prior mean
norm_k0  = init_param.norm_k0;   % strength on mean
norm_phi = init_param.norm_phi;  % D×D scale matrix
norm_v0  = init_param.norm_v0;   % degrees of freedom

% SMC parameters
[~, K] = size(Z);         % number of time‐steps in this window
dt      = init_param.dt;  
sample  = init_param.sample;  % number of particles
D       = init_param.D;       % latent state dimension

% Noise variances (not used directly for Poisson obs)
x_var = init_param.x_var;  
y_var = init_param.y_var;

% Extract observation‐model parameters
W  = cur_param.W(:,1:D);    % N×D mapping from latent Y to log‐rate
W0 = cur_param.W(:,D+1);    % N×1 bias term
% Poisson model does not use z_var directly

%% Allocate storage for particles and latent variables
Xs   = zeros(D, sample, K+1);       % latent increments
Ys   = zeros(D, sample, K+1);       % integrator trajectories
Ms   = zeros(D, sample, max_events);% event marks
Ts   = zeros(sample, max_events);   % event times
Cntr = ones(sample, 1);             % event count per particle

%% Initialize particles at time k = 0
if iter == 1
    % Sample initial X, Y from NIW prior
    for s = 1:sample
        Xs(:,s,1) = mvnrnd(norm_mu0, norm_phi)';
        Ys(:,s,1) = mvnrnd(norm_mu0, norm_phi)';
        Ms(:,s,1) = zeros(D,1);  
        Ts(s,1)   = 0;
    end
else
    % Sample from updated priors in cur_param
    for s = 1:sample
        Xs(:,s,1) = mvnrnd(cur_param.x_mu0, cur_param.x_var0)';
        Ys(:,s,1) = mvnrnd(cur_param.y_mu0, cur_param.y_var0)';
        Ms(:,s,1) = zeros(D,1);
        Ts(s,1)   = 0;
    end
end

%% Sequential importance sampling and resampling
for k = 1:K
    Ls = zeros(sample,1);  % likelihood weights
    for s = 1:sample
        % Current event index
        cur_cntr = Cntr(s);
        % If time exceeds last event, propose new event
        if Ts(s,cur_cntr) < k * dt
            if cur_cntr <= cur_param.event_cnt
                a = cur_param.gam_alpha(cur_cntr);
                b = cur_param.gam_beta(cur_cntr);
            else
                a = gam_alpha; b = gam_beta;
            end
            next_wait = max(2*dt, gamrnd(a,b));
            Ts(s,cur_cntr+1) = Ts(s,cur_cntr) + next_wait;
            % Sample new mark m ~ N(mark_mean, mark_var)
            mu_m  = cur_param.mark_mean(cur_cntr,:)';
            cov_m = squeeze(cur_param.mark_var(cur_cntr,:,:));
            Ms(:,s,cur_cntr+1) = mvnrnd(mu_m, cov_m)';
            Cntr(s) = cur_cntr + 1;
        end

        % Compute SDE coefficients between events
        t2 = Ts(s,Cntr(s));
        t1 = Ts(s,Cntr(s)-1);
        ak = 1 - dt / max(0.5*dt, t2 - k*dt);

        % Propagate latent X and integrator Y in each dimension
        for d = 1:D
            b   = Ms(d,s,Cntr(s));
            bk  = b * dt / max(0.5*dt, t2 - k*dt);
            sk  = (t2 - k*dt)*(k*dt - t1)/(t2 - t1);
            Xs(d,s,k+1) = ak * Xs(d,s,k) + bk + sqrt(x_var*sk*dt)*randn;
            Ys(d,s,k+1) = Ys(d,s,k) + Xs(d,s,k)*dt*scale_adj + sqrt(y_var*dt)*randn;
        end

        % Compute Poisson observation likelihood:
        % λ = exp(W0 + W*y), loglik = obs' * log(λ) - dt * sum(λ)
        y      = Ys(:,s,k+1);
        lambda = exp(W0 + W * y);
        obs    = Z(:,k);
        loglik = obs' * log(lambda) - dt * sum(lambda);
        Ls(s)  = max(10*realmin, exp(loglik));  % avoid underflow
    end

    % Normalize weights and resample particles
    Ls = Ls / sum(Ls);
    r  = rand(sample,1);
    cumw = cumsum(Ls);
    idx  = arrayfun(@(x) find(cumw >= x, 1), r);

    % Resample all particle arrays
    Ts    = Ts(idx,:);
    Cntr  = Cntr(idx);
    Xs    = Xs(:,idx,:);
    Ys    = Ys(:,idx,:);
    Ms    = Ms(:,idx,:);
end

%% Package outputs
data.Xs   = Xs;
data.Ys   = Ys;
data.Ms   = Ms;
data.Ts   = Ts;
data.Cntr = Cntr;
end
