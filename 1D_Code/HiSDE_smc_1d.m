function data = HiSDE_smc_1d(iter, init_param, Z, cur_param)
% HiSDE_smc_1d  Run 1D Sequential Monte Carlo (SMC) for the HiSDE model
%
%   data = HiSDE_smc_1d(iter, init_param, Z, cur_param)
%
%   Inputs:
%     iter       – iteration index (1 for initialization; >1 to use updated cur_param)
%     init_param – struct of fixed model settings and priors:
%                    • max_events – maximum number of latent events
%                    • gam_a0,b0,c0,d0 – hyperpriors for Gamma waiting‐time
%                    • norm_mu0,k0,alpha,beta – hyperpriors for mark distribution
%                    • dt       – time‐step size
%                    • sample   – number of particles
%                    • x_var,y_var,z_var – variances for process, integrator, observation noise
%                    • scale_adj – scaling factor for integrator update
%     Z          – 1×K (or K×1) vector of observed data over K time‐steps
%     cur_param  – struct of current EM parameters (from HiSDE_em_1d), containing:
%                    • event_cnt – number of inferred events so far
%                    • gam_alpha,gam_beta – per‐event Gamma params (shape, scale)
%                    • mark_mean,mark_var – per‐event Normal params for marks
%                    • x_mu0,y_mu0,x_var0,y_var0 – prior means/variances for state at k=1
%
%   Output:
%     data – struct with fields:
%              • Xs   – [sample×(K+1)] latent increments for each particle
%              • Ys   – [sample×(K+1)] integrated state for each particle
%              • Ms   – [sample×max_events] marks for each event
%              • Ts   – [sample×max_events] event (tau) times for each event
%              • Cntr – [sample×1] current event count index per particle

%% Unpack hyperparameters and model settings
scale_adj   = init_param.scale_adj;
max_events  = init_param.max_events;

% Gamma‐prior hyperparameters for waiting times
gam_a0 = init_param.gam_a0;
gam_b0 = init_param.gam_b0;
gam_c0 = init_param.gam_c0;
gam_d0 = init_param.gam_d0;
% Convert to initial Gamma(shape,scale)
gam_alpha = gam_a0 / gam_b0;
gam_beta  = gam_c0 / gam_d0;

% Normal‐prior hyperparameters for marks
norm_mu0    = init_param.norm_mu0;
norm_k0     = init_param.norm_k0;
norm_alpha  = init_param.norm_alpha;
norm_beta   = init_param.norm_beta;

% SMC parameters
K      = length(Z);         % number of time‐steps in this window
dt     = init_param.dt;     
sample = init_param.sample; % number of particles

% Noise variances
x_var = init_param.x_var;
y_var = init_param.y_var;
z_var = init_param.z_var;

%% Allocate arrays for particle trajectories and latent variables
% Xs, Ys store the state evolution; Ms, Ts record event marks and times
Xs   = zeros(sample, K+1);
Ys   = zeros(sample, K+1);
Ms   = zeros(sample, max_events);
Ts   = zeros(sample, max_events);
Cntr = ones(sample, 1);    % event counter per particle (start at 1)

%% Initialize particles at time k=0 (index 1 in MATLAB)
if iter == 1
    % At first iteration, sample from prior hyperparameters
    for s = 1:sample
        Xs(s,1) = norm_mu0 + randn * sqrt(norm_beta/((norm_alpha-1)*norm_k0));
        Ys(s,1) = norm_mu0 + randn * sqrt(norm_beta/((norm_alpha-1)*norm_k0));
        % No events have occurred yet
        Ms(s,1) = 0;
        Ts(s,1) = 0;
    end
else
    % For subsequent iterations, sample from cur_param posteriors
    for s = 1:sample
        Xs(s,1) = cur_param.x_mu0 + randn * sqrt(cur_param.x_var0);
        Ys(s,1) = cur_param.y_mu0 + randn * sqrt(cur_param.y_var0);
        Ms(s,1) = 0;
        Ts(s,1) = 0;
    end
end

%% Sequential Monte Carlo over each time‐step k = 1…K
for k = 1:K
    Ls = zeros(sample, 1);  % will store observation likelihood per particle
    
    for s = 1:sample
        % Current event index for this particle
        cur_cntr = Cntr(s);
        
        % If current time k*dt exceeds last event time, propose a new event
        if Ts(s, cur_cntr) < k * dt
            if cur_cntr <= cur_param.event_cnt
                % Use learned Gamma posterior for waiting time
                a = cur_param.gam_alpha(cur_cntr);
                b = cur_param.gam_beta(cur_cntr);
            else
                % Use prior Gamma(a0/b0, c0/d0)
                a = gam_alpha;
                b = gam_beta;
            end
            % Draw next wait time, enforce minimum dt*2
            next_wait = max(dt*2, gamrnd(a, b));
            % Record new event time and mark
            Ts(s, cur_cntr+1) = Ts(s, cur_cntr) + next_wait;
            if cur_cntr <= cur_param.event_cnt
                % use posterior mark mean/var
                Ms(s, cur_cntr+1) = cur_param.mark_mean(cur_cntr) + ...
                                   randn * sqrt(cur_param.mark_var(cur_cntr));
            else
                % draw from prior Normal
                Ms(s, cur_cntr+1) = norm_mu0 + ...
                                   randn * sqrt(norm_beta/((norm_alpha-1)*norm_k0));
            end
            % Increment this particle's event counter
            Cntr(s) = cur_cntr + 1;
        end
        
        % Compute state‐transition coefficients based on event window
        t2 = Ts(s, Cntr(s));
        t1 = Ts(s, Cntr(s)-1);
        b  = Ms(s, Cntr(s));
        ak = 1 - dt / max(0.5*dt, t2 - k*dt);
        bk = b * dt / max(0.5*dt, t2 - k*dt);
        sk = (t2 - k*dt) * (k*dt - t1) / (t2 - t1);
        
        % Propagate latent increment X and integrator Y
        Xs(s, k+1) = ak * Xs(s, k) + bk + sqrt(x_var * sk * dt) * randn;
        Ys(s, k+1) = Ys(s, k) + Xs(s, k) * dt * scale_adj + sqrt(y_var * dt) * randn;
        
        % Compute observation likelihood p(Z_k | Y_{k+1})
        Ls(s) = max(10*realmin, pdf('normal', Z(k), Ys(s, k+1), sqrt(z_var)));
    end
    
    %% Resample particles proportional to weights Ls
    Ls = Ls / sum(Ls);                   % normalize weights
    r  = rand(sample, 1);
    cum_weights = cumsum(Ls);
    idx = arrayfun(@(x) find(cum_weights >= x, 1), r);
    
    % Reassign particles by sampling with replacement
    Xs   = Xs(idx, :);
    Ys   = Ys(idx, :);
    Ms   = Ms(idx, :);
    Ts   = Ts(idx, :);
    Cntr = Cntr(idx);
end

%% Package outputs
data.Xs   = Xs;
data.Ys   = Ys;
data.Ms   = Ms;
data.Ts   = Ts;
data.Cntr = Cntr;
end
