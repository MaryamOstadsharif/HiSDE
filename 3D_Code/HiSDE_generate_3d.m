function data = HiSDE_generate_3d(init_param, cur_param)
% gp_sde_generate_ay  Simulate multivariate (3D) SSM trajectories using learned GP-SDE parameters
%
%   data = gp_sde_generate_ay(init_param, cur_param)
%
%   Generates synthetic latent‐increment (X) and integrator (Y) trajectories,
%   along with event times (τ) and marks (m), under the 3-dimensional GP-SDE model
%   using the final EM-estimated parameters.
%
%   Inputs:
%     init_param – struct of fixed hyperparameters and model settings:
%                    • max_events – maximum latent events per trajectory
%                    • gam_a0,b0,c0,d0 – hyperpriors for Gamma waiting-time
%                    • norm_mu0,k0,phi,v0 – NIW prior for marks and initial state
%                    • K         – number of time-steps
%                    • dt        – time-step size
%                    • sample    – number of simulated trajectories
%                    • x_var,y_var – latent‐process and integrator noise variances
%                    • scale_adj – integrator scaling factor
%     cur_param  – struct of EM-estimated parameters:
%                    • event_cnt   – inferred number of events
%                    • gam_alpha,beta – per-event Gamma(shape,scale)
%                    • mark_mean,var  – per-event posterior mark mean & covariance
%                    • x_mu0,y_mu0   – prior mean of X(0), Y(0)
%                    • x_var0,y_var0 – prior covariance of X(0), Y(0)
%
%   Output:
%     data – struct containing simulated trajectories and event variables:
%              • Xs   – [3×sample×(K+1)] latent increments for each dimension
%              • Ys   – [3×sample×(K+1)] integrator trajectories
%              • Ms   – [3×sample×max_events] event marks (vector per event)
%              • Ts   – [sample×max_events] event times (τ_i) per trajectory
%              • Cntr – [sample×1] count of events in each trajectory

%% Unpack model settings and priors
scale_adj = init_param.scale_adj;     % scaling for integrator update
max_events = init_param.max_events;   % maximum latent events
gam_a0 = init_param.gam_a0;           % Gamma prior hyperparameters
gam_b0 = init_param.gam_b0;
gam_c0 = init_param.gam_c0;
gam_d0 = init_param.gam_d0;
% Convert hyperpriors to prior Gamma(shape,scale)
gam_alpha = gam_a0 / gam_b0;
gam_beta  = gam_c0 / gam_d0;

% NIW prior hyperparameters for 3-dimensional marks and states
norm_mu0 = init_param.norm_mu0;       
norm_k0  = init_param.norm_k0;
norm_phi = init_param.norm_phi;
norm_v0  = init_param.norm_v0;

% Fixed simulation parameters
K      = init_param.K;        % number of time steps
dt     = init_param.dt;       % time step size
sample = init_param.sample;   % number of trajectories to simulate

% Noise variances
x_var = init_param.x_var;      % latent‐increment noise variance
y_var = init_param.y_var;      % integrator noise variance

%% Preallocate arrays for outputs
% Dimensions: 3 (state dim) × sample × (K+1 time points)
Xs   = zeros(3, sample, K+1);      % latent increments
Ys   = zeros(3, sample, K+1);      % integrator trajectories
Ms   = zeros(3, sample, max_events); % marks per event
Ts   = zeros(sample, max_events);  % event times per trajectory
Cntr = ones(sample, 1);            % event counter per trajectory

%% Initialize trajectories at time k = 0
for s = 1:sample
    % Draw initial state X(0) ~ N(x_mu0, x_var0)
    Xs(:,s,1) = mvnrnd(cur_param.x_mu0, cur_param.x_var0)';
    % Draw initial integrator Y(0) ~ N(y_mu0, y_var0)
    Ys(:,s,1) = mvnrnd(cur_param.y_mu0, cur_param.y_var0)';
    % No events at start
    Ms(:,s,1) = [0; 0; 0];
    Ts(s,1)   = 0;
end

%% Simulate forward for k = 1…K
for k = 1:K
    for s = 1:sample
        cur_cntr = Cntr(s);  % current event index
        % If time exceeds last event, sample a new event
        if Ts(s,cur_cntr) < k * dt
            if cur_cntr <= cur_param.event_cnt
                % Use learned Gamma posterior for waiting time
                a = cur_param.gam_alpha(cur_cntr);
                b = cur_param.gam_beta(cur_cntr);
            else
                % Use prior Gamma for waiting time
                a = gam_alpha;
                b = gam_beta;
            end
            % Draw next wait time and record event time
            next_wait = max(2*dt, gamrnd(a, b));
            Ts(s,cur_cntr+1) = Ts(s,cur_cntr) + next_wait;
            % Sample mark from multivariate Normal
            if cur_cntr <= cur_param.event_cnt
                cov_m = squeeze(cur_param.mark_var(cur_cntr,:,:));
                mu_m  = cur_param.mark_mean(cur_cntr,:)';
            else
                cov_m = norm_phi;
                mu_m  = norm_mu0 * ones(3,1);
            end
            Ms(:,s,cur_cntr+1) = mvnrnd(mu_m, cov_m)';
            % Increment event counter
            Cntr(s) = cur_cntr + 1;
        end

        % Compute SDE coefficients between events
        t2 = Ts(s,Cntr(s));
        t1 = Ts(s,Cntr(s)-1);
        ak = 1 - dt / max(0.5*dt, t2 - k*dt);

        % Propagate X and Y in each dimension
        for d = 1:3
            b  = Ms(d,s,Cntr(s));
            bk = b * dt / max(0.5*dt, t2 - k*dt);
            sk = (t2 - k*dt) * (k*dt - t1) / (t2 - t1);

            % Latent increment update
            Xs(d,s,k+1) = ak * Xs(d,s,k) + bk + sqrt(x_var*sk*dt)*randn;
            % Integrator update
            Ys(d,s,k+1) = Ys(d,s,k) + Xs(d,s,k)*dt*scale_adj + sqrt(y_var*dt)*randn;
        end
    end
end

%% Package outputs into data struct
data.Xs   = Xs;
data.Ys   = Ys;
data.Ms   = Ms;
data.Ts   = Ts;
data.Cntr = Cntr;
end
