function data = HiSDE_generate_1d(init_param, cur_param)
% HiSDE_generate_1d  Simulate 1D SSM trajectories using learned HiSDE parameters
%
%   data = HiSDE_generate_1d(init_param, cur_param)
%
%   Generates synthetic latent‐increment (X) and integrator (Y) trajectories,
%   together with event times (τ) and marks (m), under the Hierarchical SDE model
%   using the final EM‐estimated parameters.
%
%   Inputs:
%     init_param – struct of fixed model settings and priors:
%                    • max_events – maximum number of latent events
%                    • gam_a0,b0,c0,d0 – hyperpriors for Gamma waiting‐times
%                    • norm_mu0,k0,alpha,beta – hyperpriors for mark distribution
%                    • K           – number of time‐steps
%                    • dt          – time‐step size
%                    • sample      – number of trajectories to simulate
%                    • x_var,y_var,z_var – process, integrator, observation variances
%                    • scale_adj   – scale factor for integrator update
%     cur_param  – struct of EM‐estimated parameters:
%                    • event_cnt   – number of inferred events
%                    • gam_alpha   – per‐event Gamma shape parameters
%                    • gam_beta    – per‐event Gamma scale parameters
%                    • mark_mean   – per‐event posterior mean of marks
%                    • mark_var    – per‐event posterior variance of marks
%                    • x_mu0,y_mu0 – initial prior means for X and Y
%                    • x_var0,y_var0 – initial prior variances for X and Y
%
%   Output:
%     data – struct with simulated trajectories and event variables:
%              • Xs   – [sample×(K+1)] simulated latent increments
%              • Ys   – [sample×(K+1)] simulated integrator values
%              • Ms   – [sample×max_events] simulated marks m_i
%              • Ts   – [sample×max_events] event times τ_i
%              • Cntr – [sample×1] number of events per trajectory

%% Unpack model settings and priors
scale_adj = init_param.scale_adj;
max_events = init_param.max_events;
gam_a0 = init_param.gam_a0;
gam_b0 = init_param.gam_b0;
gam_c0 = init_param.gam_c0;
gam_d0 = init_param.gam_d0;

% Compute prior Gamma(shape,scale) from hyperpriors
gam_alpha = gam_a0 / gam_b0;
gam_beta  = gam_c0 / gam_d0;

norm_mu0   = init_param.norm_mu0;
norm_k0    = init_param.norm_k0;
norm_alpha = init_param.norm_alpha;
norm_beta  = init_param.norm_beta;

K      = init_param.K;      % total time‐steps
dt     = init_param.dt;     % time‐step size
sample = init_param.sample; % number of trajectories

x_var = init_param.x_var;   % process variance
y_var = init_param.y_var;   % integrator noise variance
z_var = init_param.z_var;   % (not used here but from model)

%% Preallocate arrays for simulation outputs
% Xs,Ys store the dynamic state; Ms,Ts record event marks and times
Xs   = zeros(sample, K+1);
Ys   = zeros(sample, K+1);
Ms   = zeros(sample, max_events);
Ts   = zeros(sample, max_events);
Cntr = ones(sample, 1);  % event counter per trajectory

%% Initialize each trajectory at time k=0
for s = 1:sample
    % Draw initial X from its prior N(x_mu0, x_var0)
    Xs(s,1) = cur_param.x_mu0 + randn * sqrt(cur_param.x_var0);
    % Draw initial Y from its prior N(y_mu0, y_var0)
    Ys(s,1) = cur_param.y_mu0 + randn * sqrt(cur_param.y_var0);
    % No events at start
    Ms(s,1) = 0;
    Ts(s,1) = 0;
end

%% Simulate forward in time for k = 1…K
for k = 1:K
    for s = 1:sample
        % Current event index for trajectory s
        cur_cntr = Cntr(s);

        % If time k*dt passes the last event, sample a new event
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
            % Draw next waiting time, enforce a minimum of 2·dt
            next_wait = max(2*dt, gamrnd(a, b));
            % Schedule new event time (quantized and offset by 0.5·dt)
            Ts(s,cur_cntr+1) = floor(Ts(s,cur_cntr) + next_wait) + 0.5*dt;
            % Sample corresponding mark
            if cur_cntr <= cur_param.event_cnt
                Ms(s,cur_cntr+1) = cur_param.mark_mean(cur_cntr) + ...
                                   randn * sqrt(cur_param.mark_var(cur_cntr));
            else
                Ms(s,cur_cntr+1) = norm_mu0 + ...
                                   randn * sqrt(norm_beta/((norm_alpha-1)*norm_k0));
            end
            % Increment the event counter
            Cntr(s) = cur_cntr + 1;
        end

        % Compute SDE coefficients between events
        t2 = Ts(s,Cntr(s));
        t1 = Ts(s,Cntr(s)-1);
        b  = Ms(s,Cntr(s));
        ak = 1 - dt / max(0.5*dt, t2 - k*dt);
        bk = b * dt / max(0.5*dt, t2 - k*dt);
        sk = (t2 - k*dt)*(k*dt - t1)/(t2 - t1);

        % Propagate latent increment X
        Xs(s,k+1) = ak * Xs(s,k) + bk + sqrt(x_var * sk * dt) * randn;
        % Update integrator Y
        Ys(s,k+1) = Ys(s,k) + Xs(s,k) * dt * scale_adj + sqrt(y_var * dt) * randn;
    end
end

%% Package outputs into data struct
data.Xs   = Xs;
data.Ys   = Ys;
data.Ms   = Ms;
data.Ts   = Ts;
data.Cntr = Cntr;
end
