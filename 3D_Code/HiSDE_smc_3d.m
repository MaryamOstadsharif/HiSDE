function data = HiSDE_smc_3d(iter, init_param, Z, cur_param)
% gp_sde_smc_ay  Run multivariate (3D) Sequential Monte Carlo for the GP-SDE model
%
%   data = gp_sde_smc_ay(iter, init_param, Z, cur_param)
%
%   Inputs:
%     iter       – iteration index (1 for initialization; >1 to use updated cur_param)
%     init_param – struct of fixed model settings and priors:
%                    • max_events – maximum latent events
%                    • gam_a0,b0,c0,d0 – Gamma hyperpriors for waiting times
%                    • norm_mu0,k0,phi,v0 – NIW prior for marks and initial state
%                    • K, dt, sample – time‐steps, time increment, and number of particles
%                    • x_var, y_var – process and integrator noise variances
%                    • scale_adj    – integrator scaling factor
%     Z          – [N×K] observed data matrix (N dims × time‐steps)
%     cur_param  – struct of current EM parameters:
%                    • W      – measurement matrix (N×3)
%                    • z_var  – observation noise covariance (N×N)
%                    • x_mu0,y_mu0,x_var0,y_var0 – priors for initial X and Y
%                    • mark_mean, mark_var            – per‐event posterior mark params
%                    • gam_alpha, gam_beta            – per‐event posterior waiting‐time params
%                    • event_cnt                      – inferred event count
%
%   Output:
%     data – struct containing SMC outputs:
%              • Xs   – [3×sample×(K+1)] latent increments per particle
%              • Ys   – [3×sample×(K+1)] integrator trajectories per particle
%              • Ms   – [3×sample×max_events] sampled marks per event
%              • Ts   – [sample×max_events] sampled event times per particle
%              • Cntr – [sample×1] current event count per particle

%% 1) Unpack hyperparameters and model settings
scale_adj = init_param.scale_adj;
max_events = init_param.max_events;

% Gamma‐prior hyperparameters for waiting times
gam_a0 = init_param.gam_a0;
gam_b0 = init_param.gam_b0;
gam_c0 = init_param.gam_c0;
gam_d0 = init_param.gam_d0;
% Convert to prior Gamma(shape, scale)
gam_alpha = gam_a0 / gam_b0;
gam_beta  = gam_c0 / gam_d0;

% NIW prior hyperparameters for marks and initial state
norm_mu0 = init_param.norm_mu0;
norm_k0  = init_param.norm_k0;
norm_phi = init_param.norm_phi;
norm_v0  = init_param.norm_v0;

% SMC settings
K      = length(Z);         % number of time‐steps in this window
dt     = init_param.dt;     
sample = init_param.sample; % number of particles

% Noise parameters
x_var = init_param.x_var;
y_var = init_param.y_var;

% Measurement parameters from cur_param
W     = cur_param.W;        % N×3 measurement matrix
z_var = cur_param.z_var;    % N×N observation noise covariance

%% 2) Allocate storage for particle trajectories and latent variables
Xs   = zeros(3, sample, K+1);
Ys   = zeros(3, sample, K+1);
Ms   = zeros(3, sample, max_events);
Ts   = zeros(sample, max_events);
Cntr = ones(sample, 1);      % event counter per particle (starts at 1)

%% 3) Initialize particles at time step 0
if iter == 1
    % Draw X(0), Y(0) from NIW prior
    for s = 1:sample
        Xs(:,s,1) = mvnrnd(norm_mu0, norm_phi)';
        Ys(:,s,1) = mvnrnd(norm_mu0, norm_phi)';
        Ms(:,s,1) = [0;0;0];    % no marks yet
        Ts(s,1)   = 0;          % no events yet
    end
else
    % Draw X(0), Y(0) from updated cur_param priors
    for s = 1:sample
        Xs(:,s,1) = mvnrnd(cur_param.x_mu0, cur_param.x_var0)';
        Ys(:,s,1) = mvnrnd(cur_param.y_mu0, cur_param.y_var0)';
        Ms(:,s,1) = [0;0;0];
        Ts(s,1)   = 0;
    end
end

%% 4) Sequential importance sampling and resampling over time steps
for k = 1:K
    Ls = zeros(sample,1);  % log‐likelihood weights for each particle
    for s = 1:sample
        cur_cntr = Cntr(s);
        % Propose new event if current time passes last event
        if Ts(s,cur_cntr) < k * dt
            if cur_cntr <= cur_param.event_cnt
                % use learned posterior Gamma
                a = cur_param.gam_alpha(cur_cntr);
                b = cur_param.gam_beta(cur_cntr);
            else
                % use prior Gamma
                a = gam_alpha;
                b = gam_beta;
            end
            next_wait = max(2*dt, gamrnd(a,b));
            Ts(s,cur_cntr+1) = Ts(s, cur_cntr) + next_wait;
            % Sample new mark
            if cur_cntr <= cur_param.event_cnt
                cov_m = squeeze(cur_param.mark_var(cur_cntr,:,:));
                mu_m  = cur_param.mark_mean(cur_cntr,:)';
            else
                cov_m = norm_phi;
                mu_m  = norm_mu0;
            end
            Ms(:,s,cur_cntr+1) = mvnrnd(mu_m, cov_m)';
            Cntr(s) = cur_cntr + 1;
        end
        % Propagate latent increment X and integrator Y for each of 3 dims
        t2 = Ts(s, Cntr(s));
        t1 = Ts(s, Cntr(s)-1);
        ak = 1 - dt / max(0.5*dt, t2 - k*dt);
        for d = 1:3
            b    = Ms(d,s,Cntr(s));
            bk   = b * dt / max(0.5*dt, t2 - k*dt);
            sk   = (t2 - k*dt)*(k*dt - t1)/(t2 - t1);
            Xs(d,s,k+1) = ak * Xs(d,s,k) + bk + sqrt(x_var*sk*dt)*randn;
            Ys(d,s,k+1) = Ys(d,s,k) + Xs(d,s,k)*dt*scale_adj + sqrt(y_var*dt)*randn;
        end
        % Compute observation likelihood p(Z(:,k) | Ys(:,s,k+1))
        Ls(s) = max(10*realmin, mvnpdf(Z(:,k), W*Ys(:,s,k+1), z_var));
    end
    % Normalize weights and resample indices
    Ls = Ls / sum(Ls);
    r  = rand(sample,1);
    cumw = cumsum(Ls);
    idx  = arrayfun(@(x) find(cumw >= x, 1), r);
    % Resample all variables
    Ts    = Ts(idx,:);
    Cntr  = Cntr(idx);
    Xs    = Xs(:,idx,:);
    Ys    = Ys(:,idx,:);
    Ms    = Ms(:,idx,:);
end

%% 5) Pack outputs into struct
data.Xs   = Xs;
data.Ys   = Ys;
data.Ms   = Ms;
data.Ts   = Ts;
data.Cntr = Cntr;
end
