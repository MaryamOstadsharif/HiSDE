function new_param = HiSDE_em_md(Z,data,init_param,cur_param)
% HiSDE_em_md  Perform one EM update for the multi-dimensional HiSDE model
%
%   new_param = HiSDE_em_md(Z, data, init_param, cur_param)
%
%   This function performs the 'M' step of the EM algorithm for the
%   multi-dimensional hierarchical SDE (HiSDE) model. It updates:
%     • Initial state priors (x_mu0, x_var0 and y_mu0, y_var0)
%     • Per-event mark distributions (mark_mean, mark_var)
%     • Per-event waiting-time Gamma parameters (gam_alpha, gam_beta)
%     • Observation mapping W (using Poisson likelihood)
%
%   Inputs:
%     Z          – [N×K] observed data matrix (N observed channels × K time steps)
%     data       – struct with SMC outputs:
%                    • Xs   – [D×sample×(K+1)] latent increments per particle
%                    • Ys   – [D×sample×(K+1)] integrator trajectories per particle
%                    • Ms   – [D×sample×max_events] marks per event per particle
%                    • Ts   – [sample×max_events] event times per particle
%                    • Cntr – [sample×1] number of events per particle
%     init_param – struct of fixed hyperparameters and settings:
%                    • gam_a0,b0,c0,d0 – Gamma prior hyperparameters
%                    • norm_mu0,k0,phi,v0 – NIW prior hyperparameters
%                    • dt, sample     – time step and number of particles
%                    • train_sub      – percentage of particles to use for W update
%     cur_param  – struct of current EM parameters:
%                    • x_mu0, x_var0 – priors for X(0)
%                    • y_mu0, y_var0 – priors for Y(0)
%                    • mark_mean, mark_var – per-event mark parameters
%                    • gam_alpha, gam_beta – per-event waiting-time parameters
%                    • W               – observation mapping matrix (N×(D+1))
%                    • event_cnt       – number of inferred events
%
%   Output:
%     new_param  – struct containing updated parameters, including fields:
%                    • x_mu0, x_var0
%                    • y_mu0, y_var0
%                    • mark_mean, mark_var
%                    • gam_alpha, gam_beta
%                    • W (updated observation mapping)
%                    • event_cnt
%% take data out
Xs = data.Xs ;
Ys = data.Ys ;
Ms = data.Ms ;
Ts = data.Ts ;
Cntr = data.Cntr;

%% model parameters
K   = size(Z,2);
dt  = init_param.dt;
sample = init_param.sample;

%% load D and N
N  = init_param.N;
D  = init_param.D;

%% default Gamma parameters
gam_a0 = init_param.gam_a0;
gam_b0 = init_param.gam_b0;
gam_c0 = init_param.gam_c0;
gam_d0 = init_param.gam_d0;

%% default Normal Paramater
norm_mu0 = init_param.norm_mu0;
norm_k0  = init_param.norm_k0 ;
norm_phi = init_param.norm_phi;
norm_v0 = init_param.norm_v0;

%% create array
new_param.mark_mean =  cur_param.mark_mean;
new_param.mark_var = cur_param.mark_var;
new_param.gam_alpha= cur_param.gam_alpha;
new_param.gam_beta = cur_param.gam_beta;


%% new param
xt = Xs(:,:,1);
mx = mean(xt,2);
new_param.x_mu0 = (norm_k0 * norm_mu0 + sample* mx)/(norm_k0 + sample);
S = zeros(D,D);
for s=1:sample
    S = S + (xt(:,s)-mx) * (xt(:,s)-mx)'; 
end
P = (mx - norm_mu0) * (mx - norm_mu0)';
tmp_var = norm_phi + S + P * (norm_k0 * sample)/(norm_k0 + sample);
new_param.x_var0 = tmp_var / (norm_v0+sample+D+2);


yt = Ys(:,:,1);
my = mean(yt,2);
new_param.y_mu0 = (norm_k0 * norm_mu0 + sample* my)/(norm_k0 + sample);
S = zeros(D,D);
for s=1:sample
    S = S + (yt(:,s)-my) * (yt(:,s)-my)'; 
end
P = (my - norm_mu0) * (my - norm_mu0)';
tmp_var = norm_phi + S + P * (norm_k0 * sample)/(norm_k0 + sample);
new_param.y_var0 = tmp_var / (norm_v0+sample+D+2);

for cntr = 2:max(Cntr)

    ind = find(Cntr >= cntr);
    xt  = Ms(:,ind,cntr);
    mx  = mean(xt,2);
    S = zeros(D,D);
    t_sample = length(ind);
    for s=1:t_sample
         S = S + (xt(:,s)-mx) * (xt(:,s)-mx)'; 
    end
    P = (mx - norm_mu0) * (mx - norm_mu0)';
    tmp_var = norm_phi + S + P * (norm_k0 * t_sample)/(norm_k0 + t_sample);
    new_param.mark_mean(cntr-1,:)  = (norm_k0 * norm_mu0 + t_sample* mx)/(norm_k0 + t_sample);
    new_param.mark_var(cntr-1,:,:) = tmp_var/(norm_v0+t_sample+D+2);

    
    ts = Ts(ind,cntr) - Ts(ind,cntr-1);
    [alpha_map, theta_map] = map_gamma_shape_scale(ts, gam_a0, gam_b0, gam_c0, gam_d0);
    new_param.gam_alpha(cntr-1) = alpha_map;
    new_param.gam_beta(cntr-1) = theta_map;
end

%% sub-sample
% Calculate number of elements to sample
num_to_sample = round((init_param.train_sub/100) * sample);
% Generate sequence
sequence = 1:sample;
% Randomly sample without replacement
sampled = randsample(sequence, num_to_sample);

% optimization
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton','Display','off');
% predictor vector
Yp = ones(num_to_sample * K,D+1);  
for i = 1:num_to_sample
    Yp((i-1)*K + 1 : i*K,1:D) = squeeze(Ys(:, sampled(i), 2:K+1))';
end
% now, upda(te W and also z_var
W_pre  = cur_param.W;   
W_pos  = W_pre;
for ns = 1:N
    % obs of one neuron
    Zp  = repmat(Z(ns,:)',num_to_sample,1);
    % adjst parameter for that neuron
    w0  = W_pre(ns,:)';
    % Negative log-likelihood function
    neg_log_likelihood = @(w) compute_neg_log_likelihood(w,Yp,Zp,dt);
    [w_opt, fval]      = fminunc(neg_log_likelihood, w0, options);
    W_pos(ns,:) = w_opt; 
end

new_param.W = W_pos;

new_param.event_cnt=max(Cntr)-1; 

% ---- Helper Function ----
function nll = compute_neg_log_likelihood(w, x, s, dt)
    % calculat x*w
    eta = x * w;
    % calulate rate
    lambda = exp(eta)*dt;
    % calculate negative of log of likelihood
    p1 = sum(s.*eta);    % count log of lamba for all active points
    p2 = - sum(lambda);  % log(exp(-lambda for all times)) for all points
    nll = -(p1 + p2);  % Negative log-likelihood
end

end
    