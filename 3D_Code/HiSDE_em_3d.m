function new_param = HiSDE_em_3d(Z, data, init_param, cur_param)
% gp_sde_em_ay  Perform one EM update for the multivariate (3D) GP-SDE model
%
%   new_param = gp_sde_em_ay(Z, data, init_param, cur_param)
%
%   Inputs:
%     Z          – observed time series vector (length K)
%     data       – struct from SMC step containing:
%                    • Xs   – [sample×3×(K+1)] latent increments
%                    • Ys   – [sample×3×(K+1)] integrator trajectories
%                    • Ms   – [sample×3×max_events] marks per event per dimension
%                    • Ts   – [sample×max_events] event times (τ_i)
%                    • Cntr – [sample×1] number of events per particle
%     init_param – struct of fixed hyperparameters:
%                    • gam_a0,b0,c0,d0 – Gamma‐prior for waiting times
%                    • norm_mu0,k0,phi,v0 – Normal‐Inverse‐Wishart prior for initial state and marks
%                    • dt, sample      – time‐step size and number of particles
%     cur_param  – struct of current EM parameters:
%                    • x_mu0, x_var0   – prior mean & covariance for X(0)
%                    • y_mu0, y_var0   – prior mean & covariance for Y(0)
%                    • mark_mean, mark_var – per‐event posterior mean & covariance of marks
%                    • gam_alpha, gam_beta – per‐event posterior Gamma shape & scale
%
%   Output:
%     new_param  – struct containing updated EM parameters, same fields as cur_param:
%                    • x_mu0, x_var0
%                    • y_mu0, y_var0
%                    • mark_mean, mark_var
%                    • gam_alpha, gam_beta
%                    • event_cnt       – updated number of inferred events

%% take data out
Xs = data.Xs ;
Ys = data.Ys ;
Ms = data.Ms ;
Ts = data.Ts ;
Cntr = data.Cntr;

%% model parameters
K   = length(Z);%init_param.K;
dt  = init_param.dt;
sample = init_param.sample;

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
S = zeros(3,3);
for s=1:sample
    S = S + (xt(:,s)-mx) * (xt(:,s)-mx)'; 
end
P = (mx - norm_mu0) * (mx - norm_mu0)';
tmp_var = norm_phi + S + P * (norm_k0 * sample)/(norm_k0 + sample);
new_param.x_var0 = tmp_var / (norm_v0+sample+3+2);


yt = Ys(:,:,1);
my = mean(yt,2);
new_param.y_mu0 = (norm_k0 * norm_mu0 + sample* my)/(norm_k0 + sample);
S = zeros(3,3);
for s=1:sample
    S = S + (yt(:,s)-my) * (yt(:,s)-my)'; 
end
P = (my - norm_mu0) * (my - norm_mu0)';
tmp_var = norm_phi + S + P * (norm_k0 * sample)/(norm_k0 + sample);
new_param.y_var0 = tmp_var / (norm_v0+sample+3+2);

for cntr = 2:max(Cntr)

    ind = find(Cntr >= cntr);
    xt  = Ms(:,ind,cntr);
    mx  = mean(xt,2);
    S = zeros(3,3);
    t_sample = length(ind);
    for s=1:t_sample
         S = S + (xt(:,s)-mx) * (xt(:,s)-mx)'; 
    end
    P = (mx - norm_mu0) * (mx - norm_mu0)';
    tmp_var = norm_phi + S + P * (norm_k0 * t_sample)/(norm_k0 + t_sample);
    new_param.mark_mean(cntr-1,:)  = (norm_k0 * norm_mu0 + t_sample* mx)/(norm_k0 + t_sample);
    new_param.mark_var(cntr-1,:,:) = tmp_var/(norm_v0+t_sample+3+2);

    
    ts = Ts(ind,cntr) - Ts(ind,cntr-1);
    [alpha_map, theta_map] = map_gamma_shape_scale(ts, gam_a0, gam_b0, gam_c0, gam_d0);
    new_param.gam_alpha(cntr-1) = alpha_map;
    new_param.gam_beta(cntr-1) = theta_map;

    
end

new_param.event_cnt=max(Cntr)-1; 


end
    