clear all
close all

%% inside lopp
in_loop  = 2;
in_start = 1;

%% call the following functions first
% set model initial paramaters
[init_param,cur_param] = gp_sde_init_ay();
% create Z
[Z,Y]  = model_z_ay(init_param);
% for test
cur_param.W = init_param.W;
cur_param.z_var = init_param.z_var; 
% now, call smc
for iter = 1:20
    
    for l=1:in_loop
        data = gp_sde_smc_ay(in_start,init_param,Z(:,1:min(100+iter*100,init_param.K)),cur_param);
        % now call em
        cur_param = gp_sde_em_ay(Z,data,init_param,cur_param);
        % make the sde noise scaled
        in_start = in_start + 1; 
        cur_param.W = init_param.W;
        cur_param.z_var = init_param.z_var; 
        l
    end
end
Xs = squeeze( data.Xs );
raw_tau = data.Ts(1, 1:cur_param.event_cnt);  
tau_idx = round(raw_tau ./ init_param.dt) + 1; 
% suppose cur_param.tau contains your inducing‐point indices
gp_sde_gpfit_pr(Z, data.Ys, Y, Xs, init_param.dt, tau_idx);