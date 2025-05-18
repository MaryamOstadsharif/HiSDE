function [init_param,cur_param] = gp_sde_int_ay()

%% max events
init_param.max_events = 100;
%% default Gamma parameters
%init_param.gam_alpha = 250;
%init_param.gam_beta  = 0.1;

init_param.gam_a0 = 20;
init_param.gam_b0 = 2;
init_param.gam_c0 = 5;
init_param.gam_d0 = 6;

%% default Normal Paramater
init_param.norm_mu0 = 0;
init_param.norm_k0  = 1;
init_param.norm_alpha = 3;
init_param.norm_beta = 100;

%% model fixed parameters
init_param.K  = 500;
init_param.dt = 0.5;
init_param.sample = 10000; 

%% model extra parameters
init_param.x_var = 1e-1;
init_param.y_var = 1e-4;
init_param.z_var = 0.1;

cur_param.mark_mean = zeros(init_param.max_events,1);
cur_param.mark_var = zeros(init_param.max_events,1);
cur_param.gam_alpha= zeros(init_param.max_events,1);
cur_param.gam_beta = zeros(init_param.max_events,1);
cur_param.event_cnt = 0;

% scale adjust
init_param.scale_adj = 1;


