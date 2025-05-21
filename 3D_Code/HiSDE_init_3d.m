function [init_param,cur_param] = gp_sde_int_ay()

%% max events
init_param.max_events = 100;

%% default Gamma parameters
init_param.gam_a0 = 30;
init_param.gam_b0 = 2;
init_param.gam_c0 = 5;
init_param.gam_d0 = 6;

%% set observation dimension
init_param.N = 10;

%% default Normal Paramater
init_param.norm_mu0 = zeros(3,1);
init_param.norm_k0  = 1;
init_param.norm_phi = 4 * eye(3,3);
init_param.norm_v0  = 3 + 2;

%% model number of samples
init_param.K  = 500;
init_param.dt = 0.5;
init_param.sample = 20000; 

%% model extra parameters
init_param.x_var = 10*(1e-1);
init_param.y_var = 1e-3;
diag_elements    = 0.1*abs(randn(init_param.N,1));
init_param.z_var = diag(diag_elements);
init_param.W = 0.1*randn(init_param.N,3);

% mark info
cur_param.mark_mean = zeros(init_param.max_events,3);
cur_param.mark_var = zeros(init_param.max_events,3,3);
cur_param.gam_alpha= zeros(init_param.max_events,1);
cur_param.gam_beta = zeros(init_param.max_events,1);
cur_param.event_cnt = 0;

% scale adjust
init_param.scale_adj = 1;


