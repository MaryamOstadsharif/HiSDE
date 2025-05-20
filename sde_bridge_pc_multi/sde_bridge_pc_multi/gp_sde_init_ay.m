function [init_param,cur_param] = gp_sde_int_ay(Ns,Ds)

%% max events
init_param.max_events = 200;

%% default Gamma parameters
init_param.gam_a0 = 50;
init_param.gam_b0 = 2;
init_param.gam_c0 = 5;
init_param.gam_d0 = 6;

%% set observation dimension
init_param.N = Ns;
init_param.D = Ds;

%% default Normal Paramater
init_param.norm_mu0 = zeros(init_param.D,1);
init_param.norm_k0  = 1;
init_param.norm_phi = 4 * eye(init_param.D,init_param.D);
init_param.norm_v0  = 3 + 2;

%% model number of samples
init_param.K  = 1000;
init_param.dt = 0.5;
init_param.sample = 5000; 

%% model extra parameters
init_param.x_var = 10*1e-2;
init_param.y_var = 1e-3;
diag_elements    = 0.1*abs(randn(init_param.N,1));
init_param.z_var = diag(diag_elements);
init_param.W = 0.15*randn(init_param.N,init_param.D+1); % one for dc

% mark info
cur_param.mark_mean = zeros(init_param.max_events,init_param.D);
cur_param.mark_var = zeros(init_param.max_events,init_param.D,init_param.D);
cur_param.gam_alpha= zeros(init_param.max_events,1);
cur_param.gam_beta = zeros(init_param.max_events,1);
cur_param.event_cnt = 0;

% scale adjust
init_param.scale_adj = 1;

% training sample %
init_param.train_sub = 5;


