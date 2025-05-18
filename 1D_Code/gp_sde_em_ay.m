function new_param = gp_sde_em_ay(Z,data,init_param,cur_param,mode)

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
norm_mu0   = init_param.norm_mu0;
norm_k0    = init_param.norm_k0;
norm_alpha = init_param.norm_alpha;
norm_beta  = init_param.norm_beta;

%% create array
new_param.mark_mean =  cur_param.mark_mean;
new_param.mark_var = cur_param.mark_var;
new_param.gam_alpha= cur_param.gam_alpha;
new_param.gam_beta = cur_param.gam_beta;


%% new param
xt = Xs(:,1);
if mode == 2
    [xt,nrpt_ind] = unique(xt);
    keep_ms = length(xt);
end
mx = mean(xt);
new_param.x_mu0 = (norm_k0 * norm_mu0 + sample* mx)/(norm_k0 + sample);
sx = sum((xt-new_param.x_mu0).^2);
new_param.x_var0 = (2*norm_beta +  sx +  norm_k0*(norm_mu0-new_param.x_mu0)^2)/(2*norm_alpha+sample+3);


yt = Ys(:,1);
if mode == 2
    yt = yt(nrpt_ind);
end
my = mean(yt);
new_param.y_mu0 = (norm_k0 * norm_mu0 + sample* my)/(norm_k0 + sample);
sy = sum((yt-new_param.y_mu0).^2);
new_param.y_var0 = (2*norm_beta +  sy +  norm_k0*(norm_mu0-new_param.y_mu0)^2)/(2*norm_alpha+sample+3);


for cntr = 2:max(Cntr)

    ind = find(Cntr >= cntr);
    ms  = Ms(ind,cntr);
    if mode == 2
     [ms,nrpt_ind] = unique(ms);
     keep_ms = [keep_ms length(ms)];
    end

    temp_mean = mean(ms);
    update_mean = (norm_k0 * norm_mu0 + length(ms)* temp_mean)/(norm_k0 + length(ms));
    new_param.mark_mean(cntr-1) = update_mean;
    temp_s = sum((ms-update_mean).^2);
    new_param.mark_var(cntr-1) = (2*norm_beta +  temp_s +  norm_k0*(norm_mu0-update_mean)^2)/(2*norm_alpha+length(ms)+3);

    ts = Ts(ind,cntr) - Ts(ind,cntr-1);
    if mode == 2
        ts = ts(nrpt_ind);
    end
    [alpha_map, theta_map] = map_gamma_shape_scale(ts, gam_a0, gam_b0, gam_c0, gam_d0);
    new_param.gam_alpha(cntr-1) = alpha_map;
    new_param.gam_beta(cntr-1) = theta_map;

    
    %mean_ts = mean(ts);
    %var_ts  = max(1,var(ts));
    %new_param.gam_alpha(cntr-1) = mean_ts^2 /var_ts ;
    %new_param.gam_beta(cntr-1) = var_ts/mean_ts;

    % comment this later
    %new_param.gam_alpha(cntr-1) = init_param.gam_alpha;
    %new_param.gam_beta(cntr-1) = init_param.gam_beta;

end

new_param.event_cnt=max(Cntr)-1; 
if mode == 2
    keep_ms
end

end
    