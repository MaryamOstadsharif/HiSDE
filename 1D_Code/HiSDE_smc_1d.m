function data = gp_sde_smc_ay(iter,init_param,Z,cur_param)

scale_adj = init_param.scale_adj;
%% max events
max_events = init_param.max_events;
%% default Gamma parameters
gam_a0  = init_param.gam_a0;
gam_b0  = init_param.gam_b0;
gam_c0  = init_param.gam_c0;
gam_d0  = init_param.gam_d0;

gam_alpha = gam_a0 /gam_b0;
gam_beta  = gam_c0 / gam_d0;

%% default Normal Paramater
norm_mu0   = init_param.norm_mu0;
norm_k0    = init_param.norm_k0;
norm_alpha = init_param.norm_alpha;
norm_beta  = init_param.norm_beta;

%% model fixed parameters
K   = length(Z);%init_param.K;
dt      = init_param.dt;
sample  = init_param.sample;

%% model extra parameters
x_var   = init_param.x_var;
y_var   = init_param.y_var;
z_var   = init_param.z_var;

%% create samples from mark
Xs   = zeros(sample,K+1);  % Xs sample, K+1
Ys   = zeros(sample,K+1);  % Ys sample, K+1
Ms   = zeros(sample,max_events);  % Ms
Ts   = zeros(sample,max_events); % Tau s
Cntr = zeros(sample,1);

%% First Iteration
if iter == 1
    for s=1:sample
        Xs(s,1) = norm_mu0 +  randn * sqrt( norm_beta /((norm_alpha-1)*norm_k0));
        Ys(s,1) = norm_mu0 + randn * sqrt( norm_beta /((norm_alpha-1)*norm_k0));
        Ms(s,1) = 0; 
        Ts(s,1) = 0;
        Cntr(s) = 1;
    end
else
    % Next Iteration
    for s=1:sample
        Xs(s,1) = cur_param.x_mu0 + randn * sqrt( cur_param.x_var0);
        Ys(s,1) = cur_param.y_mu0 + randn * sqrt( cur_param.y_var0);
        Ms(s,1) = 0;
        Ts(s,1) = 0;
        Cntr(s) = 1;
    end

end
%% Go over samples
for k=1:K
    Ls = zeros(sample,1);
    % for every sample
    for s=1:sample
        % find counter number, update if new event is needed
         cur_cntr = Cntr(s); 
         if Ts(s,cur_cntr) < k*dt
            % it is visited, draw sample based on visited mark/event
            if cur_cntr <= cur_param.event_cnt
                tmp_gam_alpha = cur_param.gam_alpha(cur_cntr);
                tmp_gam_beta = cur_param.gam_beta(cur_cntr);
                next_wait = max(dt* 2,gamrnd(tmp_gam_alpha,tmp_gam_beta));
                Ts(s,cur_cntr+1) = Ts(s,cur_cntr) + next_wait;
                Ms(s,cur_cntr+1) = cur_param.mark_mean(cur_cntr) + randn * sqrt(cur_param.mark_var(cur_cntr));
            else
                next_wait = max(dt * 2,gamrnd(gam_alpha,gam_beta));
                Ts(s,cur_cntr+1) = Ts(s,cur_cntr) + next_wait;
                Ms(s,cur_cntr+1) = norm_mu0 + randn * sqrt( norm_beta /((norm_alpha-1)*norm_k0));
            end
            Cntr(s) = cur_cntr + 1;
        end
        % now draw sample
        t2 = Ts(s,Cntr(s));
        t1 = Ts(s,Cntr(s)-1);
        b  = Ms(s,Cntr(s));
        ak = 1 - (dt/max(0.5*dt,t2-k*dt));
        bk = b * dt/ max(0.5*dt,t2-k*dt);
        sk = (t2-k*dt)*(k*dt-t1)/(t2-t1);

        
        % % update
        % R = y_var*dt;
        % S = z_var;
        % 
        % temp_var   = R*S/(R+S);
        % temp_mean = temp_var * ((Xs(s,k) * dt + (1-rate_dec*dt) * Ys(s,k))/R + Z(k)/S);
        % Ys(s,k+1) = temp_mean + sqrt(temp_var) * randn;
        % Xs(s,k+1) = ak * Xs(s,k) + bk + sqrt(x_var*sk*dt)*randn; 
        % % calculate likelihood
        % Ls(s) = pdf('normal',Z(k),Ys(s,k+1),sqrt(z_var)) * pdf('normal',Ys(s,k+1),(1-rate_dec*dt) *Ys(s,k) + Xs(s,k) * dt,sqrt(R)) / pdf('normal',Ys(s,k+1),temp_mean,sqrt(temp_var));

        Xs(s,k+1) = ak * Xs(s,k) + bk + sqrt(x_var*sk*dt)*randn; 
        % This is the Y, the integrator
        Ys(s,k+1) = Ys(s,k) + Xs(s,k) * dt * scale_adj + sqrt(y_var*dt)*randn;
        % % calculate likelihood
        Ls(s) = max(10*realmin,pdf('normal',Z(k),Ys(s,k+1),sqrt(z_var)));
       
        
    end
    
    % now, resample
    Ls  = Ls/sum(Ls);
    r   = rand(sample, 1);
    cumulative_L = cumsum(Ls);
    indices      = arrayfun(@(x) find(cumulative_L >= x, 1, 'first'), r);
    Xs  = Xs(indices,:);
    Ys  = Ys(indices,:);
    Ms  = Ms(indices,:);
    Ts  = Ts(indices,:);
    Cntr = Cntr(indices);
end
    
%% send the output
data.Xs = Xs;
data.Ys = Ys;
data.Ms= Ms;
data.Ts= Ts;
data.Cntr = Cntr;

end

