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
norm_mu0 = init_param.norm_mu0;
norm_k0  = init_param.norm_k0 ;
norm_phi = init_param.norm_phi;
norm_v0 = init_param.norm_v0;

%% model fixed parameters
K   = size(Z,2);%init_param.K;
dt      = init_param.dt;
sample  = init_param.sample;

D = init_param.D;

%% model extra parameters
x_var   = init_param.x_var;
y_var   = init_param.y_var;

% load model params
W  = cur_param.W(:,1:D);
W0 = cur_param.W(:,D+1);
z_var = cur_param.z_var;


%% create samples from mark
Xs   = zeros(D,sample,K+1);  %3, Xs sample, K+1
Ys   = zeros(D,sample,K+1);  % Ys sample, K+1
Ms   = zeros(D,sample,max_events);  % Ms
Ts   = zeros(sample,max_events); % Tau s
Cntr = zeros(sample,1);

%% First Iteration
if iter == 1
    for s=1:sample
        tmp_xs = mvnrnd(norm_mu0,norm_phi) ;
        Xs(:,s,1) = tmp_xs;
        tmp_ys = mvnrnd(norm_mu0,norm_phi) ;
        Ys(:,s,1) = tmp_ys;
        Ms(:,s,1) = zeros(1,D);
        Ts(s,1) = 0;
        Cntr(s) = 1;
    end
else
    % Next Iteration
    for s=1:sample
        tmp_xs = mvnrnd(cur_param.x_mu0,cur_param.x_var0) ;
        Xs(:,s,1) = tmp_xs;
        tmp_ys = mvnrnd(cur_param.y_mu0,cur_param.y_var0) ;
        Ys(:,s,1) = tmp_ys;
        Ms(:,s,1) = zeros(1,D);
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
                next_wait = max(dt*2,gamrnd(tmp_gam_alpha,tmp_gam_beta));
                Ts(s,cur_cntr+1) = Ts(s,cur_cntr) + next_wait;
                
                Ms(:,s,cur_cntr+1) = mvnrnd(cur_param.mark_mean(cur_cntr,:)',squeeze(cur_param.mark_var(cur_cntr,:,:)));
            else
                next_wait = max(dt*2,gamrnd(gam_alpha,gam_beta));
                Ts(s,cur_cntr+1) = Ts(s,cur_cntr) + next_wait;

                Ms(:,s,cur_cntr+1) = mvnrnd(norm_mu0,norm_phi);
            end
            Cntr(s) = cur_cntr + 1;
        end
        % now draw sample
        t2 = Ts(s,Cntr(s));
        t1 = Ts(s,Cntr(s)-1);
        ak = 1 - (dt/max(0.5*dt,t2-k*dt));
        for d=1:D
            b  = Ms(d,s,Cntr(s));
            bk = b * dt/ max(0.5*dt,(t2-k*dt));
            sk = (t2-k*dt)*(k*dt-t1)/(t2-t1);
            Xs(d,s,k+1) = ak * Xs(d,s,k) + bk + sqrt(x_var*sk*dt)*randn; 
            % This is the Y, the integrator
            Ys(d,s,k+1) = Ys(d,s,k) + Xs(d,s,k) * dt * scale_adj + sqrt(y_var*dt)*randn;
        end
        

        
        % calculate likelihood more efficiently
        obs    = Z(:,k);
        y      = Ys(:,s,k+1);
        lambda = exp(W0 + W * y);

        % Use dot product for better efficiency and numerical stability
        loglik = obs' * log(lambda) - dt * sum(lambda);
        cur_ll = exp(loglik);
       
        % Clamp to avoid underflow
        Ls(s) = max(10*realmin, cur_ll);
        
    end
    
    % now, resample
    Ls  = Ls/sum(Ls);
    r   = rand(sample, 1);
    cumulative_L = cumsum(Ls);
    indices      = arrayfun(@(x) find(cumulative_L >= x, 1, 'first'), r);
    Ts  = Ts(indices,:);
    for d=1:D
        Xs(d,:,:)  = Xs(d,indices,:);
        Ys(d,:,:)  = Ys(d,indices,:);
        Ms(d,:,:)  = Ms(d,indices,:);
    end
    Cntr = Cntr(indices);
end
    
%% send the output
data.Xs = Xs;
data.Ys = Ys;
data.Ms= Ms;
data.Ts= Ts;
data.Cntr = Cntr;

end

