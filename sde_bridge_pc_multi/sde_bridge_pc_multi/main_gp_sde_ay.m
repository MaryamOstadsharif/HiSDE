clear all
close all

%% inside lopp
in_loop  = 2;
in_start = 1;

%% call the following functions first
% set model initial paramaters
[init_param,cur_param] = gp_sde_init_ay();
% create Z
[Z,Y]  = model_md_ay(init_param);
% for test
cur_param.W = init_param.W;
cur_param.z_var = init_param.z_var; 
% now, call smc
for iter = 1:20
    
    for l=1:in_loop
        data = gp_sde_smc_ay(in_start,init_param,Z(:,1:min(100+iter*100,init_param.K)),cur_param);
        % now call em
        cur_param = gp_sde_em_ay(Z(:,1:min(100+iter*100,init_param.K)),data,init_param,cur_param);
        % make the sde noise scaled
        in_start = in_start + 1; 
        cur_param.W = init_param.W;
        cur_param.z_var = init_param.z_var; 
    end

  
    %% update
    % plot ms
    figure(1)
    subplot(5,2,1)
    d=1;
    plot(cur_param.mark_mean(1:cur_param.event_cnt,d));hold on
    plot(cur_param.mark_mean(1:cur_param.event_cnt,d)+sqrt(cur_param.mark_var(1:cur_param.event_cnt,d,d)));
    plot(cur_param.mark_mean(1:cur_param.event_cnt,d)-sqrt(cur_param.mark_var(1:cur_param.event_cnt,d,d)));
    hold off
    subplot(5,2,3)
    d=2;
    plot(cur_param.mark_mean(1:cur_param.event_cnt,d));hold on
    plot(cur_param.mark_mean(1:cur_param.event_cnt,d)+sqrt(cur_param.mark_var(1:cur_param.event_cnt,d,d)));
    plot(cur_param.mark_mean(1:cur_param.event_cnt,d)-sqrt(cur_param.mark_var(1:cur_param.event_cnt,d,d)));
    hold off
    subplot(5,2,5)
    d=3;
    plot(cur_param.mark_mean(1:cur_param.event_cnt,d));hold on
    plot(cur_param.mark_mean(1:cur_param.event_cnt,d)+sqrt(cur_param.mark_var(1:cur_param.event_cnt,d,d)));
    plot(cur_param.mark_mean(1:cur_param.event_cnt,d)-sqrt(cur_param.mark_var(1:cur_param.event_cnt,d,d)));
    hold off
    subplot(5,2,7)
    mt = cur_param.gam_alpha(1:cur_param.event_cnt) .* cur_param.gam_beta(1:cur_param.event_cnt); 
    st = mt .* cur_param.gam_beta(1:cur_param.event_cnt);
    plot(mt);hold on
    plot(mt+sqrt(st));
    plot(mt-sqrt(st));
    hold off
    
    subplot(5,2,2)
    plot(Y');
    axis tight
    
    subplot(5,2,4)
    for d=1:4
        tYs = squeeze(data.Ys(d,1:10,:));
        plot(tYs');hold on
    end
    hold off
    axis tight

    subplot(5,2,6)
    for d=1:3
        tXs = squeeze(data.Xs(d,1:10,:));
        plot(tXs');hold on
    end
    hold off
    axis tight

    % subplot(5,2,8)
    % for s=1:100:init_param.sample
    %     plot(data.Ts(s,1:data.Cntr(s))/init_param.dt,data.Ms(s,1:data.Cntr(s)))
    %     hold on
    % end
    % hold off
    % xlim([0 init_param.K])
    % title(['iter' num2str(iter) 'x noise=' num2str(init_param.x_var) ' y noise=' num2str(init_param.y_var)])
    % 
    
    subplot(5,2,10)
    data = gp_sde_generate_ay(init_param,cur_param);
    % plot(mean(data.Ys));hold on
    % plot(mean(data.Xs));hold off
    % axis tight
    % 
    % figure(2)
    % plot(data.Xs')
    % pause(0.1)

    
    


end
