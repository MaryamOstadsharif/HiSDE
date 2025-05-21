clear all
close all

%% inside lopp
in_loop  = 2;
in_start = 1;

%% load data file
%data = readmatrix('trial_1356_data.csv', 'NumHeaderLines', 1);
data = readmatrix('trial_748_data.csv', 'NumHeaderLines', 1);
% get neuron channels
Z   = data(:,10:end)';
% find channels which have more than 5 spikes
ind = find(sum(Z')>5);
% now, just pick those channels
Z   = Z(ind,800:1600);
% now get the lenght of data
data_len = size(Z,2); 
%% call the following functions first
% set model initial paramaters
[init_param,cur_param] = gp_sde_init_ay(length(ind),3);
% for test
cur_param.W  = init_param.W;
cur_param.z_var = init_param.z_var;
% now, call smc
for iter = 1:4
    
    for l=1:in_loop
        [iter l]
        [iter l 1]
        data = gp_sde_smc_ay(in_start,init_param,Z(:,1:min(iter*200,data_len)),cur_param);
        % now call em
        [iter l 2]
        cur_param = gp_sde_em_ay(Z(:,1:min(iter*00,data_len)),data,init_param,cur_param);
        % make the sde noise scaled
        in_start = in_start + 1; 
        cur_param.W = init_param.W;
        cur_param.z_var = init_param.z_var; 
    end

    subplot(5,2,2)
    imagesc(Z(:,1:min(iter*200,data_len)))
    
    subplot(5,2,4)
    for d=1:3
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
    
   % subplot(5,2,10)
   % data = gp_sde_generate_ay(init_param,cur_param);
    % plot(mean(data.Ys));hold on
    % plot(mean(data.Xs));hold off
    % axis tight
    % 
    % figure(2)
    % plot(data.Xs')
    % pause(0.1)

    save('fit_data.mat','data','cur_param','init_param');

end
