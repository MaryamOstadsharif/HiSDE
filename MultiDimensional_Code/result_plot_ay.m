close all
%% Load data

data = readmatrix('trial_748_data.csv', 'NumHeaderLines', 1);
% find channels which have more than 5 spikes
ind = find(sum(data(:,10:end))>5);
% get neuron channels
Z   = data(800:1600,10:end)';

% now, just pick those channels
Z   = Z(ind,:);
% now get the lenght of data
data_len = size(Z,2); 

%% Load Model Fit
load('fit_data_748.mat');
Y = squeeze(mean(data.Ys(:,:,:),2));
Yp = Y;
Y = [Y;ones(1,size(Y,2))];
W = cur_param.W;
lambda = [];
for c = 1:length(ind)
    tmp = exp(Y'*W(c,:)');
    lambda =[lambda;tmp'];
end

% Plot Neuron Rster Plot
[num_neurons, num_timepoints] = size(Z);
spike_height = 2.6;  % Height of each spike mark

figure(1);
hold on;

for neuron = 1:num_neurons
    spike_times = find(Z(neuron, :) == 1);
    for t = spike_times
        % Draw a short vertical line for each spike
        plot([t t], [neuron - spike_height/2, neuron + spike_height/2], 'k');
    end
end

xlabel('Time (msec)');
ylabel('Neuron Index');
%title('Raster Plot');
ylim([0.5, num_neurons + 0.5]);
xlim([1, num_timepoints]);
set(gca, 'YDir', 'reverse');  % Optional: Reverse Y to match raster convention
hold off
axis tight

% % Figure, plot Y
% figure;
% plot(Yp')
% xlabel('Time');
% ylabel('Y');
% axis tight

% % Figure, plot Y
% figure;
% start_ind = 1;
% end_ind   = 800;
% plot3(Yp(1,start_ind:end_ind),Yp(2,start_ind:end_ind),Yp(3,start_ind:end_ind))
% xlabel('Time');
% ylabel('Y');
% axis tight


% Figure, log Lambda
figure(3);
imagesc(log(lambda))
xlabel('Time (msec)');
ylabel('Spike Rate');
axis tight
colorbar

figure(2);
plot(Yp','LineWidth',2)
ms = cur_param.gam_alpha .* cur_param.gam_beta;
ms = ms(1:cur_param.event_cnt)/init_param.dt;
hold on
plot(cumsum(ms),-12,'ko','MarkerSize',12,'LineWidth',3)
xlabel('Time (msec)');
ylabel('Y');
axis tight
hold off
