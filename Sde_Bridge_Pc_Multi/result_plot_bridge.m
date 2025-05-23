close all
%% Load and preprocess data
% Read CSV of neural spike data, skipping header line
data = readmatrix('trial_748_data.csv', 'NumHeaderLines', 1);

% Identify channels with more than 5 total spikes across the recording
ind = find(sum(data(:,10:end)) > 5);

% Extract spike matrix Z: timepoints 800–1600, channels in columns 10:end
% Transpose so rows = channels, columns = time
Z = data(800:1600, 10:end)';

% Keep only the active channels
Z = Z(ind, :);

% Determine length of the time series
data_len = size(Z, 2);

%% Load model fit results
% Assumes 'fit_data_748.mat' contains variables:
%   data.Ys       – SMC integrator trajectories [3×sample×(K+1)]
%   cur_param     – struct with learned parameters
%   init_param    – struct with model settings
load('fit_data_748.mat');

% Compute the mean integrator trajectory across particles: [3×K+1]
Y = squeeze(mean(data.Ys, 2));
Yp = Y;  % keep a copy for plotting

% Append a row of ones for the bias term in the rate model
Y = [Y; ones(1, size(Y, 2))];

% Extract learned measurement weights (including bias) [N×4]
W = cur_param.W;

% Compute predicted firing rate λ for each channel and time: [N×K]
lambda = [];
for c = 1:length(ind)
    % For channel c, compute λ = exp( W(c,:) * Y )
    tmp = exp(Y' * W(c, :)');
    lambda = [lambda; tmp'];
end

%% Figure 1: Spike raster plot
[num_neurons, num_timepoints] = size(Z);
spike_height = 2.6;  % vertical length of each spike mark

figure(1);
hold on;
for neuron = 1:num_neurons
    % Find all timepoints where this neuron fired
    spike_times = find(Z(neuron, :) == 1);
    for t = spike_times
        % Draw a vertical line at each spike time
        plot([t t], [neuron - spike_height/2, neuron + spike_height/2], 'k');
    end
end
% Label axes and invert Y for conventional raster orientation
xlabel('Time (msec)');
ylabel('Neuron Index');
ylim([0.5, num_neurons + 0.5]);
xlim([1, num_timepoints]);
set(gca, 'YDir', 'reverse');
hold off;
axis tight;

%% Figure 3: Heatmap of log‐predicted firing rates
figure(3);
imagesc(log(lambda));      % color = log rate
xlabel('Time (msec)');
ylabel('Neuron Index');
axis tight;
colorbar;

%% Figure 2: Latent Y trajectories with event markers
figure(2);
% Plot each dimension of the mean integrator over time
plot(Yp', 'LineWidth', 2);

% Compute mean waiting‐times in time‐steps from learned Gamma params
ms = cur_param.gam_alpha .* cur_param.gam_beta;
ms = ms(1:cur_param.event_cnt) / init_param.dt;

% Mark inferred event times along the x-axis at a constant y-level
hold on;
plot(cumsum(ms), -12 * ones(size(ms)), 'ko', ...
     'MarkerSize', 12, 'LineWidth', 3);
hold off;

xlabel('Time (msec)');
ylabel('Y (Integrator)');
axis tight;
