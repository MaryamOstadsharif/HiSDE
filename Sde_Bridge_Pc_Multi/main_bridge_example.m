% Main script for GP‐SDE fitting on neural spike data (last application)

clear all
close all

%% 1) Configure inference loop parameters
in_loop  = 2;    % number of SMC+EM updates per outer iteration
in_start = 1;    % starting index for the sliding observation window

%% 2) Load and preprocess neural data
% Read CSV (skip header) containing spike counts / voltages
data = readmatrix('trial_748_data.csv', 'NumHeaderLines', 1);

% Extract channels (columns 10:end) and transpose to [channels × time]
Z = data(:,10:end)';

% Select only channels with more than 5 total spikes
ind = find(sum(Z,2) > 5);

% Restrict to those active channels and a time‐window (columns 800–1600)
Z = Z(ind, 800:1600);

% Determine length of the time series
data_len = size(Z, 2);

%% 3) Initialize GP‐SDE model parameters
% [init_param, cur_param] = HiSDE_init_md(N, dim)
% Inputs:
%   N   – number of observed channels (here: length(ind))
%   dim – latent state dimensionality (here: 3)
[init_param, cur_param] = HiSDE_init_md(length(ind), 3);

% For testing, set the measurement matrix and noise variance to true values
cur_param.W     = init_param.W;
cur_param.z_var = init_param.z_var;

%% 4) Inference loop: gradually expand data window and alternate SMC + EM
for iter = 1:4
    for l = 1:in_loop
        % Display current iteration counters
        disp(['EM iter = ' num2str(iter) ', inner loop = ' num2str(l)]);
        
        %% 4a) Sequential Monte Carlo step
        % data = HiSDE_smc_md(start_idx, init_param, Z_window, cur_param)
        %   start_idx  – index at which the window begins
        %   Z_window   – Z(:,1:window_end) is the observed subset
        data = HiSDE_smc_md( ...
            in_start, ...
            init_param, ...
            Z(:,1:min(iter*200, data_len)), ...
            cur_param ...
        );

        %% 4b) Expectation–Maximization update
        % cur_param = HiSDE_em_md(Z_window, data, init_param, cur_param)
        cur_param = HiSDE_em_md( ...
            Z(:,1:min(iter*200, data_len)), ...
            data, ...
            init_param, ...
            cur_param ...
        );

        % Move window start forward by one time‐step
        in_start = in_start + 1;

        % Reset measurement parameters after EM
        cur_param.W     = init_param.W;
        cur_param.z_var = init_param.z_var;
    end

    %% 5) Visualize intermediate results for this outer iteration
    % 5a) Heatmap of observed spikes in the current window
    subplot(5,2,2);
    imagesc(Z(:,1:min(iter*200, data_len)));
    title(['Observed data (up to t=' num2str(min(iter*200, data_len)) ')']);
    xlabel('Time index');
    ylabel('Channel');

    % 5b) First 10 inferred integrator trajectories (Y)
    subplot(5,2,4);
    hold on;
    for d = 1:3
        tYs = squeeze(data.Ys(d,1:10,:));  % [10 trajectories × time]
        plot(tYs');
    end
    hold off;
    axis tight;
    title('Integrator Y (first 10 particles)');
    xlabel('Time index');
    ylabel('Y');

    % 5c) First 10 latent increments (X)
    subplot(5,2,6);
    hold on;
    for d = 1:3
        tXs = squeeze(data.Xs(d,1:10,:));
        plot(tXs');
    end
    hold off;
    axis tight;
    title('Latent increment X (first 10 particles)');
    xlabel('Time index');
    ylabel('X');

    %% 6) Save fitted data and parameters for later analysis
    save('fit_data.mat', 'data', 'cur_param', 'init_param');
end
