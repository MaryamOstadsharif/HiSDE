function GP_fit_compare(Z, data, init_param, cur_param, mt)
% GP_fit_compare  Compare Gaussian‐Process fits on uniform vs. model‐driven inducing points
%
%   GP_fit_compare(Z, data, init_param, cur_param, mt)
%
%   Inputs:
%     Z          – 1×K (or K×1) observed signal time series
%     data       – struct with fields from SMC:
%                    • Ts   – [sample×event_cnt] event times (absolute)
%                    • Ms   – [sample×event_cnt] marks per event
%                    • Cntr – [sample×1] count of events per particle
%                    • Ys   – [sample×(K+1)] latent integrator trajectories
%     init_param – struct of model settings (fields dt, K, z_var, sample, etc.)
%     cur_param  – struct of learned parameters (event_cnt, gam_alpha, gam_beta, mark_mean, mark_var)
%     mt         – 1×event_cnt vector of mean waiting times (per event)
%
%   Behavior:
%     1. Computes uniform grid of inducing points based on avg waiting time
%     2. Fits two GPR models:
%         • one on uniformly spaced time points
%         • one on event‐driven inducing times inferred by SSM
%     3. Predicts the full signal under each model
%     4. Computes and prints MSE for both fits
%     5. Plots:
%         (1) original Z
%         (2) mean of latent increment X over particles
%         (3) scatter/line of mark vs. time for each particle
%         (4) overlay of original Z, uniform‐GP, and model‐GP predictions

%% 1) Compute uniform‐spacing indices based on average waiting time
mean_interval_uni = mean(mt, 'omitnan');  
fprintf('Overall mean interval: %.3f s\n', mean_interval_uni);

% Convert to number of time‐steps
t_points = round(mean_interval_uni / init_param.dt);
fprintf('t_points (steps): %d\n', t_points);

% Build uniform index grid
t_ind_uni = 1:t_points:init_param.K;
fprintf('Uniform indices: %s\n', mat2str(t_ind_uni));

% Subsample Z at uniform points
Z_ind_uni = Z(t_ind_uni);

%% 2) Fit GP on uniform subsample
%   fitrgp requires column vectors
gpr_uni = fitrgp(...
    t_ind_uni', Z_ind_uni', ...
    'BasisFunction',  'linear', ...
    'KernelFunction', 'squaredexponential', ...
    'Sigma',          sqrt(init_param.z_var), ...
    'ConstantSigma',  true, ...
    'Standardize',    true, ...
    'PredictMethod',  'exact');

% Predict over full time grid
t_full = (1:init_param.K)';
[Z_gp_uni, Z_std_uni] = predict(gpr_uni, t_full);

%% 3) Compute model‐driven inducing indices from waiting times
% Convert each mt to time‐step increments
t_pts_model = round(mt / init_param.dt);
% Cumulative sum to get event times (exclude final event)
t_cum = cumsum(t_pts_model(1:end-1));
% Keep only those within series length
inds = t_cum(t_cum < init_param.K);
% Assemble [1, event‐times, K] and remove duplicates
% after you compute inds:
inds = inds(:)';   % ensure inds is 1×M

t_ind_model = unique([1, inds, init_param.K], 'stable');  % now all are 1×N

fprintf('Model inducing indices: %s\n', mat2str(t_ind_model));
fprintf('  → %d points, min=%d, max=%d\n', numel(t_ind_model), t_ind_model(1), t_ind_model(end));

% Extract Z values at model‐driven points
Z_ind_model = Z(t_ind_model);

%% 4) Fit GP on model‐driven subsample
gpr_model = fitrgp(...
    t_ind_model', Z_ind_model', ...
    'BasisFunction',  'linear', ...
    'KernelFunction', 'squaredexponential', ...
    'Sigma',          sqrt(init_param.z_var), ...
    'ConstantSigma',  true, ...
    'Standardize',    true, ...
    'PredictMethod',  'exact');

% Predict on full grid
[Z_gp_model, Z_std_model] = predict(gpr_model, t_full);

%% 5) Compute and print MSE for both approaches
mse_uni   = mean((Z(:) - Z_gp_uni).^2);
mse_model = mean((Z(:) - Z_gp_model).^2);

fprintf('Uniform grid MSE: %.4f\n', mse_uni);
fprintf('Model‐driven MSE: %.4f\n', mse_model);

%% 6) Prepare mean of latent increments for plotting
meanY = mean(data.Ys(:,1:init_param.K), 1);

%% 7) Plot results in a 4‐panel figure
hF = figure('Units','inches','Position',[1 1 6 8], ...
            'PaperUnits','inches','PaperPosition',[1 1 6 8], ...
            'Visible','on');
set(hF, 'DefaultAxesFontName','Arial', 'DefaultAxesFontSize',12);

% Panel 1: Original signal Z
ax1 = subplot(4,1,1);
plot(t_full, Z, 'k-', 'LineWidth', 1);
xlabel('Time index','FontSize',14);
ylabel('Z','FontSize',14);
title('Observed signal','FontSize',12);

% Panel 2: Mean latent increment X
ax2 = subplot(4,1,2);
meanX = mean(data.Xs,1);
plot(1:numel(meanX), meanX, 'b-', 'LineWidth', 1.5);
xlim([1, init_param.K]);
xlabel('Time index','FontSize',14);
ylabel('X','FontSize',14);
title('Mean latent increment','FontSize',12);

% Panel 3: Marks vs. event times for each particle
ax3 = subplot(4,1,3);
hold(ax3, 'on');
for s = 1:init_param.sample
    cnt = data.Cntr(s);
    if cnt > 0
        tau_idx = data.Ts(s,1:cnt) / init_param.dt;
        plot(ax3, tau_idx, data.Ms(s,1:cnt), 'o-','LineWidth',1);
    end
end
hold(ax3, 'off');
xlim([1, init_param.K]);
xlabel('Time index','FontSize',14);
ylabel('Mark m','FontSize',14);
title('Marks vs. inferred times','FontSize',12);

% Panel 4: Overlay original, uniform‐GP, and model‐GP
ax4 = subplot(4,1,4);
plot(ax4, t_full, Z, 'k-', 'LineWidth', 1.5); hold(ax4, 'on');
plot(ax4, t_full, Z_gp_uni, 'r--', 'LineWidth', 1.5);
plot(ax4, t_full, meanY, 'b:', 'LineWidth', 1.5);
hold(ax4, 'off');
xlabel('Time index','FontSize',14);
legend(ax4, {'Z','GP\_uniform','Latent‐mean'}, 'Location','northeast');
title('GP fits comparison','FontSize',12);

end
