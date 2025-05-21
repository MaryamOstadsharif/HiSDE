function gp_sde_gpfit_pr(Z, data, init_param, cur_param, mt)
% gp_sde_gpfit_pr  Fit a GP to inducing‐point subsample of Z, compare Y variation, and plot in subplots
%
% gp_sde_gpfit_pr(Z, data, init_param, cur_param, mt)
%   Z           [1×K] observed signal
%   data        struct with fields Ts, Ms, Cntr, Ys
%   init_param  struct with fields dt, K, sample, z_var, …
%   cur_param   struct of current model parameters
%   mt          [1×event_cnt] mean inter‐event intervals

% fit Gaussian‐process regression for the uniform times
mean_interval_uni = mean(mt,'omitnan');
fprintf('Overall mean interval: %.3f s\n', mean_interval_uni);

t_points = round(mean_interval_uni / init_param.dt);
fprintf('t_points: %.3f s\n', t_points);

t_ind_uni = 1:t_points:init_param.K;
fprintf('t_indl: %.3f s\n', t_ind_uni);
Z_ind_uni = Z(t_ind_uni);

gprMdl = fitrgp(t_ind_uni', Z_ind_uni', ...
    'BasisFunction',  'linear', ...
    'KernelFunction', 'squaredexponential', ...
    'Sigma',          sqrt(init_param.z_var), ...
    'ConstantSigma', true, ...
    'Standardize',    true, ...
    'PredictMethod',  'exact');
t_full = 1:init_param.K;
[Z_gp_uni, Z_std_uni] = predict(gprMdl, t_full');

% fit Gaussian‐process regression for the model times
% --- build inducing‐point indices without ever exceeding K ---
t_points = round(mt / init_param.dt);
t_cum = cumsum(t_points(1:end-1));
inds  = t_cum(t_cum < init_param.K);

% force inds to be a row
inds = inds(:)';

% now concatenate 1, inds, and K horizontally
t_ind_model = [ 1, inds, init_param.K ];

% remove any duplicates (e.g. if 1 or K were already in inds)
t_ind_model = unique(t_ind_model, 'stable');

Z_ind_model = Z(t_ind_model);

% optional diagnostics

fprintf('Model inducing indices: %s\n', mat2str(t_ind_model));
fprintf('  → %d points, min=%d, max=%d\n', ...
         numel(t_ind_model), t_ind_model(1), t_ind_model(end));

% ensure column vectors
t_ind_model = t_ind_model(:);
Z_ind_model = Z_ind_model(:);

% fit a single GP on the model‐times subsample
gprMdl_model = fitrgp( ...
    t_ind_model,      Z_ind_model, ...
    'BasisFunction',  'linear', ...
    'KernelFunction', 'squaredexponential', ...
    'Sigma',          sqrt(init_param.z_var), ...
    'ConstantSigma',  true, ...
    'Standardize',    true, ...
    'PredictMethod',  'exact' );

% predict over the full grid (also as a column)
t_full = (1:init_param.K)';
[Z_gp_model, Z_std_model] = predict(gprMdl_model, t_full);

mse_uni   = mean( (Z(:) - Z_gp_uni).^2 );
mse_model = mean( (Z(:) - Z_gp_model).^2 );

fprintf('uniformly spaced points yield an MSE of %.4f,\n', mse_uni);
fprintf('whereas our proposed model achieves an MSE of %.4f\n', mse_model);
meanY = mean(data.Ys(:,1:init_param.K), 1);
t = (1:init_param.K);

hF = figure(3); clf(hF);
    set(hF, ...
        'Visible','on', ...
        'Units','inches', ...
        'Position',[1 1 6 8], ...       % 6" wide, 8" tall
        'PaperUnits','inches', ...
        'PaperPosition',[1 1 6 8], ...
        'PaperSize',[6 8] ...
    );
    % make all axes use Arial / 12 pt by default
    set(hF, ...
        'DefaultAxesFontName','Arial', ...
        'DefaultAxesFontSize',12, ...
        'DefaultTextFontName','Arial', ...
        'DefaultTextFontSize',12 ...
    );

    t_full = (1:init_param.K);
    % 1) Original Z
    ax1 = subplot(4,1,1);
    plot(t_full, Z, 'k-','LineWidth',1);
    xlabel('Time index','FontSize',14,'FontName','Arial');
    ylabel('Z','FontSize',14,'FontName','Arial');

    % 2) Mean of X
    ax2 = subplot(4,1,2);
    meanX = mean(data.Xs,1);
    plot(1:numel(meanX), meanX, 'b-','LineWidth',1.5);
    xlim([1 init_param.K]);
    xlabel('Time index','FontSize',14,'FontName','Arial');
    ylabel('X','FontSize',14,'FontName','Arial');

    % 3) Mark vs. τ
    ax3 = subplot(4,1,3);
    hold(ax3,'on');
    for s = 1:init_param.sample
        cnt = data.Cntr(s);
        if cnt>0
            tau_idx = data.Ts(s,1:cnt)/init_param.dt;
            plot(ax3, tau_idx, data.Ms(s,1:cnt), 'o-','LineWidth',1);
        end
    end
    hold(ax3,'off');
    xlim(ax3,[1 init_param.K]);
    xlabel(ax3,'Time index','FontSize',14,'FontName','Arial');
    ylabel(ax3,'m','FontSize',14,'FontName','Arial');

    % 4) overlay of original and both GP fits
    ax4 = subplot(4,1,4);
    plot(ax4, t_full, Z,          'k-','LineWidth',1.5); hold(ax4,'on');
    plot(ax4, t_full, Z_gp_uni,   'r--','LineWidth',1.5);
    plot(ax4, t_full, meanY, 'b:','LineWidth',1.5);
    hold(ax4,'off');
    xlabel(ax4,'Time','FontSize',12,'FontName','Arial');
    
    % remove the built-in y-label
    ax4.YLabel.String  = '';
    ax4.YLabel.Visible = 'off';
    
    % normalized coords for the custom label:
    x0 = -0.08;    % a little left of the axis
    y0 = 0.50;     % vertical center
    
    % draw the Z
    text(ax4, x0, y0, 'Z', ...
         'Units','normalized', ...
         'HorizontalAlignment','center', ...
         'VerticalAlignment','middle', ...
         'FontName','Arial', ...
         'FontSize',12);
    
    % draw the hat above it
    text(ax4, x0, y0 + 0.03, '^', ...
         'Units','normalized', ...
         'HorizontalAlignment','center', ...
         'VerticalAlignment','bottom', ...
         'FontName','Arial', ...
         'FontSize',12);
    
    % finally re-add the legend up at top-right
    legend(ax4, {'Z','GP\_match','Z-sde'}, ...
           'FontSize',12,'FontName','Arial', ...
           'Location','northeast');

    drawnow;

    % % now save each panel separately
    % panels = {ax1, ax2, ax3, ax4};
    % for i = 1:4
    %     hf = figure('Units','inches','Position',[1 1 6 4], ...
    %                 'PaperUnits','inches','PaperPosition',[1 1 6 4], ...
    %                 'PaperSize',[6 4], ...
    %                 'Visible','off');
    %     set(hf, 'DefaultAxesFontName','Arial', 'DefaultAxesFontSize',12);
    % 
    %     % copy the axes
    %     newAx = copyobj(panels{i}, hf);
    %     set(newAx, 'Position',[.15 .15 .8 .75]);
    % 
    %     % if this is panel 4, re‐add the legend
    %     if i == 4
    %         legend(newAx, ...
    %                {'Z','GP\_match','Y'}, ...    % same labels you used originally
    %                'FontSize',12, ...
    %                'FontName','Arial', ...
    %                'Location','northeast',...
    %                'Interpreter','tex');      
    %     end
    % 
    %     % save as SVG
    %     print(hf, sprintf('panel_%d',i), '-dsvg');
    %     close(hf);
    % end
    save( '1d_plot_and_data.mat' );
end