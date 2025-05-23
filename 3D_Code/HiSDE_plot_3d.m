function HiSDE_plot_3d(Z, Y_fit, Y_true, Xs, dt, tau_idx)
% HiSDE_plot_3d  Visualize 3D SSM inference and inducing‐point results
%
%   HiSDE_plot_3d(Z, Y_fit, Y_true, Xs, dt, tau_idx)
%
%   Inputs:
%     Z         – observed data matrix (time‐steps × features), used for heatmap
%     Y_fit     – inferred integrator trajectories from SMC 
%                 (particles × 3 × time‐steps)
%     Y_true    – true latent trajectories (3 × time‐steps)
%     Xs        – latent increment trajectories (either 3×time‐steps or particles×3×time‐steps)
%     dt        – scalar time‐step size
%     tau_idx   – 1×M vector of integer time‐step indices of inferred events
%
%   Behavior:
%     Generates and saves three figures:
%       1) Heatmap of Z over time
%       2) 3D plot comparing mean(Y_fit), Y_true, and inducing‐point arrows
%       3) Time series of each X dimension with event markers

  %% 1) Heatmap of observed data Z
  fig1 = figure('Name','Z_heatmap', ...
                'Units','normalized', ...
                'Position',[0.1 0.1 0.8 0.8]);
  ax1 = axes(fig1, 'Position',[0.05 0.05 0.90 0.90]);
  imagesc(ax1, Z);                 % display Z as color image
  axis(ax1, 'tight');
  colorbar(ax1);
  xlabel(ax1, 'Time index');
  ylabel(ax1, 'Feature / Dimension');
  title(ax1, 'Observed data heatmap');

  % Set export size to 8×6 inches and save
  w = 8; h = 6;
  set(fig1, ...
      'Units','inches', ...
      'Position',[1 1 w h], ...
      'PaperUnits','inches', ...
      'PaperPosition',[0 0 w h], ...
      'PaperSize',[w h]);
  saveas(fig1, 'Z_heatmap.svg');

  %% 2) 3D trajectories and inducing‐point arrows
  fig2 = figure('Name','3D_Trajectories', ...
                'Units','normalized', ...
                'Position',[0.1 0.1 0.8 0.8]);
  ax2 = axes(fig2, 'Position',[0.10 0.10 0.80 0.80]);

  % Compute mean inferred trajectory across particles
  Yfit_mean = squeeze(mean(Y_fit, 1));  % results in 3×time‐steps
  h1 = plot3(ax2, Yfit_mean(1,:), Yfit_mean(2,:), Yfit_mean(3,:), ...
             '.-','LineWidth',1); hold(ax2,'on');

  % Plot true latent trajectory
  h2 = plot3(ax2, Y_true(1,:), Y_true(2,:), Y_true(3,:), ...
             'r--','LineWidth',1.5);

  % Extract valid inducing‐point coordinates from Y_true
  valid = tau_idx(tau_idx>=1 & tau_idx<=size(Y_true,2));
  P     = Y_true(:, valid);
  h3    = plot3(ax2, P(1,:), P(2,:), P(3,:), 'ko', ...
               'MarkerFaceColor','k','MarkerSize',6);

  % Add mid‐segment arrows showing direction between inducing points
  dP     = diff(P,1,2);                           % 3×(M–1)
  segLen = sqrt(sum(dP.^2,1));                    % 1×(M–1)
  unitD  = dP ./ segLen;                          % 3×(M–1)
  V      = unitD .* (0.2 .* segLen);              % scale arrows
  midPts = (P(:,1:end-1) + P(:,2:end)) / 2;        % 3×(M–1)
  quiver3(ax2, midPts(1,:), midPts(2,:), midPts(3,:), ...
          V(1,:),     V(2,:),     V(3,:),     0, ...
          'Color','k','MaxHeadSize',0.2,'LineWidth',1);

  hold(ax2,'off');
  xlabel(ax2,'Y_1');
  ylabel(ax2,'Y_2');
  zlabel(ax2,'Y_3');
  grid(ax2,'on');
  legend(ax2, [h1, h2, h3], {'Inferred mean','True','Events'}, 'Location','best');
  title(ax2, '3D latent trajectories with inducing points');

  % Save as PNG at 6×4 inches
  w = 6; h = 4;
  set(fig2, ...
      'Units','inches', ...
      'Position',[1 1 w h], ...
      'PaperUnits','inches', ...
      'PaperPosition',[0 0 w h], ...
      'PaperSize',[w h]);
  saveas(fig2, '3D_Trajectories.png');

  % (3) Xs plot
  fig3 = figure('Name','Xs_plot','Units','normalized','Position',[0.1 0.1 0.8 0.8]);
  ax3 = axes(fig3, 'Position',[0.10 0.10 0.80 0.80]);
  if ndims(Xs)==3 && size(Xs,1)==3
    Xs = squeeze(Xs(:,1,:));
  elseif ismatrix(Xs)
    if size(Xs,2)==3 && size(Xs,1)~=3, Xs = Xs.'; end
  else, error('Xs dims wrong'), end

  Nx = size(Xs,2); tX = 1:Nx;
  valid_idx = tau_idx(tau_idx>=1 & tau_idx<=Nx);
  cols = {[0 0.4470 0.7410],[0 0.6740 0.1880],[0.8500 0.3250 0.0980]};

  hold(ax3,'on')
    for i=1:3
      hL(i) = plot(ax3, tX, Xs(i,:), 'LineWidth',1, 'Color', cols{i});
      plot(ax3, tX(valid_idx), Xs(i,valid_idx), 'ko','MarkerFaceColor','k', ...
           'MarkerSize',6,'HandleVisibility','off');
    end
  hold(ax3,'off');
  legend(ax3, hL, {'X_1','X_2','X_3'}, 'Location','best');
  xlabel(ax3,'Time index'); ylabel(ax3,'X'); xlim(ax3,[1 Nx]);

  % now fix the export size to 6×4 inches
  w = 8; h = 6;
  set(fig3, ...
    'Units','inches', ...
    'Position',[1 1 w h], ...
    'PaperUnits','inches', ...
    'PaperPosition',[0 0 w h], ...
    'PaperSize',[w h]);
  saveas(fig3, 'Xs_plot.svg')
end