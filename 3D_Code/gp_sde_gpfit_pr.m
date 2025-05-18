function gp_sde_gpfit_pr(Z, Y_fit, Y_true, Xs, dt, tau_idx)
  % gp_sde_gpfit_pr - Fits a Gaussian Process model within a stochastic 
  % differential equation (SDE) framework and evaluates the predictions.
  %
  %   Z        - (n x d) matrix of latent state trajectories or features used for GP fitting,
  %              where n is the number of time steps and d is the dimensionality.
  %   Y_fit    - (n x m) matrix of predicted outputs from the GP model, where m is the 
  %              number of output dimensions.
  %   Y_true   - (n x m) matrix of true observed outputs for comparison with Y_fit.
  %   Xs       - (k x d) matrix of input test points for GP prediction (optional or used internally).
  %   dt       - Scalar representing the time step size used in the SDE discretization.
  %   tau_idx  - Integer index or array indicating the specific time lag(s) or indices
  %              for evaluation or integration in the GP-SDE context.

  % (1) Z heatmap — almost full‐figure
  fig1 = figure('Name','Z_image', ...
                'Units','normalized', ...
                'Position',[0.1 0.1 0.8 0.8]);  % [x y w h] relative to screen
  ax1 = axes(fig1, 'Position',[0.05 0.05 0.90 0.90]);  % 90% of fig

  imagesc(ax1, Z)
  axis(ax1, 'tight')
  colorbar(ax1)
  xlabel(ax1, 'Time index')
  ylabel(ax1, 'Z')


  % now fix the export size to 6×4 inches
  w = 8; h = 6;
  set(fig1, ...
    'Units','inches', ...
    'Position',[1 1 w h], ...
    'PaperUnits','inches', ...
    'PaperPosition',[0 0 w h], ...
    'PaperSize',[w h]);
  saveas(fig1, 'Z_image.svg')


  % (2) 3D Trajectories
  fig2 = figure('Name','Trajectories','Units','normalized','Position',[0.1 0.1 0.8 0.8]);
  ax2 = axes(fig2, 'Position',[0.10 0.10 0.80 0.80]);
  t1 = mean(squeeze(Y_fit(1,:,:))); t2 = mean(squeeze(Y_fit(2,:,:))); t3 = mean(squeeze(Y_fit(3,:,:)));
  h1 = plot3(ax2, t1, t2, t3, '.-','Color',[0 0.4470 0.7410],'LineWidth',1); hold(ax2,'on');
  h2 = plot3(ax2, Y_true(1,:), Y_true(2,:), Y_true(3,:), 'r--','LineWidth',1.5);
  mask_valid = tau_idx>=1 & tau_idx<=size(Y_true,2);
  valid     = tau_idx(mask_valid);
  P         = Y_true(:, valid);

  % inducing points (black)
  h3 = plot3(ax2, P(1,:), P(2,:), P(3,:), 'ko', ...
               'MarkerFaceColor','k','MarkerSize',6);
  % --- mid‐segment arrows ---
  dP     = diff(P,1,2);               % now 3×(M–1)
  segLen = sqrt(sum(dP.^2,1));        % 1×(M–1)
  unitD  = dP ./ segLen;              % 3×(M–1)
  V      = unitD .* (0.2 .* segLen);  % 3×(M–1)
  midPts = (P(:,1:end-1) + P(:,2:end)) / 2;  % 3×(M–1)

  quiver3(ax2, midPts(1,:), midPts(2,:), midPts(3,:), ...
          V(1,:),     V(2,:),     V(3,:),     0, ...
          'Color','k','MaxHeadSize',0.2,'LineWidth',1);
  hold(ax2,'off');
  xlabel(ax2,'Y_1'); ylabel(ax2,'Y_2'); zlabel(ax2,'Y_3'); grid(ax2,'on');
  legend(ax2, [h1 h2 h3], {'Y','Y_{true}','Inducing pts'}, 'Location','best');

  % now fix the export size to 6×4 inches
  w = 6; h = 4;
  set(fig1, ...
    'Units','inches', ...
    'Position',[1 1 w h], ...
    'PaperUnits','inches', ...
    'PaperPosition',[0 0 w h], ...
    'PaperSize',[w h]);
  saveas(fig2, 'Trajectories.png')


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
