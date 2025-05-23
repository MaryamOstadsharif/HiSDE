function [Z, Y] = HiSDE_model_z_3d(init_param)
% model_z_ay  Simulate 3D Lorenz‐based latent trajectories and corresponding noisy observations
%
%   [Z, Y] = model_z_ay(init_param)
%
%   Inputs:
%     init_param – struct containing model settings:
%                    • K      – number of time‐steps
%                    • dt     – time‐step size (unused for Lorenz sim, hard‐coded inside)
%                    • z_var  – observation noise covariance (N×N)
%                    • W      – measurement matrix (N×3) mapping 3D latent to observations
%
%   Outputs:
%     Z – [N×K] matrix of simulated observations at each time‐step
%     Y – [3×K] matrix of true latent state trajectories (centered to zero mean each dimension)

%% Unpack parameters
K      = init_param.K;      % number of time‐steps to simulate
% dt      = init_param.dt;   % time step from init_param (we override below for Lorenz)
S      = init_param.z_var;  % observation noise covariance (N×N)
W      = init_param.W;      % measurement matrix (N×3)

%% 1) Simulate Lorenz dynamics for latent state
%   Use a finer dt for Lorenz system to ensure stability
lorenz_dt = 0.01;            
y0 = [-1, 1, -1];            % initial condition for [x,y,z]
[~, ys] = Simulate_Lorenz([0, (K-1)*lorenz_dt], lorenz_dt, y0);
% ys is [K×3] array of latent states over time

% Center each coordinate to zero mean
ys(:,1) = ys(:,1) - mean(ys(:,1));
ys(:,2) = ys(:,2) - mean(ys(:,2));
ys(:,3) = ys(:,3) - mean(ys(:,3));

%% 2) Generate noisy observations Z from latent state Y
%   For each time‐step i, sample Z(:,i) ~ N(W * Y(:,i), S)
N = size(W,1);              % observation dimension
Z = zeros(N, K);
for i = 1:K
    % Extract latent vector at time i (3×1)
    y_i = ys(i, :)';
    % Sample observation with multivariate Normal noise
    Z(:, i) = mvnrnd(W * y_i, S)';
end

%% 3) Return Y in 3×K orientation
%   ys is K×3; transpose to 3×K for consistency with other functions
Y = ys';
end
