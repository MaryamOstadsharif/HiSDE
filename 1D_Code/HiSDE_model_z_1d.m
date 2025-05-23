function Z = HiSDE_model_z_1d(model, init_param)
% HiSDE_model_z_1d  Generate synthetic 1D observations for the HiSDE model
%
%   Z = HiSDE_model_z_1d(model, init_param)
%
%   Inputs:
%     model      – integer code selecting the signal type:
%                    1: exponentiated cosine wave
%                    2: power-law trend with floor
%                    3: reversed chirp with exponential transform
%                    4: Lorenz attractor’s third coordinate
%     init_param – struct returned by HiSDE_init_1d containing:
%                    • K     – number of time‐steps
%                    • dt    – time‐step size
%                    • z_var – observation noise variance
%
%   Output:
%     Z          – 1×K (or K×1 for model 4) vector of noisy observations

%% Unpack parameters
K     = init_param.K;      % total number of samples
dt    = init_param.dt;     % time increment
z_var = init_param.z_var;  % variance of additive Gaussian noise

%% Generate signal depending on selected model
if model == 1
    % Model 1: exponentiated cosine wave
    %   Z_k = exp(cos(2π·0.017·dt·k)) + noise
    k = 1:K;
    Z = exp(cos(2*pi*0.017*dt*k));
    Z = Z + sqrt(z_var) * randn(size(Z));

elseif model == 2
    % Model 2: power-law growth with minimum value
    %   Z_k = max(0.5·dt·k^1.3, 10) + noise
    k = 1:K;
    Z = max(dt * (k.^1.3) * 0.5, 10);
    Z = Z + sqrt(z_var) * randn(size(Z));

elseif model == 3
    % Model 3: time-reversed chirp transformed by exponential
    %   1) Generate chirp over [0, end]
    %   2) Reverse it, transform via 7·exp(–chirp)
    %   3) Center to zero mean and add noise
    ts = dt * (1:K);
    Z  = chirp(ts, 0.001, 8, 0.002);   % linear‐frequency‐sweep chirp
    Z  = Z(end:-1:1);                  % reverse time
    Z  = 7 * exp(-Z) + sqrt(z_var) * randn(size(Z));
    Z  = Z - mean(Z);                  % center to zero mean

elseif model == 4
    % Model 4: Lorenz‐system third coordinate
    %   1) Simulate Lorenz dynamics over time [0, (K-1)*dt]
    %   2) Extract z‐axis and add noise
    dt_sim = 0.01;                      % Lorenz requires finer dt
    y0     = [1, -1, -1];               % initial condition for [x,y,z]
    [~, Y] = lorenz_dynamics([0, (K-1)*dt_sim], dt_sim, y0);
    Z      = Y(:,3) + sqrt(z_var) * randn(size(Y(:,3)));
    % Note: output is K×1 here (instead of 1×K for other models)

else
    error('HiSDE_model_z_1d:InvalidModel', ...
          'Model code must be 1, 2, 3, or 4.');
end
end
