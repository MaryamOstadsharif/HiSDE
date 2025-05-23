function [t_uniform, y_uniform] = Simulate_Lorenz(tspan, dt, y0, sigma, beta, rho)
% Simulate_Lorenz  Simulate and interpolate the Lorenz dynamical system
%
%   [t_uniform, y_uniform] = Simulate_Lorenz(tspan, dt, y0, sigma, beta, rho)
%
%   Solves the classic Lorenz equations over a given time span, then
%   interpolates the solution onto a uniform time grid.
%
%   Inputs:
%     tspan – 1×2 vector [t_start, t_end], the time interval for integration
%     dt    – scalar, desired time‐step for the uniform output grid
%     y0    – 1×3 initial state vector [x0, y0, z0]
%     sigma – (optional) Lorenz σ parameter (default = 10)
%     beta  – (optional) Lorenz β parameter (default = 8/3)
%     rho   – (optional) Lorenz ρ parameter (default = 14)
%
%   Outputs:
%     t_uniform  – 1×M vector of uniformly spaced time points from t_start to t_end
%     y_uniform  – M×3 matrix of state trajectories [x, y, z] at t_uniform
%
%   Behavior:
%     1. Uses ode45 to integrate the Lorenz ODE from t_start to t_end.
%     2. Builds a uniform time vector t_uniform with step size dt.
%     3. Linearly interpolates the nonuniform ODE45 solution onto t_uniform.

    %% Handle default parameter values
    if nargin < 4, sigma = 10;      end  % default σ
    if nargin < 5, beta  = 8/3;     end  % default β
    if nargin < 6, rho   = 14;      end  % default ρ

    %% Define the Lorenz ODE system as a nested function
    function dydt = lorenz_ode(~, y)
        % lorenz_ode  Right‐hand side of the Lorenz equations
        %
        %   dydt = lorenz_ode(t, y)
        %     y      – 3×1 state vector [x; y; z]
        %     dydt   – 3×1 time derivative [dx/dt; dy/dt; dz/dt]
        dydt = zeros(3,1);
        dydt(1) = sigma * (y(2) - y(1));         % dx/dt
        dydt(2) = y(1) * (rho   - y(3)) - y(2);  % dy/dt
        dydt(3) = y(1) * y(2)       - beta*y(3);% dz/dt
    end

    %% 1) Integrate ODE with variable step solver
    %   t_sol and y_sol are the nonuniform output of ode45
    [t_sol, y_sol] = ode45(@lorenz_ode, tspan, y0);

    %% 2) Build uniform time grid for output
    t_uniform = tspan(1):dt:tspan(2);

    %% 3) Interpolate the solution onto the uniform grid
    %   y_uniform is M×3, where M = numel(t_uniform)
    y_uniform = interp1(t_sol, y_sol, t_uniform, 'linear');

end
