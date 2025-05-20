function [t_uniform, y_uniform] = lorenz_dynamics(tspan, dt, y0, sigma, beta, rho)
% LORENZ_DYNAMICS Simulates the Lorenz system.
%
%   [t, y] = lorenz_dynamics(tspan, y0, sigma, beta, rho)
%
% Inputs:
%   tspan - Time span for simulation (e.g., [0 50])
%   y0    - Initial state (e.g., [1, 1, 1])
%   sigma - Parameter sigma (default: 10)
%   beta  - Parameter beta  (default: 8/3)
%   rho   - Parameter rho   (default: 28)
%
% Outputs:
%   t     - Time vector
%   y     - State trajectory (Nx3 matrix: columns are x, y, z)

    % Set default parameters if not provided
    if nargin < 4, sigma = 10; end
    if nargin < 5, beta = 8/3; end
    if nargin < 6, rho = 14; end

    % Lorenz system equations
    function dydt = lorenz_ode(t, y)
        dydt = zeros(3,1);
        dydt(1) = sigma * (y(2) - y(1));
        dydt(2) = y(1) * (rho - y(3)) - y(2);
        dydt(3) = y(1) * y(2) - beta * y(3);
    end

    % Solve the ODE
    [t, y] = ode45(@lorenz_ode, tspan, y0);
    % Step 2: Create uniform time vector
    t_uniform = tspan(1):dt:tspan(2);

    % Step 3: Interpolate state onto uniform time grid
    y_uniform = interp1(t, y, t_uniform, 'linear');  % or 'spline' if smoother

end
