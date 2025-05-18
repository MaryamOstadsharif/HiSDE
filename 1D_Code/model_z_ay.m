function Z = model_z_ay(model,init_param)

%% Setting
K  = init_param.K;
dt = init_param.dt;
z_var = init_param.z_var; 

%% Create simulation data
if model == 1
    k  = 1:K;
    Z = exp(cos(2*pi*0.017*dt*k));%+ 0.5 * sin(2*pi*0.037*dt*k) ;
    Z = Z + sqrt(z_var) * randn(size(Z));
end
if model == 2
    k  = 1:K;
    Z  = max(dt * k.^1.3*0.5,10) ;
    Z  = Z + sqrt(z_var) * randn(size(Z));
end
if model == 3
    ts  = dt * (1:K);
    Z = chirp(ts,0.001,8,0.002);
    Z = Z(end:-1:1);
    Z  = 7*exp(-Z)+ sqrt(z_var) * randn(size(Z));
    Z = Z-mean(Z);
end
if model ==4
    %% Create Lorenz
    dt = 0.01;
    y0 = [1 -1 -1];
    [ts, ys] = lorenz_dynamics([0 (K-1)*dt], dt, y0);
    Y = ys;
    
    %% Create Z
    Z = Y(:,3) + sqrt(z_var) * randn(size(Y(:,3)));
end
