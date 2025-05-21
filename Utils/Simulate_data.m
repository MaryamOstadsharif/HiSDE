function [Z,Y] = model_z_ay(init_param)

%% Setting
K  = init_param.K;
dt = init_param.dt;
S = init_param.z_var; 
W = init_param.W;

%% Create Lorenz
y0 = [-1 1 -1];
dt = 0.01;
[ts, ys] = lorenz_dynamics([0 (K-1)*dt], dt, y0);
Y = ys;
Y(:,1) = Y(:,1)-mean(Y(:,1));
Y(:,2) = Y(:,2)-mean(Y(:,2));
Y(:,3) = Y(:,3)-mean(Y(:,3));

%% Create Z
Z = zeros(init_param.N,K);
for i=1:K
    Z(:,i) = mvnrnd(W*Y(i,:)',S);
end
Y= Y';
