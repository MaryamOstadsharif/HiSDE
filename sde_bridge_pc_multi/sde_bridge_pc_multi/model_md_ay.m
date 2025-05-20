function [Z,Y] = model_md_ay(init_param)

%% Setting
K  = init_param.K;
dt = init_param.dt;
S = init_param.z_var; 
W = init_param.W;
N = init_param.N;

%% Create Lorenz
y0 = [-1 1 -1];
dt = 0.01;
[ts, ys] = lorenz_dynamics([0 (K-1)*dt], dt, y0);
temp = ys;
Y(:,1) = temp(:,1)-mean(temp(:,1));
Y(:,2) = temp(:,2)-mean(temp(:,2));
Y(:,3) = temp(:,3)-mean(temp(:,3));
Y(:,4) = temp(:,2).*temp(:,3)/20;

%% Create Z
Z = zeros(init_param.N,K);
for i=1:K
    lambda = exp(W*Y(i,:)')*dt;
    for d=1:N
        if rand < lambda(d)
            Z(d,i) = 1;
        else
            Z(d,i) = 0;
        end
    end
end

Y= Y';
