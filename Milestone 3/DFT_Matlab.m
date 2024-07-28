%%---------------- DFT MATLAB IMPLEMENTATION ---------------------
%% value initialization
N = 2^15;
x = zeros(N, 1);

for i = 1:N
    x(i) = i;
end

xr = zeros(N, 1);
xi = zeros(N, 1);

%% setting up clock
t0 = datetime("now");

%% calcs
for k = 0:1:(size(x)-1)
    for n = 0:1:(size(x)-1)
        theta = (2 * pi * k * n)/N;
        xr(k+1) = xr(k+1) + (x(n+1) * cos(theta));
        xi(k+1) = xi(k+1) - (x(n+1) * sin(theta));
    end
end

%% time after calcs
t1 = datetime("now");
ms = milliseconds(t1 - t0);

%for i = 1:size(x)
%   fprintf("%f + j(%f)\n", xr(i), xi(i))
%end
fprintf('Time elapsed: %fms\n', ms)


