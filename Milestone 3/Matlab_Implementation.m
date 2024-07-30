%%---------------- MATLAB IMPLEMENTATION ---------------------
%% value initialization
N = 2^10;
fprintf("Number of Elements: %d\n", N);
x = zeros(N, 1);
y = zeros(N, 1);
for i = 0:N-1
    x(i+1) = i;
end

xr = zeros(N, 1);
xi = zeros(N, 1);
total_time = 0;

%% calcs for DFT
loop = 1;   % change to 1 for correctness checking
fprintf("Number of Loops: %d\n", loop);
for i = 1:loop
    t0 = datetime("now");
    for k = 0:1:(size(x)-1)
        for n = 0:1:(size(x)-1)
            theta = (2 * pi * k * n)/N;
            xr(k+1) = xr(k+1) + (x(n+1) * cos(theta));
            xi(k+1) = xi(k+1) - (x(n+1) * sin(theta));
        end
    end
    t1 = datetime("now");
    ms = milliseconds(t1 - t0);
    total_time = total_time + ms;
    xr = zeros(N, 1);
    xi = zeros(N, 1);
end
average_time = total_time/5;

%for i = 1:size(x)
%   fprintf("%f + j(%f)\n", xr(i), xi(i))
%end
fprintf('Average time elapsed (DFT): %fms\n', average_time)


%% output results for DFT correction checking, used for 2^10 for fast output
% fileID = fopen('sample.txt','w');
% for i = 1:1:size(x)
%     fprintf(fileID,'%f %f\n',xr(i),xi(i));
% end
% fclose(fileID);

%% calcs for IDFT
total_time = 0;
for i = 1:loop
    t0 = datetime("now");
    for k = 0:1:(size(x)-1)
        for n = 0:1:(size(x)-1)
            theta = (2 * pi * k * n)/N;
            y(k+1) = y(k+1)+ xr(k+1)  * cos(theta) - xi(k+1)  * sin(theta);
        end
        y(k+1) = y(k+1)/N;
    end
    t1 = datetime("now");
    ms = milliseconds(t1 - t0);
    total_time = total_time + ms;
    y = zeros(N, 1);
end
average_time = total_time/5;

%for i = 1:size(x)
%   fprintf("%f + j(%f)\n", xr(i), xi(i))
%end
fprintf('Average time elapsed (IDFT): %fms\n', average_time)
