%%---------------- MATLAB IMPLEMENTATION ---------------------
%% value initialization
N = 2^14;
fprintf("Number of Elements: %d\n", N);
x = zeros(N, 1);
for i = 0:N-1
    x(i+1) = i;
end
loop = 5;   % change to 1 for correctness checking
fprintf("Number of Loops: %d\n", loop);


%% calcs for DFT

total_time = 0;
for i = 1:loop
    xr = zeros(N, 1);
    xi = zeros(N, 1);
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
end
average_time = total_time/5;

fprintf('Average time elapsed (DFT): %fms\n', average_time)

%% calcs for DFT built-in
total_time = 0;
for i = 1:loop
    t0 = datetime("now");
    temp = fft(x);
    t1 = datetime("now");
    bixr = real(temp); %bi for built-in
    bixi = imag(temp); %bi for built-in
    ms = milliseconds(t1 - t0);
    total_time = total_time + ms;
end
average_time = total_time/5;


%for i = 1:size(x)
%   fprintf("%f + j(%f)\n", xr(i), xi(i))
%end
fprintf('Average time elapsed (DFT built-in): %fms\n', average_time)


%% output results for DFT correction checking, used for 2^10 for fast output
fileID = fopen('sample.txt','w');
for i = 1:1:size(x)
    fprintf(fileID,'%f %f\n',xr(i),xi(i));
end
for i = 1:1:size(x)
    fprintf(fileID,'%f %f\n',bixr(i),bixi(i));
end
fclose(fileID);

%% calcs for IDFT
total_time = 0;
for i = 1:loop
    y = zeros(N, 1);
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
end
average_time = total_time/5;

%for i = 1:size(x)
%   fprintf("%f\n", y(i))
%end
fprintf('Average time elapsed (IDFT): %fms\n', average_time)

%% calcs for IDFT built-in
total_time = 0;
for i = 1:loop
    y = zeros(N, 1);
    t0 = datetime("now");
    biy = ifft(temp); %bi for built-in
    t1 = datetime("now");
    ms = milliseconds(t1 - t0);
    total_time = total_time + ms;
end
average_time = total_time/5;
fprintf('Average time elapsed (IDFT built-in): %fms\n', average_time)
