close all;
clear all;
clc;

rng(0);

L_x = 100;      % duration of observation of the input signal
fe = 1;         % sampling frequency
Te = 1/fe;      % sampling period
N_x_param = L_x/Te;
k_x = (0:N_x_param)'; 
t_x = k_x*Te;


%% 1. signal x

k1 = 0:N_x_param/2-1;
x1 = zeros(length(k1),1);

k2 = N_x_param/2:N_x_param;
x2 = ones(length(k2),1);

x = [x1; x2];
N_x = length(x);

%% 2. filtre h

sigma_h = 4*Te;       
mu_h = 15*Te;
L_h = 30*Te;
t_h = (0:Te:L_h)';
N_h = length(t_h);

h = (1/(sigma_h*sqrt(2*pi))) * exp(-((t_h-mu_h).^2)/(2*sigma_h^2));

%% 3. conv

y_full = conv(x,h);

% Section 2 says y0[Nh], ..., y0[Nh + Nx - 1]
y_nb = y_full(N_h:N_h+N_x-1);

%% 4. add AWGN

SNR = 30; % dB

sigmab2 = 10^(-SNR/10)*mean(y_nb.^2);
sigmab = sqrt(sigmab2);

w = randn(size(y_nb))*sigmab;
y = y_nb + w;

measured_snr = 10*log10(mean(y_nb.^2)/var(w));


%% 5. toeplitz

first_col = zeros(N_x,1);
first_col(1) = h(end);

first_row = zeros(1,N_x);
first_row(1:N_h) = h(end:-1:1);

H = toeplitz(first_col, first_row);

x_ls_matrix = H\y;

%% 2.1 least squares

c_h = [h(end); zeros(N_x-N_h,1); h(1:end-1)];

lambda_h = fft(c_h, N_x);
Y = fft(y, N_x);

X_ls_fft = Y ./ lambda_h;
x_ls_fft = real(ifft(X_ls_fft, N_x));

%% 2.2 regularized least squares

d = [1; zeros(N_x-2,1); -1];
lambda_d = fft(d, N_x);

log10_alphas = -7:0.1:2;

err_rls = zeros(size(log10_alphas));
x_rls_all = zeros(N_x,length(log10_alphas));

% to find the best alpha as possible
for i_alpha = 1:length(log10_alphas)

    alpha = 10^log10_alphas(i_alpha);

    g_RLS = conj(lambda_h) ./ ...
        (abs(lambda_h).^2 + alpha*abs(lambda_d).^2);

    X_rls_fft = g_RLS .* Y;
    x_rls_all(:,i_alpha) = real(ifft(X_rls_fft, N_x));

    err_rls(i_alpha) = norm(x - x_rls_all(:,i_alpha))^2;

end

[best_err, best_idx] = min(err_rls);
alpha_opt = 10^log10_alphas(best_idx);

x_rls_fft = x_rls_all(:,best_idx);


%% 6. erros 

err_ls_matrix = norm(x - x_ls_matrix)^2;
err_ls_fft = norm(x - x_ls_fft)^2;
err_rls_fft = norm(x - x_rls_fft)^2;

fprintf('Measured SNR = %.4f dB\n', measured_snr);
fprintf('Noise variance = %.4e\n', sigmab2);
fprintf('Matrix LS error = %.4e\n', err_ls_matrix);
fprintf('Fast LS error = %.4e\n', err_ls_fft);
fprintf('Best fast RLS alpha = %.4e\n', alpha_opt);
fprintf('Best fast RLS error = %.4e\n', err_rls_fft);


%% 7. Figures

figure(1);

subplot(2,2,1);
plot(t_x, x, 'k', 'LineWidth', 1.2);
hold on;
plot(t_x, y, 'Color', [0.7 0.7 0.7]);
hold off;
xlabel('time (s)');
ylabel('amplitude');
title('Input signal and noisy data');
legend('x','y');
grid on;

subplot(2,2,2);
plot(t_x, x, 'k', 'LineWidth', 1.2);
hold on;
plot(t_x, x_ls_matrix, 'b');
plot(t_x, x_ls_fft, 'r');
hold off;
xlabel('time (s)');
ylabel('amplitude');
title('Least squares deconvolution');
legend('x','matrix LS','fast LS');
grid on;

subplot(2,2,3);
plot(log10_alphas, 10*log10(err_rls), 'LineWidth', 1.2);
xlabel('log10(alpha)');
ylabel('error energy (dB)');
title('Regularized error sweep');
grid on;

subplot(2,2,4);
plot(t_x, x, 'k', 'LineWidth', 1.2);
hold on;
plot(t_x, x_rls_fft, 'r', 'LineWidth', 1.2);
hold off;
xlabel('time (s)');
ylabel('amplitude');
title(['Fast regularized LS, alpha = ' num2str(alpha_opt)]);
legend('x','fast RLS');
grid on;

%% 8. Spectral analysis

figure(2);

freq = (-N_x/2:N_x/2-1)'*fe/N_x;

subplot(1,3,1);
plot(freq, fftshift(20*log10(abs(lambda_h))), 'LineWidth', 1.2);
xlabel('frequency (Hz)');
ylabel('dB');
title('Blur frequency response');
grid on;

subplot(1,3,2);
plot(freq, fftshift(20*log10(abs(1./lambda_h))), 'LineWidth', 1.2);
xlabel('frequency (Hz)');
ylabel('dB');
title('Inverse filter response');
grid on;

g_RLS_opt = conj(lambda_h) ./ ...
    (abs(lambda_h).^2 + alpha_opt*abs(lambda_d).^2);

subplot(1,3,3);
plot(freq, fftshift(20*log10(abs(g_RLS_opt))), 'LineWidth', 1.2);
xlabel('frequency (Hz)');
ylabel('dB');
title('Regularized inverse response');
grid on;