close all;
clear all;
clc;

L_x = 100;      % duration of observation of the input signal
fe = 1;         % sampling frequency
Te = 1/fe;      % sampling period
N_x = L_x/Te;   % number of points in signal x
k_x = (0:N_x)'; % time index (vetor coluna)
t_x = k_x*Te;   % time basis

%**************************************************************************
%% 1. Signal x
k1 = 0:N_x/2-1;
x1 = zeros(length(k1),1);
k2 = N_x/2:N_x;
x2 = ones(length(k2),1);

x = [x1; x2];
N_total = length(x);

%% 2. Filtre Gaussien h
sigma = 4;
sigma_h = sigma*Te;
mu_h = 15*Te;
L_h = 30*Te;
t_h = (0:Te:L_h)';
N_h = length(t_h);

% Filtro gaussiano (vetor coluna)
h = (1/(sigma_h*sqrt(2*pi))) * exp(-((t_h-mu_h).^2) / (2*sigma_h^2));

%% 3. Convolution
% Ao invés de 'same', fazemos a convolução completa e extraímos a parte válida
% para que o modelo matricial/circulante funcione com a FFT.
y_full = conv(x,h);
y_nb = y_full(N_h:N_h+N_total-1);

%% 4. Add noise
snr = 30; %dB
sigmab2 = 10^(-snr/10)*mean(y_nb.^2);
sigmab = sqrt(sigmab2);
w = randn(size(y_nb)) * sigmab; %noise
y = y_nb + w;

%==========================================================================
%% 2.1 Moindres Carrés (Rapide via FFT)

% Construção da aproximação circulante do filtro h
c_h = [h(end); zeros(N_total-N_h,1); h(1:end-1)];
lambda_h = fft(c_h);
y_fft = fft(y);

x_hat = y_fft ./ lambda_h;
x_rec = real(ifft(x_hat)); % Uso de real() para ignorar resíduos imaginários numéricos

figure(1);
subplot(2,1,1);
plot(t_x, x, 'b', 'LineWidth', 1.5);
hold on;
plot(t_x, y, 'Color', [0.7 0.7 0.7]); % Mostrando o sinal ruidoso
plot(t_x, x_rec, 'r', 'LineWidth', 1); 
xlabel('temps (s)'); ylabel('amplitude'); 
title('Partie 2.1 : Moindres Carrés (Rapide via FFT)');
legend('Signal original', 'Signal observé (y)', 'Signal reconstruit');
grid on;

%==========================================================================
%% 2.2 Moindres Carrés Régularisés (RLS)

alpha = 0.25;
% Operador de diferença finita de primeira ordem
d = [1; zeros(N_total-2,1); -1];
lambda_d = fft(d); % Corrigido de d1 para d

% Filtro Inverso Regularizado no domínio da frequência
g_RLS = conj(lambda_h) ./ (abs(lambda_h).^2 + alpha * abs(lambda_d).^2);

X_hat_RLS = y_fft .* g_RLS;
x_rec_fft_RLS = real(ifft(X_hat_RLS));

subplot(2,1,2);
plot(t_x, x, 'b', 'LineWidth', 1.5);
hold on;
plot(t_x, x_rec_fft_RLS, 'g', 'LineWidth', 1.5);
xlabel('temps (s)'); ylabel('amplitude'); 
title(['Partie 2.2 : Moindres Carrés Régularisés (\alpha = ' num2str(alpha) ')']);
legend('Signal original', 'Signal reconstruit RLS');
grid on;

%==========================================================================
%% Análise Espectral
figure(2);
freq = (-N_total/2:N_total/2-1)' * fe/N_total;

subplot(1,3,1);
plot(freq, fftshift(20*log10(abs(lambda_h))), 'b', 'LineWidth', 1.5);
xlabel('Fréquence (Hz)'); ylabel('Magnitude (dB)');
title('Réponse du filtre H (Flou)');
grid on;

subplot(1,3,2);
plot(freq, fftshift(20*log10(abs(1./lambda_h))), 'r', 'LineWidth', 1.5);
xlabel('Fréquence (Hz)'); ylabel('Magnitude (dB)');
title('Filtre Inverse (Non régularisé)');
grid on;

subplot(1,3,3);
plot(freq, fftshift(20*log10(abs(g_RLS))), 'g', 'LineWidth', 1.5);
xlabel('Fréquence (Hz)'); ylabel('Magnitude (dB)');
title(['Filtre Inverse Régularisé (\alpha = ' num2str(alpha) ')']);
grid on;