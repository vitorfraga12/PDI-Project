%-------------------------------------------------------------------------
% Part 3 - Image deconvolution by quadratic regularization
% Noise variation with several SNR values
%-------------------------------------------------------------------------

close all;
clear all;
clc;

rng(0);

%% ------------------------------------------------------------------------
% 1. Load original image
% -------------------------------------------------------------------------

x = double(imread('cameraman.tif'));

% Normalize image between 0 and 1
x = x / max(x(:));

[N_rows, N_cols] = size(x);

if N_rows ~= N_cols
    error('The image must be square for this project.');
end

N = N_rows;

%% ------------------------------------------------------------------------
% 2. Direct problem: blur image with 5x5 averaging filter
% -------------------------------------------------------------------------

h1 = (1/25) * ones(5,5);

% Project asks for conv2 in 'same' mode
y0 = conv2(x, h1, 'same');

%% ------------------------------------------------------------------------
% 3. Frequency response of blur filter
% -------------------------------------------------------------------------

lambda_h = psf2otf_local(h1, [N N]);

epsi = 1e-10;

%% ------------------------------------------------------------------------
% 4. Regularization operator d1
% -------------------------------------------------------------------------

d1 = [ 0 -1  0;
      -1  4 -1;
       0 -1  0];

lambda_d = psf2otf_local(d1, [N N]);

%% ------------------------------------------------------------------------
% 5. Vary noise level
% -------------------------------------------------------------------------

SNR_list = [50 40 30 20 10];

log10_alphas = -8:0.1:0;

alpha_opt_list = zeros(size(SNR_list));
err_ls_list = zeros(size(SNR_list));
err_rls_list = zeros(size(SNR_list));
noise_var_list = zeros(size(SNR_list));

% Store images for display
y_all = cell(length(SNR_list),1);
x_ls_all = cell(length(SNR_list),1);
x_rls_all = cell(length(SNR_list),1);

for i_snr = 1:length(SNR_list)

    SNR = SNR_list(i_snr);

    %% Add Gaussian noise

    sigma_w2 = 10^(-SNR/10) * mean(y0(:).^2);
    sigma_w = sqrt(sigma_w2);

    w = sigma_w * randn(size(y0));
    y = y0 + w;

    measured_snr = 10*log10(mean(y0(:).^2) / var(w(:)));

    noise_var_list(i_snr) = sigma_w2;

    fprintf('\nSNR asked = %.1f dB\n', SNR);
    fprintf('Measured SNR = %.4f dB\n', measured_snr);
    fprintf('Noise variance = %.4e\n', sigma_w2);

    Y = fft2(y);

    %% --------------------------------------------------------------------
    % Least-squares reconstruction
    % ---------------------------------------------------------------------

    X_ls = Y ./ (lambda_h + epsi);
    x_ls = real(ifft2(X_ls));

    err_ls = norm(x(:) - x_ls(:))^2;

    fprintf('LS error = %.4e\n', err_ls);

    %% --------------------------------------------------------------------
    % Regularized least-squares reconstruction
    % ---------------------------------------------------------------------

    err_alpha = zeros(size(log10_alphas));
    x_alpha_all = zeros(N,N,length(log10_alphas));

    for i_alpha = 1:length(log10_alphas)

        alpha = 10^log10_alphas(i_alpha);

        g_RLS = conj(lambda_h) ./ ...
            (abs(lambda_h).^2 + alpha*abs(lambda_d).^2 + epsi);

        X_rls = g_RLS .* Y;
        x_rls_temp = real(ifft2(X_rls));

        x_alpha_all(:,:,i_alpha) = x_rls_temp;

        err_alpha(i_alpha) = norm(x(:) - x_rls_temp(:))^2;

    end

    [best_err, best_idx] = min(err_alpha);

    alpha_opt = 10^log10_alphas(best_idx);
    x_rls = x_alpha_all(:,:,best_idx);

    fprintf('Best alpha = %.4e\n', alpha_opt);
    fprintf('RLS error = %.4e\n', best_err);

    %% Store results

    alpha_opt_list(i_snr) = alpha_opt;
    err_ls_list(i_snr) = err_ls;
    err_rls_list(i_snr) = best_err;

    y_all{i_snr} = y;
    x_ls_all{i_snr} = x_ls;
    x_rls_all{i_snr} = x_rls;

end

%% ------------------------------------------------------------------------
% 6. Display results for each noise level
% -------------------------------------------------------------------------

for i_snr = 1:length(SNR_list)

    SNR = SNR_list(i_snr);

    figure;

    subplot(2,2,1);
    imagesc(x);
    colormap gray;
    axis image off;
    title('Original image x');

    subplot(2,2,2);
    imagesc(y_all{i_snr});
    colormap gray;
    axis image off;
    title(['Blurred noisy image, SNR = ', num2str(SNR), ' dB']);

    subplot(2,2,3);
    imagesc(x_ls_all{i_snr});
    colormap gray;
    axis image off;
    title('Least-squares reconstruction');

    subplot(2,2,4);
    imagesc(x_rls_all{i_snr});
    colormap gray;
    axis image off;
    title(['Regularized LS, \alpha = ', num2str(alpha_opt_list(i_snr))]);

end

%% ------------------------------------------------------------------------
% 7. Plot alpha evolution with noise level
% -------------------------------------------------------------------------

figure;

semilogy(SNR_list, alpha_opt_list, 'o-', 'LineWidth', 1.2);
xlabel('SNR in dB');
ylabel('Optimal \alpha');
title('Evolution of optimal \alpha with noise level');
grid on;

%% ------------------------------------------------------------------------
% 8. Plot reconstruction errors
% -------------------------------------------------------------------------

figure;

semilogy(SNR_list, err_ls_list, 'o-', 'LineWidth', 1.2);
hold on;
semilogy(SNR_list, err_rls_list, 's-', 'LineWidth', 1.2);
hold off;

xlabel('SNR in dB');
ylabel('Reconstruction error');
title('LS and RLS errors for different noise levels');
legend('Least squares', 'Regularized least squares');
grid on;

%% ------------------------------------------------------------------------
% 9. Spectral analysis using last SNR case
% -------------------------------------------------------------------------

freq = (-N/2:N/2-1)/N;

g_RLS_opt = conj(lambda_h) ./ ...
    (abs(lambda_h).^2 + alpha_opt_list(end)*abs(lambda_d).^2 + epsi);

figure;

subplot(1,3,1);
imagesc(freq, freq, fftshift(20*log10(abs(lambda_h) + epsi)));
axis image;
colorbar;
xlabel('\nu_x');
ylabel('\nu_y');
title('Blur frequency response');

subplot(1,3,2);
imagesc(freq, freq, fftshift(20*log10(abs(1./(lambda_h + epsi)))));
axis image;
colorbar;
xlabel('\nu_x');
ylabel('\nu_y');
title('Inverse filter response');

subplot(1,3,3);
imagesc(freq, freq, fftshift(20*log10(abs(g_RLS_opt) + epsi)));
axis image;
colorbar;
xlabel('\nu_x');
ylabel('\nu_y');
title('Regularized inverse response');

%% ------------------------------------------------------------------------
% Local function: PSF to OTF
% -------------------------------------------------------------------------

function otf = psf2otf_local(psf, outSize)

    psfSize = size(psf);

    padded = zeros(outSize);

    padded(1:psfSize(1), 1:psfSize(2)) = psf;

    shift_r = -floor(psfSize(1)/2);
    shift_c = -floor(psfSize(2)/2);

    padded = circshift(padded, [shift_r, shift_c]);

    otf = fft2(padded);

end