%-------------------------------------------------------------------------
% Part 3 - Image deconvolution by quadratic regularization
% Version where alpha varies from very small to very large values
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

% The project asks to use conv2 in 'same' mode
y0 = conv2(x, h1, 'same');

%% ------------------------------------------------------------------------
% 3. Add Gaussian white noise with fixed SNR
% -------------------------------------------------------------------------

SNR = 30; % dB

sigma_w2 = 10^(-SNR/10) * mean(y0(:).^2);
sigma_w = sqrt(sigma_w2);

w = sigma_w * randn(size(y0));
y = y0 + w;

measured_snr = 10*log10(mean(y0(:).^2) / var(w(:)));

fprintf('Measured SNR = %.4f dB\n', measured_snr);
fprintf('Noise variance = %.4e\n', sigma_w2);

%% ------------------------------------------------------------------------
% 4. FFT of blur filter
% -------------------------------------------------------------------------

lambda_h = psf2otf_local(h1, [N N]);

Y = fft2(y);

epsi = 1e-10;

%% ------------------------------------------------------------------------
% 5. Least-squares reconstruction
% -------------------------------------------------------------------------

X_ls = Y ./ (lambda_h + epsi);
x_ls = real(ifft2(X_ls));

err_ls = norm(x(:) - x_ls(:))^2;

fprintf('Least-squares reconstruction error = %.4e\n', err_ls);

%% ------------------------------------------------------------------------
% 6. Regularization operator d1
%
% Discrete Laplacian:
%
%       [ 0  -1   0
%        -1   4  -1
%         0  -1   0 ]
%
% -------------------------------------------------------------------------

d1 = [ 0 -1  0;
      -1  4 -1;
       0 -1  0];

lambda_d = psf2otf_local(d1, [N N]);

%% ------------------------------------------------------------------------
% 7. Alpha variation
%
% Very small alpha  -> close to inverse filtering, noise amplification
% Medium alpha      -> better compromise
% Very large alpha  -> oversmoothing
% -------------------------------------------------------------------------

alpha_list = [1e-10 1e-8 1e-6 1e-4 1e-3 1e-2 ...
              1e-1 1 10 100 1e3 1e4];

x_rls_all = cell(length(alpha_list),1);
err_rls_list = zeros(size(alpha_list));

for i_alpha = 1:length(alpha_list)

    alpha = alpha_list(i_alpha);

    g_RLS = conj(lambda_h) ./ ...
        (abs(lambda_h).^2 + alpha*abs(lambda_d).^2 + epsi);

    X_rls = g_RLS .* Y;

    x_rls = real(ifft2(X_rls));

    x_rls_all{i_alpha} = x_rls;

    err_rls_list(i_alpha) = norm(x(:) - x_rls(:))^2;

    fprintf('alpha = %.1e, RLS error = %.4e\n', ...
        alpha, err_rls_list(i_alpha));

end

%% ------------------------------------------------------------------------
% 8. Find best alpha among tested values
% -------------------------------------------------------------------------

[best_err, best_idx] = min(err_rls_list);
alpha_best = alpha_list(best_idx);
x_rls_best = x_rls_all{best_idx};

fprintf('\nBest alpha among tested values = %.4e\n', alpha_best);
fprintf('Best RLS error = %.4e\n', best_err);

%% ------------------------------------------------------------------------
% 9. Display original, blurred noisy, and LS reconstruction
% -------------------------------------------------------------------------

figure;

subplot(1,3,1);
imagesc(x);
colormap gray;
axis image off;
title('Original image x');

subplot(1,3,2);
imagesc(y);
colormap gray;
axis image off;
title(['Blurred noisy image, SNR = ', num2str(SNR), ' dB']);

subplot(1,3,3);
imagesc(x_ls);
colormap gray;
axis image off;
title('Least-squares reconstruction');

%% ------------------------------------------------------------------------
% 10. Display RLS results for different alpha values
% -------------------------------------------------------------------------

figure;

for i_alpha = 1:length(alpha_list)

    subplot(3,4,i_alpha);

    imagesc(x_rls_all{i_alpha});
    colormap gray;
    axis image off;

    title(['\alpha = ', num2str(alpha_list(i_alpha),'%.0e')]);

end

sgtitle('Regularized LS reconstruction for different alpha values');

%% ------------------------------------------------------------------------
% 11. Display best RLS result separately
% -------------------------------------------------------------------------

figure;

subplot(1,3,1);
imagesc(x);
colormap gray;
axis image off;
title('Original image x');

subplot(1,3,2);
imagesc(y);
colormap gray;
axis image off;
title('Blurred noisy image y');

subplot(1,3,3);
imagesc(x_rls_best);
colormap gray;
axis image off;
title(['Best RLS, \alpha = ', num2str(alpha_best,'%.1e')]);

%% ------------------------------------------------------------------------
% 12. Error as function of alpha
% -------------------------------------------------------------------------

figure;

loglog(alpha_list, err_rls_list, 'o-', 'LineWidth', 1.2);
xlabel('\alpha');
ylabel('Reconstruction error ||x - x_{RLS}||^2');
title('Effect of \alpha on regularized reconstruction');
grid on;

%% ------------------------------------------------------------------------
% 13. Spectral analysis for each alpha
% -------------------------------------------------------------------------

freq = (-N/2:N/2-1)/N;

figure;

for i_alpha = 1:length(alpha_list)

    alpha = alpha_list(i_alpha);

    g_RLS = conj(lambda_h) ./ ...
        (abs(lambda_h).^2 + alpha*abs(lambda_d).^2 + epsi);

    subplot(3,4,i_alpha);

    imagesc(freq, freq, fftshift(20*log10(abs(g_RLS) + epsi)));
    axis image;
    colorbar;

    xlabel('\nu_x');
    ylabel('\nu_y');
    title(['\alpha = ', num2str(alpha,'%.0e')]);

end

sgtitle('Regularized inverse filter response for different alpha values');
%% ------------------------------------------------------------------------
% Frequency response of the averaging filter h1
% -------------------------------------------------------------------------

freq = (-N/2:N/2-1)/N;

figure;

imagesc(freq, freq, fftshift(20*log10(abs(lambda_h) + epsi)));
axis image;
colorbar;
colormap jet;

xlabel('\nu_x');
ylabel('\nu_y');
title('Frequency response of the 5x5 averaging filter H_1(\nu_x,\nu_y)');
%% ------------------------------------------------------------------------
% Local function: PSF to OTF
%
% This avoids needing the Image Processing Toolbox function psf2otf.
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