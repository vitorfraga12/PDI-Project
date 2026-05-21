close all;
clear all;
clc;

rng(0);

%% 4.1 - 1D signal

N = 101;
t = (0:N-1)';

x = zeros(N,1);
x(51:end) = 1;

Te = 1;
sigma_h = 5*Te;
t_h = (-15:15)'*Te;
h = exp(-(t_h.^2)/(2*sigma_h^2));
h = h/sum(h);

y0 = conv(x,h,'same');

SNR = 30;
sigma2 = 10^(-SNR/10)*mean(y0.^2);
y = y0 + sqrt(sigma2)*randn(size(y0));

alpha = 1e-2;
T = 5e-3;
x0 = y;

opt1 = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','off', ...
    'SpecifyObjectiveGradient',true, ...
    'MaxIterations',300, ...
    'MaxFunctionEvaluations',5000);

[x_ep,~,~,out1] = fminunc(@(z) cost_1d(z,y,h,alpha,T), x0, opt1);
err_1d = norm(x-x_ep)^2;

fprintf('1D: alpha = %.1e, T = %.1e, error = %.4e, iterations = %d\n', ...
    alpha,T,err_1d,out1.iterations);

figure;
plot(t,x,'k','LineWidth',1.4);
hold on;
plot(t,y,'Color',[0.7 0.7 0.7]);
plot(t,x_ep,'r','LineWidth',1.2);
hold off;
grid on;
xlabel('time (s)');
ylabel('amplitude');
title('1D edge-preserving restoration');
legend('original x','blurred noisy y','edge-preserving');

%% 4.2 - image

x = double(imread('cameraman.tif'));
x = x/max(x(:));
x = imresize(x,[64 64]);
[N1,N2] = size(x);

h = ones(5,5)/25;
y0 = conv2(x,h,'same');

SNR = 30;
sigma2 = 10^(-SNR/10)*mean(y0(:).^2);
y = y0 + sqrt(sigma2)*randn(size(y0));

% quadratic regularization, used only for comparison
alpha_l2 = 3.162278e-1;
d1 = [0 -1 0; -1 4 -1; 0 -1 0];
H = psf2otf_local(h,[N1 N2]);
D = psf2otf_local(d1,[N1 N2]);
Y = fft2(y);
epsi = 1e-10;

G_l2 = conj(H)./(abs(H).^2 + alpha_l2*abs(D).^2 + epsi);
x_l2 = real(ifft2(G_l2.*Y));

% edge-preserving regularization
alpha = 3e-3;
T = 1e-2;
x0 = x_l2(:);      % a good starting point, already computed above

opt2 = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','off', ...
    'SpecifyObjectiveGradient',true, ...
    'MaxIterations',200, ...
    'MaxFunctionEvaluations',20000);

[z_ep,~,~,out2] = fminunc(@(z) cost_2d(z,y,h,alpha,T,N1,N2), x0, opt2);
x_ep = reshape(z_ep,N1,N2);

err_l2 = norm(x(:)-x_l2(:))^2;
err_ep = norm(x(:)-x_ep(:))^2;

fprintf('Image L2: alpha = %.1e, error = %.4e\n', alpha_l2, err_l2);
fprintf('Image EP: alpha = %.1e, T = %.1e, error = %.4e, iterations = %d\n', ...
    alpha,T,err_ep,out2.iterations);

figure;
subplot(2,2,1);
imagesc(x); caxis([0 1]); colormap gray; axis image off;
title('original');

subplot(2,2,2);
imagesc(y); caxis([0 1]); colormap gray; axis image off;
title('blurred + noisy');

subplot(2,2,3);
imagesc(x_l2); caxis([0 1]); colormap gray; axis image off;
title(['quadratic, \alpha = ',num2str(alpha_l2,'%.1e')]);

subplot(2,2,4);
imagesc(x_ep); caxis([0 1]); colormap gray; axis image off;
title(['edge-preserving, \alpha = ',num2str(alpha,'%.1e'),', T = ',num2str(T,'%.1e')]);

%% functions

function [F,g] = cost_1d(x,y,h,alpha,T)

    r = conv(x,h,'same') - y;
    F = sum(r.^2);
    g = 2*conv(r,flipud(h),'same');

    d = x(1:end-1) - x(2:end);
    F = F + alpha*sum(sqrt(d.^2 + T^2) - T);

    p = d./sqrt(d.^2 + T^2);
    gr = zeros(size(x));
    gr(1:end-1) = gr(1:end-1) + p;
    gr(2:end) = gr(2:end) - p;

    g = g + alpha*gr;

end

function [F,g] = cost_2d(z,y,h,alpha,T,N1,N2)

    X = reshape(z,N1,N2);

    R = conv2(X,h,'same') - y;
    F = sum(R(:).^2);
    G = 2*conv2(R,rot90(h,2),'same');

    dx = X(:,1:end-1) - X(:,2:end);
    dy = X(1:end-1,:) - X(2:end,:);

    F = F + alpha*(sum(sqrt(dx(:).^2 + T^2) - T) + ...
                   sum(sqrt(dy(:).^2 + T^2) - T));

    px = dx./sqrt(dx.^2 + T^2);
    py = dy./sqrt(dy.^2 + T^2);

    Gr = zeros(N1,N2);
    Gr(:,1:end-1) = Gr(:,1:end-1) + px;
    Gr(:,2:end) = Gr(:,2:end) - px;
    Gr(1:end-1,:) = Gr(1:end-1,:) + py;
    Gr(2:end,:) = Gr(2:end,:) - py;

    G = G + alpha*Gr;
    g = G(:);

end

function otf = psf2otf_local(psf,outSize)

    s = size(psf);
    aux = zeros(outSize);
    aux(1:s(1),1:s(2)) = psf;
    aux = circshift(aux,[-floor(s(1)/2),-floor(s(2)/2)]);
    otf = fft2(aux);

end
