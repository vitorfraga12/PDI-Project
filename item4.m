close all;
clear all;
clc;
rng(0);

%% ========================================================================
%% 4.1 - 1D signal
%% ========================================================================
N = 101;
t = (0:N-1)';
x_1d = zeros(N,1);
x_1d(51:end) = 1;
Te = 1;
sigma_h = 5*Te;
t_h = (-15:15)'*Te;
h_1d = exp(-(t_h.^2)/(2*sigma_h^2));
h_1d = h_1d/sum(h_1d);
y0_1d = conv(x_1d,h_1d,'same');
SNR = 30;
sigma2_1d = 10^(-SNR/10)*mean(y0_1d.^2);
y_1d = y0_1d + sqrt(sigma2_1d)*randn(size(y0_1d));
alpha_1d = 1e-2;
T_1d = 5e-3;
x0_1d = y_1d;

opt1 = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','off', ...
    'SpecifyObjectiveGradient',true, ...
    'MaxIterations',300, ...
    'MaxFunctionEvaluations',5000);

[x_ep_1d,~,~,out1] = fminunc(@(z) cost_1d(z,y_1d,h_1d,alpha_1d,T_1d), x0_1d, opt1);
err_1d = norm(x_1d-x_ep_1d)^2;

fprintf('1D: alpha = %.1e, T = %.1e, error = %.4e, iterations = %d\n', ...
    alpha_1d, T_1d, err_1d, out1.iterations);

figure('Name', '1D Signal Restoration');
plot(t,x_1d,'k','LineWidth',1.4);
hold on;
plot(t,y_1d,'Color',[0.7 0.7 0.7]);
plot(t,x_ep_1d,'r','LineWidth',1.2);
hold off;
grid on;
xlabel('time (s)');
ylabel('amplitude');
title('1D edge-preserving restoration');
legend('original x','blurred noisy y','edge-preserving');

%% ========================================================================
%% 4.2 - image (4x4 Grid Search with 3D Error Surface Plot)
%% ========================================================================
x = double(imread('cameraman.tif'));
x = x/max(x(:));
x = imresize(x,[48 48]); 
[N1,N2] = size(x);

h = ones(5,5)/25;
y0 = conv2(x,h,'same');
SNR = 30;
sigma2 = 10^(-SNR/10)*mean(y0(:).^2);
y = y0 + sqrt(sigma2)*randn(size(y0));

% --- 4x4 Grid Search Parameters ---
alpha_grid = logspace(-3, -1.5, 4); 
T_grid     = logspace(-3, -1, 4); 

% Matriz para armazenar os erros e gerar o gráfico 3D posterior
error_matrix = zeros(length(alpha_grid), length(T_grid));

% Pre-computation for L2 baseline and initialization
alpha_l2 = 3.162278e-1;
d1 = [0 -1 0; -1 4 -1; 0 -1 0];
H = psf2otf_local(h,[N1 N2]);
D = psf2otf_local(d1,[N1 N2]);
Y = fft2(y);
epsi = 1e-10;
G_l2 = conj(H)./(abs(H).^2 + alpha_l2*abs(D).^2 + epsi);
x_l2 = real(ifft2(G_l2.*Y));
x0 = x_l2(:); 

opt2 = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'Display','off', ...
    'SpecifyObjectiveGradient',true, ...
    'MaxIterations',350, ... 
    'MaxFunctionEvaluations',15000);

% Reference Images Figure
figure('Name', 'Reference Images');
subplot(1,3,1); imagesc(x); caxis([0 1]); colormap gray; axis image off; title('Original (48x48)');
subplot(1,3,2); imagesc(y); caxis([0 1]); colormap gray; axis image off; title('Blurred + Noisy');
subplot(1,3,3); imagesc(x_l2); caxis([0 1]); colormap gray; axis image off; title('Quadratic (L2)');

% Grid Search Results Figure (Matrix of 16 Subplots)
figure('Name', '4x4 Grid Search: Edge-Preserving Parameters');
num_rows = length(alpha_grid);
num_cols = length(T_grid);
idx_plot = 1;

best_error = Inf;
best_x_ep = zeros(N1,N2);
best_alpha = 0;
best_T = 0;
best_row = 0;
best_col = 0;

fprintf('\n--- Starting 4x4 Grid Search (16 combinations) ---\n');
for i = 1:num_rows
    for j = 1:num_cols
        alpha_atual = alpha_grid(i);
        T_atual = T_grid(j);
        
        tic;
        [z_ep,~,~,out2] = fminunc(@(z) cost_2d(z,y,h,alpha_atual,T_atual,N1,N2), x0, opt2);
        x_ep = reshape(z_ep,N1,N2);
        elapsed_time = toc;
        
        % Calculando o erro médio quadrático (MSE)
        current_error = mean((x(:) - x_ep(:)).^2);
        error_matrix(i,j) = current_error; % Salva o erro na matriz para o plot 3D
        
        fprintf('Grid [%d/16]: alpha = %.1e, T = %.1e | Error = %.4e | Iterations = %d | Time = %.2fs\n', ...
            idx_plot, alpha_atual, T_atual, current_error, out2.iterations, elapsed_time);
        
        if current_error < best_error
            best_error = current_error;
            best_x_ep = x_ep;
            best_alpha = alpha_atual;
            best_T = T_atual;
            best_row = i;
            best_col = j;
        end
        
        subplot(num_rows, num_cols, idx_plot);
        imagesc(x_ep); 
        caxis([0 1]); 
        colormap gray; 
        axis image off;
        title(['\alpha=', num2str(alpha_atual,'%.1e'), ' | T=', num2str(T_atual,'%.1e')]);
        
        idx_plot = idx_plot + 1;
    end
end
fprintf('--- Grid Search Completed ---\n');

%% ========================================================================
%% NOVO: Gráfico 3D da Superfície de Erros (MSE)
%% ========================================================================
figure('Name', '3D Error Surface');
% Criamos uma malha com os valores em escala logarítmica para correta visualização
[X_mesh, Y_mesh] = meshgrid(log10(T_grid), log10(alpha_grid));

% Plota a superfície 3D
surf(X_mesh, Y_mesh, error_matrix);
shading interp; % Deixa as transições de cores suaves
colorbar;       % Mostra a barra de intensidade do erro
colormap jet;   % Paleta clássica (azul = erro baixo, vermelho = erro alto)

xlabel('log10(T)');
ylabel('log10(\alpha)');
zlabel('Mean Squared Error (MSE)');
title('3D Optimization Error Surface');
grid on;

% Adiciona um marcador (X vermelho) exatamente no ponto de menor erro
hold on;
plot3(log10(best_T), log10(best_alpha), best_error, 'rx', 'MarkerSize', 15, 'LineWidth', 3);
legend('Error Surface', 'Optimal Combination', 'Location', 'best');
hold off;

% Rotaciona levemente a visualização inicial para o efeito 3D ficar nítido
view(-35, 30); 

%% ========================================================================
%% Plot da Melhor Imagem Isolada
%% ========================================================================
figure('Name', 'Best Restored Image');
imagesc(best_x_ep);
caxis([0 1]);
colormap gray;
axis image off;
title(sprintf('Best Edge-Preserving Image (\\alpha = %.1e, T = %.1e)', best_alpha, best_T));

%% ========================================================================
%% Cost Functions & Helpers
%% ========================================================================
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