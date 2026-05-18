%-------------------------------------------------------------------------
%----- File : deconv_MC
%----- Object   : Deconvolution using regularized least squares or not
%----- Main reference, book by J. Idier, 2008
%-------------------------------------------------------------------------

close all;
clear all;

L_x = 100;	    % duration of observation of the input signal
fe = 1;		    % sampling frequency
Te = 1/fe;	% sampling period
N_x = L_x/Te;	% number of points in signal x
k_x = 0:N_x;	% time index 
t_x= k_x*Te;    % time basis
%**************************************************************************
% 1. Signal x
k1 = 0:N_x/2-1;
x1 = zeros(1,N_x/2);
k2 = N_x/2:N_x;
x2 = ones(1,length(k2));

x = [x1'; x2'];
subplot(2,3,1)
plot(t_x,x);
xlabel('time (s)'); ylabel('amplitude'); title('Input signal x(t)');

 

%**************************************************************************
% Construction of a single Gaussian filter
%**************************************************************************
sigma=4;
sigma_h=sigma*Te;
mu_h = 15*Te
L_h=30*Te;
t_h=(0:Te:L_h);
N_h=length(t_h);
h=(1/(sigma_h*sqrt(2*pi)))*exp(-(((t_h-mu_h)/(sqrt(2)*sigma_h)).*((t_h-mu_h)/(sqrt(2)*sigma_h))));
subplot(2,3,2)
plot(t_h,h)
xlabel('time (s)'); ylabel('amplitude'); title('Impulse Response h(t)');

%--------------------------------------------------------------------------
%-
% 3. Construction of convolution matrix H (Toeplitz)

Nx_total = length(x); % O verdadeiro tamanho de x (que é 101)
N_y = Nx_total + N_h - 1; % Tamanho correto para a convolução linear

hcol_1 = [h zeros(1, N_y - N_h)]'; % Primeira coluna

hlig_1 = [h(1) zeros(1, Nx_total - 1)]; % Primeira linha (agora tem exatas 101 colunas)

H = toeplitz(hcol_1, hlig_1);


%**************************************************************************
%4. Convolution in matrix form y_nb = H x

y_nb= H * x;
k=0:1:N_y-1;
t=k*Te;
subplot(2,3,3)
plot(t,y_nb)
xlabel('time (s)'); ylabel('amplitude'); title('Output noiseless signal y_{nb}(t)');

%**************************************************************************
% 5. Gaussian additive noise

SNR = 30;
y = adgnoise(y_nb, SNR);


% time representation of noisy output signal 
subplot(2,3,6)
stairs(t,y);
xlabel('time (s)'); ylabel('amplitude'); title('Noisy output signal y(t)');

%**************************************************************************
% 6. Noise display
w = y - y_nb;
RSB_y = 10*log10(mean(y_nb.^2)/mean(w.^2))
subplot(2,3,5)
plot(t,w); 
xlabel('time (s)'); ylabel('amplitude'); title('Noise w(t)');

%**************************************************************************
% 7. L2 deconvolution by least squares

%--------------------------------------------------------------------------
x_rec= (H'*H)\(H'*y);
%--------------------------------------------------------------------------

subplot(2,3,4)
plot(t_x,x_rec)
xlabel('time (s)'); ylabel('amplitude'); title('Reconstructed signal with LS');
	 

%**************************************************************************
% 8. Deconvolution by L2-regularized least squares, penalization of first 
% order differences

% Construction of first-order difference matrix D1

Nx_total = length(x);

dcol_1 = [-1 zeros(1, Nx_total - 2)]'; % D1 terá dimensão (Nx_total-1) x Nx_total

dlig_1 = [-1 1 zeros(1, Nx_total - 2)]; 

D1 = toeplitz(dcol_1, dlig_1);
% Interval of variation of regularization coefficient
min_alpha= -7;
pas_alpha= 0.5;
max_alpha= 2;
i_alpha=0;



for var_alpha=min_alpha:pas_alpha:max_alpha,
    alpha=10^var_alpha;
    i_alpha=i_alpha+1;
    %--------------------------------------------------------------------------
    x_rec_l2(:,i_alpha)= (H'*H + alpha*(D1'*D1)) \ (H'*y);
    %--------------------------------------------------------------------------
    err_rec(:,i_alpha)= x- x_rec_l2(:,i_alpha);;
    
    Werr_rec(i_alpha)=sum(err_rec(:,i_alpha).^2);
    F(i_alpha)= norm(y - H*x_rec_l2(:,i_alpha))^2;
    G(i_alpha)=norm(D1*x_rec_l2(:,i_alpha))^2;
end

% draw reconstructed signal with the sought signal
figure
plot(t_x,x)
hold on
plot(t_x,x_rec_l2,'r')
xlabel('time (s)'); ylabel('amplitude'); title('Reconstructed input signal, reg. LS');
hold off

%**************************************************************************
% energy of the reconstruction error with respect to the regularization coefficient
% to de-comment when alpha is varying
figure
var_alpha=min_alpha:pas_alpha:max_alpha;
plot(var_alpha,10*log10(Werr_rec))
xlabel('log10(\alpha)'); ylabel('energy of reconstruction error (dB)'); title('Reconstruction error with respect to \alpha');
%**************************************************************************
% draw reconstructed signal together with the sought signal for optimal alpha 
[W_err_rec_min,i_alpha_opt] = min(Werr_rec);
alpha_opt=10^(min_alpha+pas_alpha*(i_alpha_opt-1))
figure
plot(t_x,x)
hold on
plot(t_x,x_rec_l2(:,i_alpha_opt),'r')
xlabel('time (s)'); ylabel('amplitude'); title('Reconstructed input signal for optimal alpha');
hold off

%**************************************************************************
% comparison of L2 condition numbers

%**************************************************************************
% L curve





