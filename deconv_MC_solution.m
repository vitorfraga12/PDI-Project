%-------------------------------------------------------------------------
%----- Fichier : deconv_MC
%----- Objet   : Déconvolution par moindres carrés régularisés ou non
%----- Références principales : Idier
%-------------------------------------------------------------------------

close all;
clear all;

L_x = 100;	    % durée d'observation du signal d'entrée
fe = 1;		    % fréquence d'échantillonage
Te = 1/fe;		% période d'échantillonage
N_x = L_x/Te;	% nombre de points du signal x
k_x = 0:N_x;	% index temporel
t_x= k_x*Te;    % base de temps
%**************************************************************************
% 1. Signal x
k1 = 0:N_x/2-1;
x1 = zeros(1,N_x/2);
k2 = N_x/2:N_x;
x2 = ones(1,length(k2));
x = [x1'; x2'];
subplot(2,3,1)
plot(t_x,x);
xlabel('temps (s)'); ylabel('amplitude'); title('Signal entrée x(t)');


%**************************************************************************
% 2. Réponse impulsionnelle h de filtres linéaires gaussiens
% test de plusieurs filtres gaussien ; la largeur de la réponse impulsionnelle va rendre le problème plus ou
% moins difficile
%**************************************************************************
N_sigma=30;
mu_h=15*Te;
for sigma=1:N_sigma,
   sigma_h=sigma*Te;
   L_h=30*Te;
   t_h=(0:Te:L_h);
   N_h=length(t_h);
   h=(1/(sigma_h*sqrt(2*pi)))*exp(-(((t_h-mu_h)/(sqrt(2)*sigma_h)).*((t_h-mu_h)/(sqrt(2)*sigma_h))));
   
   %-----------------------------------------------------------------------
   % 3. Construction de la matrice H de convolution (Toeplitz)
   N_x=length(x);
   N_y = N_x + N_h -1;
   
   hcol_1 = zeros(1,N_y);
   hcol_1(1:N_h) = hcol_1(1:N_h) + h;
   hlig_1 = zeros(1,N_x);  
   hlig_1(1,1) = hcol_1(1,1);
   H = toeplitz(hcol_1,hlig_1);
   cond_H(sigma)=cond(H);
end

%**************************************************************************
% Construction d'un seul filtre gaussien
%**************************************************************************
sigma=4;
sigma_h=sigma*Te;
L_h=30*Te;
t_h=(0:Te:L_h);
N_h=length(t_h);
h=(1/(sigma_h*sqrt(2*pi)))*exp(-(((t_h-mu_h)/(sqrt(2)*sigma_h)).*((t_h-mu_h)/(sqrt(2)*sigma_h))));
subplot(2,3,2)
plot(t_h,h)
xlabel('temps (s)'); ylabel('amplitude'); title('Réponse impulsionnelle h(t)');

%--------------------------------------------------------------------------
%-
% 3. Construction de la matrice H de convolution (Toeplitz)

N_y = N_x + N_h -1;


hcol_1 = zeros(1,N_y);
hcol_1(1:N_h) = hcol_1(1:N_h) + h;
hlig_1 = zeros(1,N_x);  
hlig_1(1,1) = hcol_1(1,1);
H = toeplitz(hcol_1,hlig_1);
cond_H(sigma)=cond(H);

%**************************************************************************
%4. Convolution sous forme matricielle y_nb = H x

y_nb=H*x;
k=0:1:N_y-1;
t=k*Te;
subplot(2,3,3)
plot(t,y_nb)
xlabel('temps (s)'); ylabel('amplitude'); title('Signal sortie non bruité y_{nb}(t)');

%**************************************************************************
% 5. Bruit additif gaussien

y= adgnoise(y_nb, 20);   %%% bruitage gaussien à 20 dB

% représentation temporelle du signal de sortie bruité
subplot(2,3,6)
plot(t,y);
xlabel('temps (s)'); ylabel('amplitude'); title('Signal sortie bruité y(t)');

%**************************************************************************
% 6. Représentation du bruit 
w = y - y_nb;
RSB_y=10*log10((y'*y/N_y)/var(w))
subplot(2,3,5)
plot(t,w); 
xlabel('temps (s)'); ylabel('amplitude'); title('Bruit w(t)');

%**************************************************************************
% 7. Déconvolution l2 par moindres carrés

%--------------------------------------------------------------------------
x_rec=(H'*H)\(H'*y);
%--------------------------------------------------------------------------

subplot(2,3,4)
plot(t_x,x_rec)
xlabel('temps (s)'); ylabel('amplitude'); title('Signal d''entrée reconstruit par MC');
	 

%**************************************************************************
% 8. Déconvolution par moindres carrés régularisés l2, pénalisation des
% différences premières

% Construction de la matrice D1 de différentiation

dcol_1 = zeros(1,N_x-1);
dcol_1(1) = 1; 
dlig_1 = zeros(1,N_x);  
dlig_1(1,1:2) = [1 -1];
D1 = toeplitz(dcol_1,dlig_1);


% Intervalle de variation du coefficient de régularisation
min_alpha=-7;
pas_alpha=0.1;
max_alpha=+2;
i_alpha=0;



% ligne suivante (ainsi que "end") à décommenter pour faire varier alpha

  for var_alpha=min_alpha:pas_alpha:max_alpha,

alpha=10^var_alpha;
i_alpha=i_alpha+1;
%--------------------------------------------------------------------------
x_rec_l2 (:,i_alpha)= (H'*H + alpha * D1'*D1)\(H'*y);
%--------------------------------------------------------------------------
err_rec(:,i_alpha)=x-x_rec_l2(:,i_alpha); % erreur de reconstruction

Werr_rec(i_alpha)=err_rec(:,i_alpha)'*err_rec(:,i_alpha); % energie de l'erreur de reconstruction
F(i_alpha)=norm(D1*x_rec_l2 (:,i_alpha),2)^2;
G(i_alpha)=norm(y-H*x_rec_l2 (:,i_alpha),2)^2;
end

% tracé du signal reconstruit superposé au signal recherché
figure
clf,
  plot(t_x,x)
hold on
plot(t_x,x_rec_l2,'r')
xlabel('temps (s)'); ylabel('amplitude'); title('Signal d''entrée reconstruit MC régularisés');
hold off
%pause;

%**************************************************************************
% énergie de l'erreur de reconstruction en fonction du coefficient de régularisation
% à décommenter lorsque l'on fait varier alpha
figure
var_alpha=min_alpha:pas_alpha:max_alpha;
plot(var_alpha,10*log10(Werr_rec))
xlabel('log10(\alpha)'); ylabel('énergie erreur de reconstruction (dB)'); title('Erreur de reconstruction en fonction de \alpha');


%**************************************************************************
% tracé du signal reconstruit superposé au signal recherché pour alpha
% optimal (choix "optimal" pour alpha = )
[W_err_rec_min,i_alpha_opt] = min(Werr_rec);
alpha_opt=10^(min_alpha+pas_alpha*(i_alpha_opt-1))
figure
plot(t_x,x)
hold on
plot(t_x,x_rec_l2(:,i_alpha_opt),'r')
xlabel('temps (s)'); ylabel('amplitude'); title('Signal d''entrée reconstruit régularisé pour alpha optimal');
hold off

%**************************************************************************
% comparaison des conditionnements
figure
plot(log10(svd(H'*H)))
hold on
plot(log10(svd(H'*H + alpha_opt * D1'*D1)),'g')
xlabel('index'); ylabel('log10(valeurs singulières)'); 
title('valeurs singulières')
legend(' H^TH','H^TH + \alpha D_1^TD_1');
hold off

display('conditionnement')
cond(H'*H)
%max(svd(H'*H))/min(svd(H'*H))
cond(H'*H + alpha_opt * D1'*D1)

%**************************************************************************
% Courbe en L
figure
plot(log10(G),log10(F),'.')
xlabel('log10(critére moindres carrés)'); ylabel('log10(terme de pénalité)'); title('Courbe en L');



%**************************************************************************
% conditionnement en fonction de la largeur du filtre

figure
sigma=1:N_sigma;
stem(sigma,log10(cond_H))
grid
xlabel('sigma_h'); ylabel('log10(conditionnement de H)'); title('Conditionnement en fonction de la largeur de la réponse impulsionnelle');

