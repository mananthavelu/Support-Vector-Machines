%3.1 Kernel Principal Component Analysis

%PCA is used to reveal the internal structure of the data in a way that 
%best explains the variance in the data. If a multivariate dataset is
%visualised as a set of coordinates in a high-dimensional data 
%space (1 axis per variable), PCA can supply the user with a 
%lower-dimensional picture, a projection of this object when viewed
%from its most informative viewpoint. This is done by using only the
%first few principal components so that the dimensionality of the
%transformed data is reduced.

kpca_script;

%3.2 Handwritten Digit Denoising

digitsdn

%3.3 Spectral Clustering

sclustering_script;

%3.4 Fixed-size LS-SVM
fixedsize_script1;
fixedsize_script2;

%3.5 Homework

%digits
load digits
[N, dim]=size(X);
Ntest=size(Xtest1,1);
minx=min(min(X)); 
maxx=max(max(X));



%
% Add noise to the digit maps
%

noisefactor =1;

noise = noisefactor*maxx; % sd for Gaussian noise


Xn = X; 
for i=1:N;
  randn('state', i);
  Xn(i,:) = X(i,:) + noise*randn(1, dim);
end

Xnt = Xtest1; 
for i=1:size(Xtest1,1);
  randn('state', N+i);
  Xnt(i,:) = Xtest1(i,:) + noise*randn(1,dim);
end

%
% select training set
%
Xtr = X(1:1:end,:);

sig2 =dim*mean(var(Xtr)); % rule of thumb

%sigmafactor = 0.7;

sigmafactorlist=[0.01 0.2:0.2:2];
recon_error_tr=[];
recon_error_val=[];
s=0;
for sigmafactor=sigmafactorlist,
    s=s+1;
    sig2 =dim*mean(var(Xtr)); % rule of thumb

    sig2 =sig2*sigmafactor  

    %
    % kernel based Principal Component Analysis using the original training data
    %


    disp('Kernel PCA: extract the principal eigenvectors in feature space');
    disp(['sig2 = ', num2str(sig2)]);


    % linear PCA
    [lam_lin,U_lin] = pca(Xtr);

    % kernel PCA
    [lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
    [lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);

    
    %treatment on validation set
    %
    % Denoise using the first principal components
    %
    disp(' ');
    disp(' Denoise using the first PCs');

    % choose the digits for test
    digs=[0:9]; ndig=length(digs);
    m=2; % Choose the mth data for each digit 

    Xdt=zeros(ndig,dim);

    % which number of eigenvalues of kpca
    npcs = [2.^(0:7) 190];
    lpcs = length(npcs);

    for k=1:lpcs;
        nb_pcs=npcs(k); 
        disp(['nb_pcs = ', num2str(nb_pcs)]); 
        Ud=U(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
    
        for i=1:ndig
            dig=digs(i);
            fprintf('digit %d : ', dig)
            xt=Xnt(i,:);
            xt_tr=X(i,:);    
            Xdt(i,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
            Xdt_tr(i,:) = preimage_rbf(Xtr,sig2,Ud,Xn(i,:),'denoise');
        end % for i
        recon_error_val(s,k)=immse(Xdt,Xtest1);
        recon_error_tr(s,k)=immse(Xdt_tr,X(1:ndig,:));
    end % for k

    %
    % denosing using Linear PCA for comparison
    %

    % which number of eigenvalues of pca
    npcs = [2.^(0:7) 190];
    lpcs = length(npcs);

    for k=1:lpcs;
        nb_pcs=npcs(k); 
        Ud=U_lin(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
    
        for i=1:ndig
            dig=digs(i);
            xt=Xnt(i,:);
            proj_lin=xt*Ud; % projections of linear PCA
            Xdt_lin(i,:) = proj_lin*Ud';
        end % for i
    end % for k
end

figure;
for k=1:lpcs;
    nb_pcs=npcs(k); 
    plot(sigmafactorlist,recon_error_val(:,k))
    legendInfo{k} =['n=',num2str(nb_pcs)];
    hold all;
end
xlabel('sigmafactor');
ylabel('reconstruction error');
legend(legendInfo);

%treatment on test

Xnt = Xtest2; 
for i=1:size(Xtest2,1);
  randn('state', N+i);
  Xnt(i,:) = Xtest2(i,:) + noise*randn(1,dim);
end
%
% Denoise using the first principal components
%
disp(' ');
disp(' Denoise using the first PCs');

sig2 =dim*mean(var(Xtr)); % rule of thumb
sigmafactor=0.5;
sig2=sig2*sigmafactor;  

% linear PCA
[lam_lin,U_lin] = pca(Xtr);

% kernel PCA
[lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
[lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);

  % choose the digits for test
  digs=[0:9]; ndig=length(digs);
  m=2; % Choose the mth data for each digit 

  Xdt=zeros(ndig,dim);

  %
  % figure of all digits
  %
  %
  figure; 
  colormap('gray'); 
  title('Denosing using linear PCA'); tic
  
  % which number of eigenvalues of kpca
  npcs = [2.^(0:7) 190];
  lpcs = length(npcs);
  recon_error_test=[];

  for k=1:lpcs;
      nb_pcs=npcs(k); 
      disp(['nb_pcs = ', num2str(nb_pcs)]); 
      Ud=U(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
    
      for i=1:ndig
          dig=digs(i);
          fprintf('digit %d : ', dig)
          xt=Xnt(i,:);
          if k==1 
          % plot the original clean digits
          %
            subplot(2+lpcs, ndig, i);
            pcolor(1:15,16:-1:1,reshape(Xtest2(i,:), 15, 16)'); shading interp; 
            set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
     
            if i==1, ylabel('original'), end 
     
            % plot the noisy digits 
            %
            subplot(2+lpcs, ndig, i+ndig); 
            pcolor(1:15,16:-1:1,reshape(xt, 15, 16)'); shading interp; 
            set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
            if i==1, ylabel('noisy'), end
            drawnow
      end   
          Xdt(i,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
          subplot(2+lpcs, ndig, i+(2+k-1)*ndig);
          pcolor(1:15,16:-1:1,reshape(Xdt(i,:), 15, 16)'); shading interp; 
          set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);           
          if i==1, ylabel(['n=',num2str(nb_pcs)]); end
          drawnow 
     end % for i
     recon_error_test(k,i)=immse(Xdt,Xtest2);
  end % for k


  %
  % denosing using Linear PCA for comparison
  %

  % which number of eigenvalues of pca
  npcs = [2.^(0:7) 190];
  lpcs = length(npcs);

  figure; colormap('gray');title('Denosing using linear PCA');
  
  for k=1:lpcs;
      nb_pcs=npcs(k); 
      Ud=U_lin(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
    
      for i=1:ndig
          dig=digs(i);
          xt=Xnt(i,:);
          proj_lin=xt*Ud; % projections of linear PCA
          if k==1 
            % plot the original clean digits
            %
            subplot(2+lpcs, ndig, i);
            pcolor(1:15,16:-1:1,reshape(Xtest2(i,:), 15, 16)'); shading interp; 
            set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);                
            if i==1, ylabel('original'), end  
        
            % plot the noisy digits 
            %
            subplot(2+lpcs, ndig, i+ndig); 
            pcolor(1:15,16:-1:1,reshape(xt, 15, 16)'); shading interp; 
            set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
            if i==1, ylabel('noisy'), end
        end
          Xdt_lin(i,:) = proj_lin*Ud';
          subplot(2+lpcs, ndig, i+(2+k-1)*ndig);
          pcolor(1:15,16:-1:1,reshape(Xdt_lin(i,:), 15, 16)'); shading interp; 
          set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
    
         if i==1, ylabel(['n=',num2str(nb_pcs)]), end
      end % for i
  end % for k
  
  
%Shuttle
data = load('shuttle.dat','-ascii'); function_type = 'c'; 
%data=data(1:10000,:);
addpath('../LSSVMlab')
figure
[a,b]=hist(data(:,end));
hist(data(:,end));
a
b

%stratified sampling to reduce computing time
cv = cvpartition(data(:, end), 'holdout', 1000);
data = data(cv.test, :);

cv = cvpartition(data(:, end), 'holdout', 700);
X = data(cv.test,1:end-1);
Y = data(cv.test,end);

Xtest = data(cv.test==0,1:end-1);
Ytest = data(cv.test==0,end);
figure
hist(Ytest);

model = initlssvm(X,Y,'classifier',[],[],'RBF_kernel');
model = tunelssvm(model,'simplex', 'crossvalidatelssvm',{10,'misclass'});
 
model = trainlssvm(model);
[Yh, Z] = simlssvm(model,Xtest);
err = sum(Yh~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)

Yh1=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==1
        Yh1(i,1)=1;
    else
        Yh1(i,1)=0;
    end
end

Yh2=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==2
        Yh2(i,1)=2;
    else
        Yh2(i,1)=0;
    end
end

Yh3=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==3
        Yh3(i,1)=3;
    else
        Yh3(i,1)=0;
    end
end

Yh4=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==4
        Yh4(i,1)=4;
    else
        Yh4(i,1)=0;
    end
end

Yh5=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==5
        Yh5(i,1)=5;
    else
        Yh5(i,1)=0;
    end
end

Yh6=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==6
        Yh6(i,1)=6;
    else
        Yh6(i,1)=0;
    end
end

Yh7=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==7
        Yh7(i,1)=7;
    else
        Yh7(i,1)=0;
    end
end

[X1,Y1,T1,AUC1]=perfcurve(Ytest,Yh1,1);
[X2,Y2,T2,AUC2]=perfcurve(Ytest,Yh2,2);
[X3,Y3,T3,AUC3]=perfcurve(Ytest,Yh3,3);
[X4,Y4,T4,AUC4]=perfcurve(Ytest,Yh4,4);
[X5,Y5,T5,AUC5]=perfcurve(Ytest,Yh5,5);
[X6,Y6,T6,AUC6]=perfcurve(Ytest,Yh6,6);
[X7,Y7,T7,AUC7]=perfcurve(Ytest,Yh7,7);

plot(X1,Y1)
hold on
%plot(X2,Y2)
hold on
plot(X3,Y3)
hold on
plot(X4,Y4)
hold on
plot(X5,Y5)
hold on
plot(X6,Y6)
hold on
plot(X7,Y7)
legend('ROC Y=1','ROC Y=3','ROC Y=4','ROC Y=5','ROC Y=6','ROC Y=7')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves')
hold off

testX = Xtest;
testY = Ytest;

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 6;
% function_type = 'c'; %'c' - classification, 'f' - regression  
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e_RBF,s_RBF,t_RBF] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);

kernel_type = 'lin_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e_lin,s_lin,t_lin] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);

kernel_type = 'poly_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e_poly,s_poly,t_poly] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);


%California
data = load('california.dat','-ascii'); function_type = 'f';  
%data = data(1:700,:);
addpath('../LSSVMlab')

X = data(:,1:end-1);
Y = data(:,end);
testX = [];
testY = [];

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 6; 
% function_type = 'c'; %'c' - classification, 'f' - regression  
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e_RBF,s_RBF,t_RBF] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);

kernel_type = 'lin_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e_lin,s_lin,t_lin] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);

kernel_type = 'poly_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e_poly,s_poly,t_poly] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);
