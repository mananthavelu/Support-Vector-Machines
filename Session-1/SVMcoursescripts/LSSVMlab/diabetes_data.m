%Diabetes

load diabetes

% display of the data

for i=1:8,
    figure;
    scatter(trainset(:,i),labels_train)
    xlabel(['X', num2str(i)]), ylabel('Labels');
end

% we plot each variable against the labels. Allows to see if points for each
% variable can be easily separable
% Diabetes will be more challenging than breast, data points seem not to be
% easily separable

% We use plotmatrix on first 5 variables

index= [1 2 3 4 5];
gplotmatrix(trainset(:,index),trainset(:,index), labels_train);

% seems difficult to separate

% linear model

gam = 1;
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'});
plotlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'},{alpha,b});
[Yh, Z] = simlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'},{alpha,b},testset);
err = sum(Yh~=labels_test); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(labels_test)*100)

% 21.43% error rate, quite bad. Gam has no impact on the error rate or the separation
% line 

% Rbf kernel

gam = 1;
sig2 = 50;
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'});
plotlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'},{alpha,b});
[Yh, Z] = simlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'},{alpha,b},testset);
err = sum(Yh~=labels_test); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(labels_test)*100)

% we obtain a classification error of 22.02%, worse than in the linear case

gamlist =[];
sig2list =[];
errlist =[];

for i=1:100,
    model = {trainset,labels_train,'c',[],[],'RBF_kernel','csa'};
    [gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});
    gamlist = [gamlist; gam];
    sig2list = [sig2list; sig2];
    [Yh, Z] = simlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'},{alpha,b},testset);
    err = sum(Yh~=labels_test); 
    errlist = [errlist; err];
end

%get the index of the lowest classification error
[M,I] = min(errlist) 

figure;
plot(1:1:length(gamlist), gamlist);
xlabel('run id'), ylabel('gam');
figure
plot(1:1:length(sig2list), sig2list);
xlabel('run id'), ylabel('sig2');
figure;
plot(1:1:length(errlist), errlist/length(labels_test)*100)
xlabel('run id'), ylabel('classification error');
%Everything changes a lot, also classification error, it appears almost as
%random

%ROC

%linear ROC
gam = 1;
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'});
[Ysim_lin,Ylatent] = simlssvm({trainset,labels_train,'c',gam,[], ...
'lin_kernel'},{alpha,b},testset);

%Rbf kernel
%parameters from minimum run
gam = gamlist(I);
sig2 = sig2list(I);
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,sig2, 'RBF_kernel'});
[Ysim_Rbf,Ylatent] = simlssvm({trainset,labels_train,'c',gam,sig2, ...
'RBF_kernel'},{alpha,b},testset);
roc(Ysim_lin,labels_test, 'figure');
hold on;
roc(Ysim_Rbf,labels_test, 'figure');

%linear kernel is a bit better

