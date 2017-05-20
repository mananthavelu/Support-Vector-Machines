load breast

% display of the data

for i=1:30,
    figure;
    scatter(trainset(:,i),labels_train)
    xlabel(['X', num2str(i)]), ylabel('Labels');
end

% we plot each variable against the labels. Allows to see if points for each
% variable can be easily separable
% we see that X1, X3, X4, X7, X8, X11, X13, X14, X21, X23, X24 and X28 
% seem to have good explanatory power

% as an example, for the 5 first most explanatory variables

index= [1 3 4 7 8];
gplotmatrix(trainset(:,index),trainset(:,index), labels_train);

% seems relatively easily separable

% linear model

gam = 1;
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'});
[Yh, Z] = simlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'},{alpha,b},testset);
err = sum(Yh~=labels_test); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(labels_test)*100)

% 4.73% error rate, really good. Gam has no impact on the error rate or the separation
% line 

% Rbf kernel

gam = 1;
sig2 = 20;
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'});
[Yh, Z] = simlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'},{alpha,b},testset);
err = sum(Yh~=labels_test); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(labels_test)*100)

% we obtain a classification error of 1.18%

gamlist =[];
sig2list =[];
errlist =[];

for i=1:20,
    model = {trainset,labels_train,'c',[],[],'RBF_kernel','csa'};
    [gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});
    gamlist = [gamlist; gam];
    sig2list = [sig2list; sig2];
    [Yh, Z] = simlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'},{alpha,b},testset);
    err = sum(Yh~=labels_test); 
    errlist = [errlist; err];
end

figure;
plot(1:1:length(gamlist), gamlist);
xlabel('run id'), ylabel('gam');
figure
plot(1:1:length(sig2list), sig2list);
xlabel('run id'), ylabel('sig2');
figure;
plot(1:1:length(errlist), errlist/length(labels_test)*100)
xlabel('run id'), ylabel('classification error');
%Everything changes a lot, also classification error, best run for 8, close
%to our original estimate

%ROC

%linear ROC
gam = 1;
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'});
[Ysim_lin,Ylatent] = simlssvm({trainset,labels_train,'c',gam,[], ...
'lin_kernel'},{alpha,b},testset);

%Rbf kernel
%parameters from run 8
gam = 50;
sig2 = 20;
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,sig2, 'RBF_kernel'});
[Ysim_Rbf,Ylatent] = simlssvm({trainset,labels_train,'c',gam,sig2, ...
'RBF_kernel'},{alpha,b},testset);
roc(Ysim_lin,labels_test, 'figure');
hold on;
roc(Ysim_Rbf,labels_test, 'figure');

%RBf kernel is a bit better