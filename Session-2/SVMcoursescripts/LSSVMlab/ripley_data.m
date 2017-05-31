load ripley

% display of the data

figure;
hold on;
plot(X(1:125,1),X(1:125,2),'bo');
hold all;
plot(X(126:250,1),X(126:250,2),'ro');
legend('negative class', 'positive class');
xlabel('X1'), ylabel('X2');
legend;
hold off;

% X2 seems to be more explanatory for the class. Data seem to be separated
% into 4 quadrants

% linear model

gam = 1;
[alpha,b] = trainlssvm({X,Y,'c',gam,[],'lin_kernel'});
figure;
plotlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b});
[Yh, Z] = simlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b},X);
err = sum(Yh~=Y); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Y)*100)

% 14.40% error rate. Gam has no impact on the error rate or the separation
% line. Looking at the data, a smoother line would allow a better
% classification rate

% Rbf kernel

gam = 1;
sig2 = 20;
[alpha,b] = trainlssvm({X,Y,'c',gam,sig2,'RBF_kernel'});
plotlssvm({X,Y,'c',gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
[Yh, Z] = simlssvm({X,Y,'c',gam,sig2,'RBF_kernel'},{alpha,b},X);
err = sum(Yh~=Y); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Y)*100)

% Although we obtain a lower classification error, model might not be as
% good as linear one because of the upper left quadrant

gamlist =[];
sig2list =[];
errlist =[];

for i=1:20,
    model = {X,Y,'c',[],[],'RBF_kernel','csa'};
    [gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});
    gamlist = [gamlist; gam];
    sig2list = [sig2list; sig2];
    [Yh, Z] = simlssvm({X,Y,'c',gam,sig2,'RBF_kernel'},{alpha,b},X);
    err = sum(Yh~=Y); 
    errlist = [errlist; err];
end

figure;
plot(1:1:length(gamlist), gamlist);
xlabel('run id'), ylabel('gam');
figure
plot(1:1:length(sig2list), sig2list);
xlabel('run id'), ylabel('sig2');
figure;
plot(1:1:length(errlist), errlist/length(Y)*100)
xlabel('run id'), ylabel('classification error');
%Everything changes a lot, also classification error

%ROC

%linear ROC
gam = 1;
[alpha,b] = trainlssvm({X,Y,'c',gam,[],'lin_kernel'});
[Ysim_lin,Ylatent] = simlssvm({X,Y,'c',gam,[], ...
'lin_kernel'},{alpha,b},X);

%Rbf kernel
model = {X,Y,'c',[],[],'RBF_kernel','csa'};
[gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});
[alpha,b] = trainlssvm({X,Y,'c',gam,sig2, 'RBF_kernel'});
[Ysim_Rbf,Ylatent] = simlssvm({X,Y,'c',gam,sig2, ...
'RBF_kernel'},{alpha,b},X);
roc(Ysim_lin,Y, 'figure');
hold on;
roc(Ysim_Rbf,Y, 'figure');

%RBf kernel is a bit better

%Better to do cross validation to test on the generalisation of the model
