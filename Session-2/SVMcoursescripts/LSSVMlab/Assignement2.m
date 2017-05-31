

%2.2 A Simple Example: The sinc

demofun

X = (-3:0.01:3)';
Y = sinc(X)+0.1.*randn(length(X), 1);
Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));

gam = 1;
sig2 = 9.0;
[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});
plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b});
YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},...
{alpha,b},Xtest);
plot(Xtest,Ytest,'.');
hold on;
plot(Xtest,YtestEst,'r+');
legend('Ytest','YtestEst');

gamlist=[1 10:20:90];
sig2list=[0.1 1:2:9];
errlist = [];
for gam=gamlist,
    for sig2=sig2list
        [alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});
        YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},...
        {alpha,b},Xtest);
        figure;
        plot(Xtest,Ytest,'.');
        hold on;
        plot(Xtest,YtestEst,'r+');
        legend('Ytest','YtestEst');
        title(['sig2=', num2str(sig2), ' , gam=', num2str(gam)])
    end
end

%Yes looking at the figures, there are optimal parameters minimise the
%error between the real and estimated function

%2.3 Hyper-parameter Tuning

gam = 100;
sig2 = 1.0;

cost_crossval = crossvalidate({Xtrain,Ytrain,'f',gam,sig2},10);
cost_loo = leaveoneout({Xtrain,Ytrain,'f',gam,sig2});

optFun = 'gridsearch';
[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel'}, ...
optFun,'crossvalidatelssvm',{10,'mse'});

[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2});
plotlssvm({Xtrain,Ytrain,'f',gam,sig2},{alpha,b});

gamlist=[];
sig2list=[];
costlist=[];

for i=1:20
    [gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel'}, ...
    optFun,'crossvalidatelssvm',{10,'mse'});
    [alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2});
    figure
    plotlssvm({Xtrain,Ytrain,'f',gam,sig2},{alpha,b});
    gamlist= [gamlist; gam];
    sig2list= [sig2list; sig2];
    costlist= [costlist; cost];
end

figure
plot(1:20, gamlist)
xlabel('run id'), ylabel('gam')

figure
plot(1:20, sig2list)
xlabel('run id'), ylabel('sig2')

figure
plot(1:20, costlist)
xlabel('run id'), ylabel('cost')

%Parameters change a lot but have little influence on the cost or the
%estimated function

optFun = 'simplex';
[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel'}, ...
optFun,'crossvalidatelssvm',{10,'mse'});

[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2});
plotlssvm({Xtrain,Ytrain,'f',gam,sig2},{alpha,b});

gamlist=[];
sig2list=[];
costlist=[];

for i=1:20
    [gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel'}, ...
    optFun,'crossvalidatelssvm',{10,'mse'});
    [alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2});
    figure
    plotlssvm({Xtrain,Ytrain,'f',gam,sig2},{alpha,b});
    gamlist= [gamlist; gam];
    sig2list= [sig2list; sig2];
    costlist= [costlist; cost];
end

figure
plot(1:20, gamlist)
xlabel('run id'), ylabel('gam')

figure
plot(1:20, sig2list)
xlabel('run id'), ylabel('sig2')

figure
plot(1:20, costlist)
xlabel('run id'), ylabel('cost')

%For a given starting point (calculated based on csa), gridsearch
%exhaustively considers all parameter combinations for a grid around the
%starting point. The startvalues determine the limits of the grid over parameter
%space. 

%The Nelder–Mead method or downhill simplex method or amoeba method is a 
%commonly applied numerical method used to find the minimum or maximum
%of an objective function in a multidimensional space. It is applied
%to nonlinear optimization problems for which derivatives may not be known
 
%2.4 Application of the Bayesian Framework

sig2 = 0.5; gam = 10;
criterion_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1)
criterion_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2)
criterion_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3)

[~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},1);
[~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},2);
[~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},3);

%See slide 6 of course 7
sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure');

clear;
load iris;
gam = 5; sig2 = 0.75;

bay_modoutClass({X,Y,'c',gam,sig2},'figure');
colorbar;
%Colors represent the probability that a point belongs to the positive
%class

gam = 20; sig2 = 0.75;
bay_modoutClass({X,Y,'c',gam,sig2},'figure');
colorbar;
gam = 50; sig2 = 0.75;
bay_modoutClass({X,Y,'c',gam,sig2},'figure');
colorbar;
gam = 5; sig2 = 7.5;
bay_modoutClass({X,Y,'c',gam,sig2},'figure');
colorbar;
gam = 5; sig2 = 50;
bay_modoutClass({X,Y,'c',gam,sig2},'figure');
colorbar;

X = 6.*rand(100,3)-3;
Y = sinc(X(:,1)) + 0.1.*randn(100,1);
[selected, ranking] = bay_lssvmARD({X,Y,'f',gam,sig2,});
optFun = 'simplex';
[gam,sig2,cost] = tunelssvm({X(:,selected),Y,'f',[],[],'RBF_kernel','csa'}, ...
optFun,'crossvalidatelssvm',{10,'mse'}); 
[alpha,b] = trainlssvm({X(:,selected),Y,'f',gam,sig2,'RBF_kernel'}); 
points(X(:,selected),Y)

plotlssvm({X(:,selected),Y,'f',gam,sig2,'RBF_kernel'},{alpha,b});

%cross validate
cost_crossval = [];
varlist = {'1', '2', '3', '1:2', '1:3', '2:3', '1:3'};
varlist = cellstr(varlist);

for i=1:7,
    optFun = 'simplex';
    [gam,sig2,cost] = tunelssvm({X(:,eval(char(varlist(i)))),Y,'f',[],[],'RBF_kernel','csa'}, ...
    optFun,'crossvalidatelssvm',{10,'mse'}); 
   cost_crossval(i) = cost;%crossvalidate({X(:,eval(char(varlist(i)))),Y,'f',gam,sig2},10);
end
cost_crossval
%2.5 Robust Regression

X = (-6:0.2:6)';
Y = sinc(X) + 0.1.*rand(size(X));

out = [15 17 19];
Y(out) = 0.7+0.3*rand(size(out));
out = [41 44 46];
Y(out) = 1.5+0.2*rand(size(out));

gam = 100;
sig2 = 0.1;

[alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel'},{alpha,b});

Y = sinc(X) + 0.1.*rand(size(X));

[alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
figure;
plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel'},{alpha,b});

%Outliers impact the regression locally (where outliers are).
%Learning observations with outliers without awareness may lead
%to fitting those unwanted data and may corrupt the approximation 
%function. This will result in the loss of generalization
%performance in the test phase.

out = [15 17 19];
Y(out) = 0.7+0.3*rand(size(out));
out = [41 44 46];
Y(out) = 1.5+0.2*rand(size(out));

model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'whuber';
model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
model = robustlssvm(model);
figure;
plotlssvm(model);

%Both mean squared error (MSE) and mean absolute error (MAE) 
%are used in predictive modeling. MSE has nice mathematical 
%properties which makes it easier to compute the gradient.
%However, MAE requires more complicated tools such as linear 
%programming to compute the gradient. Because of the square,
%large errors have relatively greater influence on MSE 
%than do the smaller error. Therefore, MAE is more robust to outliers
%since it does not make use of square.

%MAybe not correct, see below, not the asme in manual
%Outlier removal via Hampel identifier.
%Y = hampel(X) replaces any element in vector X that is more than three
%standard deviations from the median of itself and up to three
%neighboring elements with that median value.  The standard deviation is
%estimated by scaling the local median absolute deviation (MAD) by a
%constant scale factor

wFun = 'whampel';
model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
model = robustlssvm(model);
figure;
plotlssvm(model);

wFun = 'wlogistic';
model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
model = robustlssvm(model);
figure;
plotlssvm(model);

wFun = 'wmyriad';
model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
model = robustlssvm(model);
figure;
plotlssvm(model);

%each weighting function deals differently with outliers
%see manual from leuven for details

%2.6 Homework Problem
%2.6.1 Introduction: Time-series Prediction

load logmap;
order = 10;

%we standardize data otherwie it doesnt work 
 mu = mean(Z);
 sig = std(Z);
 Zs = (Z-mu)/sig;

X = windowize(Zs,1:(order+1));
Y = X(:,end);
X = X(:,1:order);

gam = 10; sig2 = 10;
[alpha,b] = trainlssvm({X,Y,'f',gam,sig2});

%As in the previous section, the gam and sig2 can be optimized using
%crossvalidate. In the same way,one can optimize order as a parameter. 
optFun = 'simplex';
[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel','csa'}, ...
optFun,'crossvalidatelssvm',{10,'mse'});

%Xnew = Z((end-order+1):end)';
%Z(end+1) = simlssvm({X,Y,'f',gam,sig2},{alpha,b},Xnew);

horizon = length(Ztest)-order;
[alpha,b] = trainlssvm({X,Y,'f',gam,sig2});
Zpt = predict({X,Y,'f',gam,sig2},Ztest(1:order),horizon);
error=immse(Zpt,Ztest(order+1:end));
plot([Ztest(order+1:end) Zpt]);
legend('orig','predicted');
title(sprintf('order = %d MSE %.4f',order,error));

%For loop to optimize
error = [];

for i=1:1:15,
    i
    order = i;
    X = windowize(Zs,1:(order+1));
    Y = X(:,end);
    X = X(:,1:order);
    [gam,sig2,costs] = tunelssvm({X,Y,'f',[],[],'RBF_kernel','csa'},'simplex','crossvalidatelssvm',{10,'mse'});
    model = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
    Zc = (Ztest_withoutnoise(1:order)-mu)/sig;
    Zpts = predict(model,Zc,max(size(Ztest_withoutnoise))-order);
    Zpt = (Zpts*sig)+mu;
    error(i) = immse(Ztest_withoutnoise((order+1):end),Zpt);
    figure
    plot([Ztest_withoutnoise((order+1):end) Zpt]);
    legend('orig','predicted');
    title(sprintf('order = %d MSE %.4f',order,error(i)));
end

figure
plot(1:15, error)
xlabel('Order'), ylabel('MSE')

%2.6.2 Application: Santa Fe Laser Dataset

load santafe
figure
plot(Z)
title('Santa Fe training set')
figure
plot(Ztest)
title('Santa Fe test set')

%Normalization
mu = mean(Z);
sig = std(Z);
Zs = (Z-mu)/sig;

lag = 50;

horizon = length(Ztest)-lag;
X = windowize(Zs,1:lag+1);
Y = X(1:end-lag,end); %training set
X = X(1:end-lag,1:lag); %training set

[gam,sig2,costs] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', 'csa'},'gridsearch',...
'crossvalidatelssvm',{10,'mse'});

model = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
%predict next 200-order points
Zc = (Ztest(1:lag)-mu)/sig;
prediction = predict(model,Zc,max(size(Ztest))-lag);
Zpt = (prediction*sig)+mu;
error=immse(Ztest((lag+1):end),Zpt);
figure
plot([Zpt Ztest((lag+1):end)]);
title(sprintf('order = %d MSE %.4f',lag,error))
legend('predicted','orig');

%optimization for lag
order_vec=[30 35 40 45 50 55 60 65 70 75 80]
error=[];
j=0;
for i=order_vec
    j=j+1;
    i
    lag = i;
    horizon = length(Ztest)-lag;
    X = windowize(Zs,1:lag+1);
    Y = X(1:end-lag,end); %training set
    X = X(1:end-lag,1:lag); %training set

    [gam,sig2,costs] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', 'csa'},'gridsearch',...
    'crossvalidatelssvm',{10,'mse'});

    model = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
    %predict next 200-order points
    Zc = (Ztest(1:lag)-mu)/sig;
    prediction = predict(model,Zc,max(size(Ztest))-lag);
    Zpt = (prediction*sig)+mu;
    error(j)=immse(Ztest((lag+1):end),Zpt);
    figure
    plot([Zpt Ztest((lag+1):end)]);
    title(sprintf('order = %d MSE %.4f',lag,error(j)))
    legend('predicted','orig');
end

plot(order_vec, error)
xlabel('Order'), ylabel('MSE')
