%2.4 Application of the Bayesian Framework

X = (-3:0.01:3)';
Y = sinc(X)+0.1.*randn(length(X), 1);
Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));


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