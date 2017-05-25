%2.2
%Data
X = (-3:0.01:3)';
Y = sinc(X)+0.1.*randn(length(X), 1);

%Training and validation sets are created
Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));

%HyperParameters
gam = 100;
sig2 = 1.0;

%Train the regressor
[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

%Visualization
plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b});

%Performance on the test model
YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},...
{alpha,b},Xtest);

plot(Xtest,Ytest,'.');
hold on;
plot(Xtest,YtestEst,'r+');
legend('Ytest','YtestEst');

