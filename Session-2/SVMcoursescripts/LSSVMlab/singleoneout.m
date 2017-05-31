%1.3.1.2
load iris
%Set the parameters to some value

%generate random indices
idx=randperm(size(X,1));

% create the training and validation sets
% using the randomized indices
Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

%Training

sig2=10;
gam=10;
performance = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'},10,'misclass');