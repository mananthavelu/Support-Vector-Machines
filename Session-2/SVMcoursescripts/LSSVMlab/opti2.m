load iris
%Set the parameters to some value
gam=0.1;
sig2=20;

%generate random indices
idx=randperm(size(X,1));

% create the training and validation sets
% using the randomized indices
Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

model = {X,Y,'c',[],[],'RBF_kernel','csa'};
[gam_1,sig2_1,cost_1] = tunelssvm(model,'simplex', 'crossvalidatelssvm',{10,'misclass'});

model = {X,Y,'c',[],[],'RBF_kernel','ds'};
[gam_2,sig2_2,cost_2] = tunelssvm(model,'simplex', 'crossvalidatelssvm',{10,'misclass'});

model = {X,Y,'c',[],[],'RBF_kernel','csa'};
[gam_3,sig2_3,cost_3] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});

model = {X,Y,'c',[],[],'RBF_kernel','ds'};
[gam_4,sig2_4,cost_4] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});

y=[gam_1 sig2_1; gam_2 sig2_2; gam_3 sig2_3; gam_4 sig2_4];
bar(y)
Labels = {'csa & simplex', 'ds & simplex', 'csa & gridsearch', 'ds & gridsearch'};
set(gca, 'XTick', 1:4, 'XTickLabel', Labels);
legend('gam', 'sig2');