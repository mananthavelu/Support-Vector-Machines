%Assignement 1

%Question 1.1

X1 = 1 + randn(50,2);
X2 = -1 + randn(51,2);

Y1 = ones(50,1);
Y2 = -ones(51,1);

X = [X1;X2];
Y = [Y1;Y2];

figure;
hold on;
plot(X1(:,1),X1(:,2),'ro');
plot(X2(:,1),X2(:,2),'bo');
hold off;

%Yes it is always possible to separate the points by a combination
%of line. Condition is that points of the same category remains 
%on the same side of the combined lines

%Question 1.2
%1. When adding points to the classes, the classification line (hyperplane)
%   and the margin change. if points are added on the right side of the
%   classification line, the line and margin do not change substantially

%2. Classification boundaries can change drastically
%   if outlying points are added. classification boundaries can rotate 
%   up to 180 degrees

%3. C is essentially a regularisation parameter, which controls the 
%   trade-off between achieving a low number of misclassifications 
%   and maximising the margin. Obviously, low values of C can influence 
%   the classification outcome for points close to the hyperplane

%4. Switching to Rbf kernel implies using a non linear kernel. 
%   Misclassification is less likely. Rbf allows to handle seriously
%   non-separable linear data

%5. In the equation of the Rbf kernel, sigma plays a role to be 
%   an amplifier of the distance between x and x'. If the distance 
%   between x and x' is much larger than sigma, the kernel function 
%   tends to be zero. Thus, if the sigma is very small, only the x 
%   within the certain distance can affect the predicting point.
%   In other words, smaller sigma tends to make a local classifier,
%   larger sigma tends to make a much more general classifier.
%   For almost linearly separable cases, high C and sigma are prefered 
%   because of their generalisation of the classifier

%6. 

%7. Support Vectors basicaly are the observations needed to define the 
%   fitted model. The more complex the model is (the more specific and 
%   overtrained), the more data points will be needed to specify the model
%   Very few support vectors usually define a general model (with more 
%   missclassifications), while many support vectors specify a very 
%   specific model (possibly overfit). For a linear vector machine,
%   support vectors are the observations lying within the margin or miss-
%   classified. A point becomes a support vector when he 
%   he plays a role in defining the model/classification boundaries

%8. The importance of support vectors change when they influence to a 
%   different extend the separation boundary.

%1.3
%1.3.1
democlass
load iris
[alpha,b] = trainlssvm({X,Y,'c',gam,[],'lin_kernel'});
figure;
plotlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b});

Ytest = simlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b},Xt);
err = sum(Ytest~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)

disp('Press any key to continue...'), pause, 
% Bad performance on Yt as we see 45% of observations missclassified

t = 1;
degree = 1;
[alpha,b] = trainlssvm({X,Y,'c',gam,[t;degree],'poly_kernel'});
figure;
plotlssvm({X,Y,'c',gam,[t;degree],'poly_kernel'},{alpha,b});
% Degree 1 corresponds to the linear case

degree = 2;
[alpha,b] = trainlssvm({X,Y,'c',gam,[t;degree],'poly_kernel'});
figure;
plotlssvm({X,Y,'c',gam,[t;degree],'poly_kernel'},{alpha,b});

degree = 3;
[alpha,b] = trainlssvm({X,Y,'c',gam,[t;degree],'poly_kernel'});
figure;
plotlssvm({X,Y,'c',gam,[t;degree],'poly_kernel'},{alpha,b});

% The degree of the polynomial kernel plays on the 'flexibility of the'
% decision curve. It plays a similar role as sigma in the Rbf kernel
% The higher the more flexible.

%1.3.2
gam = 1;
sig2list=[0.01, 0.1, 1, 5, 10];

errlist=[];

for sig2=sig2list,
    [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); 
    errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yt)*100)
    disp('Press any key to continue...'), pause,         
end

% make a plot of the misclassification rate wrt. sig2

figure;
plot(log(sig2list), errlist, '*-'), 
xlabel('log(sig2)'), ylabel('number of misclass'),

%1.3.3

sigma2 = 0.1
gamlist=[0.1, 1, 10, 50, 100, 1000];

errlist=[];

for gam=gamlist,
    [alpha,b] = trainlssvm({X,Y,type,gam,sigma2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({X,Y,type,gam,sigma2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({X,Y,type,gam,sigma2,'RBF_kernel'}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); 
    errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yt)*100)
    disp('Press any key to continue...'), pause,         
end

% make a plot of the misclassification rate wrt. sig2

figure;
plot(log(gamlist), errlist, '*-'), 
xlabel('log(gam)'), ylabel('number of misclass'),

% 1 to 10 seems to be a good range for gam

%1.3.1.1
% The validation set is used to compare the performance of different 
% models (in terms of their parameters) while the training set is used 
% to train the different models. Obviously, the best model will be chosen
% on the base of the validation set. Therefore, the performance of the 
% final model has to be tested on an independent set, i.e. the test
% set.%

% set the parameters to some value
gam = 0.1;
sig2 = 20;

% generate random indices
idx = randperm(size(X,1));

% create the training and validation sets
% using the randomized indices
Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);
err = sum(estYval~=Yval); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yval)*100)

sig2list=[0.1:0.1:10];
gamlist=[1:1:100];
i=0;
errlist=[];

for gam=gamlist,
    i=i+1;
    j=0;
    for sig2=sig2list,
        j=j+1;
        [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});

        % Obtain the output of the trained classifier
        estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval); 
        errlist(i,j)=sum(estYval~=Yval);
        %fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', errlist(i,j), errlist(i,j)/length(Yval)*100)
        %disp('Press any key to continue...'), pause,
    end
end

% make a plot of the misclassification rate wrt. sig2 and gam

figure;
pcolor(sig2list, gamlist, errlist)
xlabel('sigma2'), ylabel('gam'), zlabel('number of misclass'),
colorbar,
% Smaller values of sigma do not missclassify (probably overfits the model)
% Higher values of sigma have higher missclassifications, but probably
% deliver a more general model. A high gam parameter (penalisation for miss-
% classification) temd to reduce the number of missclassifications for 
% a given sigma2

%1.3.1.2
sig2=10;
gam=10;
performance = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'}, ...
10,'misclass');

% One of the main reasons for using cross-validation instead of using 
% the conventional validation (e.g. partitioning the data set into two 
% sets of 70% for training and 30% for test) is that there is not enough
% data available to partition it into separate training and test sets
% without losing significant modelling or testing capability.
% In k-fold cross-validation, the original sample is randomly partitioned 
% into k equal sized subsamples. Of the k subsamples, a single subsample
% is retained as the validation data for testing the model, and the
% remaining k ? 1 subsamples are used as training data. 
% The cross-validation process is then repeated k times (the folds),
% with each of the k subsamples used exactly once as the validation data.
% The k results from the folds can then be averaged to produce a 
% single estimation. The advantage of this method over repeated random
% sub-sampling (see below) is that all observations are used 
% for both training and validation, and each observation is used 
% for validation exactly once. 

performance = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'}, 'misclass');

% Leave-one-out cross-validation (LpO CV) involves using 1 observation
% as the validation set and the remaining observations as the training
% set. This is repeated on all ways to cut the original sample on a
% validation set of 1 observation and a training set of n-1 observations.

% Leave-one-out is better when dealing with samll data sets, with only
% few observations

%1.3.1.3

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

% The reason why results can be very different is because different
% optimization techniques are used to tune the parameters of the SVM
% model. First, an approximation of the optimum is obtained through an 
% optimisation algorithm. Then, a search is performed to improve the 
% optimum obtained at the previous step. For the search to be performed,
% an initial point has to be provided. This point is obviously the 
% optimum obtained during the first step. 
% So both steps can have different methods applied (couple simulated 
% annealing or directional search for the first step and simplex or 
% gridsearch for the second step). The initial point provided has a 
% lot of impact on the final solution. Therefore final results can differ 
% heavily

%1.3.1.4

[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2, 'RBF_kernel'});
[Ysim,Ylatent] = simlssvm({Xtrain,Ytrain,'c',gam,sig2, ...
'RBF_kernel'},{alpha,b},Xval);
roc(Ysim,Yval);

% It is not recommended to use ROC on training set because the model the
% idea is to test the generalisation power of the model. Calculating ROC
% on training set wont give any feeling on how the model performs on
% new data. Rather, ROC should be calculted for validation or development
% sets.

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Breat cancer

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

