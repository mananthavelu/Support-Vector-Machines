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

%Training
[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});

%Validating
estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);

%Performance
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
        %disp('estYval')
        %disp('Yval')
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