% use RBF kernel
%

% tune the sig2 while fix gam
%
disp('RBF kernel')
gamlist = [1,5,10,25,50,100,1000]; sig2=0.1;
errlist=[];

for gamma=gamlist
    disp(['gam : ', num2str(gamma), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({X,Y,type,gamma,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({X,Y,type,gamma,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({X,Y,type,gamma,sig2,'RBF_kernel'}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yt)*100)
    disp('Press any key to continue...'), pause,         
end

%
% make a plot of the misclassification rate wrt. sig2
%
figure;
plot(log(gamlist), errlist, '*-'), 
xlabel('log(gamma)'), ylabel('number of misclass'),