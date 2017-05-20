load breast

% display of the data

figure;
scatter(trainset,labels_train)
xlabel(['X']), ylabel('Labels');


% we plot each variable against the labels. Allows to see if points for each
% variable can be easily separable
% we see that X1, X3, X4, X7, X8, X11, X13, X14, X21, X23, X24 and X28 
% seem to have good explanatory power

% as an example, for the 5 first most explanatory variables
