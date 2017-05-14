load iris


%
% train LS-SVM classifier with linear kernel 
%
type='c'; 
gam = 1; 
disp('Linear kernel'),

[alpha,b] = trainlssvm({X,Y,type,gam,[],'lin_kernel'});

figure; plotlssvm({X,Y,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({X,Y,type,gam,[],'lin_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)