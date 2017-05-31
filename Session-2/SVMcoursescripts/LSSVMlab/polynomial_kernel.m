% Train the LS-SVM classifier using polynomial kernel
%
type='c'; 
gam = 1; 
t = 1; 
%degree = 5;
misclass=[];
error_rate=[];
for degree=1:3;
    
    [alpha,b] = trainlssvm({X,Y,type,gam,[t; degree],'poly_kernel'});

    figure; plotlssvm({X,Y,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});

    [Yht, Zt] = simlssvm({X,Y,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xt);

    err = sum(Yht~=Yt); 
    fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)
    misclass = [misclass err];
    error_rate = [error_rate err/length(Yt)*100];
end

figure
plot(error_rate)
set(gca,'xtick',0:3)
xlabel('degree of polynomial')
ylabel('Error rate')

figure
plot(misclass)
set(gca,'xtick',0:3)
xlabel('degree of polynomial')
ylabel('number of misclassification')
