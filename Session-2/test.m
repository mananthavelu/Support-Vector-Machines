load logmap;
order=10;
[error,cost,Xpt,model]=lssvm_timeseries(Z,Ztest_withoutnoise,order);

function [error,cost,Xpt,model] = lssvm_timeseries(X,Xt,order)
    mu = mean(X);
    sig = std(X);
    Xs = (X-mu)/sig;
    [model,cost] = LSSVM_NAR_RBF(Xs,order);
    figure,plotlssvm(model);
    %idx = (max(size(X))-order)+1;
    %Xpts = predict(model,Xs(idx:end),length(Xt));
    Xc = (Xt(1:order)-mu)/sig
    Xpts = predict(model,Xc,max(size(Xt))-order);
    Xpt = (Xpts*sig)+mu;
    %figure,plot([Xt Xpt]);legend('orig','predicted');
    %error = mse(Xt-Xpt);
    error = mse(Xt((order+1):end)-Xpt);
    figure,plot([Xt((order+1):end) Xpt]);legend('orig','predicted');
    title(sprintf('Performance on test set - order = %d MSE %.4f',order,error));
     
end

function [model,costs] = LSSVM_NAR_RBF(Ztrain,order)
    Xtrain = windowize(Ztrain,1:(order+1));
    Ytrain = Xtrain(:,end);
    Xtrain = Xtrain(:,1:order);
    [gam,sig2,costs] = tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel','csa'},'simplex','crossvalidatelssvm',{10,'mse'});
    model = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});
end