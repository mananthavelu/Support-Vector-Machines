load logmap;
order=10;
X = windowize(Z,1:(order+1));
Y = X(:,end);
X = X(:,1:order);

Xnew = Z((end-order+1):end)';
horizon = length(Ztest)-order;

sig2list=[0.01,0.1,1,10,15];
gamlist=[0.1,0.5,1.0,2.0,5]; 
pl=1;
figure;
for gam=gamlist
    for sig2=sig2list
        [alpha,b] = trainlssvm({X,Y,'f',gam,sig2});
        %Z(end+1) = simlssvm({X,Y,'f',gam,sig2},{alpha,b},Xnew);
        Zpt = predict({X,Y,'f',gam,sig2},Ztest(1:order),horizon);
        subplot(5,5,pl)
        plot([Ztest_withoutnoise(order+1:end) Zpt]);
        pl=pl+1;
    end
end

%Parameters
%gam = 10; sig2 = 10;
%Training SVM
%Function approximtion problem
%[alpha,b] = trainlssvm({X,Y,'f',gam,sig2});

%Paremeters= Gamma, Sigma2, Order
%Testing the model
%Xnew = Z((end-order+1):end)';
%Z(end+1) = simlssvm({X,Y,'f',gam,sig2},{alpha,b},Xnew);


%horizon = length(Ztest)-order;
%Zpt = predict({X,Y,'f',gam,sig2},Ztest(1:order),horizon);
%plot([Ztest(order+1:end) Zpt]);

%test_size = 100;
%Ztrain = Z(1:length(Z)-test_size);
%Ztest = Z(length(Z)-test_size+1:end);
