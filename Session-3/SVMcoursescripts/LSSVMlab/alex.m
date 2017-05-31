load santafe

%Normalization
mu = mean(Z);
sig = std(Z);
Zs = (Z-mu)/sig;


%A Time-Series Example: Santa Fe Laser Data Prediction
 
%Using the static regression technique, a nonlinear feedforward prediction model can be built. The NARX model takes the past measurements as input to the model
 
 % load time-series in X and Xt
X = Zs;
Xt = (Ztest-mu)/sig;
delays = 50; 
Xu = windowize(X,1:delays+1);
%The hyperparameters can be determined on a validation set. Here the data are split up in 2 distinct set of successive signals: one for training and one for validation:
 
Xtra = Xu(1:400,1:delays); Ytra = Xu(1:400,end);
Xval = Xu(401:950,1:delays); Yval = Xu(401:950,end);

[gam,sig2] = tunelssvm({Xu(:,1:delays),Xu(:,end),'f',10,50,'RBF_kernel', 'csa'},...
                          'gridsearch','crossvalidatelssvm',{10,'mse'});
%The number of lags can be determined by Automatic Relevance Determination, although this technique is known to work suboptimal in the context of recurrent models
inputs = bay_lssvmARD({Xu(:,1:delays),Xu(:,end),...
                              'f',gam,sig2,'RBF_kernel'});
%Prediction of the next 100 points is done in a recurrent way:
 
 [alpha,b] = trainlssvm({Xu(:,inputs),Xu(:,end),...
                          'f',gam,sig2,'RBF_kernel'});
 prediction = predict({Xu(:,inputs),Xu(:,end),...
                        'f',gam,sig2,'RBF_kernel'},Xt);
 Zpt = (prediction*sig)+mu;                   
 plot([Zpt Xt]);
%In Figure [*] results are shown for the Santa Fe laser data.