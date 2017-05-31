load('california.dat','-ascii');
%data = load('california.dat','-ascii');
%addpath('../LSSVMlab')

X = california(:,1:end-1);
Y = california(:,end);
testX = [];
testY = [];

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 6;
function_type = 'f'; %'c' - classification, 'f' - regression  
kernel_type = 'lin_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[out,e,s,t] = evalc('fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY)');
save('fslssvm_california.mat');