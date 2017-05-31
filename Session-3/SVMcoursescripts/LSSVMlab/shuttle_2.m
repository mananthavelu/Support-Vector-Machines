
load('shuttle.dat')
shuttle_ytrain(shuttle_ytrain~=1) = -1;
shuttle_ytest(shuttle_ytest~=1) = -1;

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 3;
function_type = 'c'; %'c' - classification, 'f' - regression  
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];
close all;
[out,e,s,t] = evalc('fslssvm(shuttle_Xtrain,shuttle_ytrain,k,function_type,kernel_type,global_opt,user_process,window,shuttle_Xtest,shuttle_ytest)');
save('fslssvm_shuttle_final.mat');