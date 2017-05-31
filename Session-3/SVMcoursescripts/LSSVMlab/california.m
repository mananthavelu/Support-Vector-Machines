clear;
clc;
close all;

%Shuttle
data = load('california.dat','-ascii'); function_type = 'c'; 
data=data(1:1000,:);
addpath('../LSSVMlab')
figure
[a,b]=hist(data(:,end));
hist(data(:,end));
a
b

%stratified sampling to reduce computing time
cv = cvpartition(data(:, end), 'holdout', 1000);
data = data(cv.test, :);

cv = cvpartition(data(:, end), 'holdout', 700);
X = data(cv.test,1:end-1);
Y = data(cv.test,end);

Xtest = data(cv.test==0,1:end-1);
Ytest = data(cv.test==0,end);
figure
hist(Ytest);

model = initlssvm(X,Y,'classifier',[],[],'RBF_kernel');
model = tunelssvm(model,'simplex', 'crossvalidatelssvm',{10,'misclass'});
 
model = trainlssvm(model);
[Yh, Z] = simlssvm(model,Xtest);
err = sum(Yh~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)

Yh1=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==1
        Yh1(i,1)=1;
    else
        Yh1(i,1)=0;
    end
end

Yh2=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==2
        Yh2(i,1)=2;
    else
        Yh2(i,1)=0;
    end
end

Yh3=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==3
        Yh3(i,1)=3;
    else
        Yh3(i,1)=0;
    end
end

Yh4=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==4
        Yh4(i,1)=4;
    else
        Yh4(i,1)=0;
    end
end

Yh5=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==5
        Yh5(i,1)=5;
    else
        Yh5(i,1)=0;
    end
end

Yh6=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==6
        Yh6(i,1)=6;
    else
        Yh6(i,1)=0;
    end
end

Yh7=Yh;
for i=1:length(Yh(:,1))
    if Yh(i,1)==7
        Yh7(i,1)=7;
    else
        Yh7(i,1)=0;
    end
end

%[X1,Y1,T1,AUC1]=perfcurve(Ytest,Yh1,1);
%[X2,Y2,T2,AUC2]=perfcurve(Ytest,Yh2,2);
%[X3,Y3,T3,AUC3]=perfcurve(Ytest,Yh3,3);
%[X4,Y4,T4,AUC4]=perfcurve(Ytest,Yh4,4);
%[X5,Y5,T5,AUC5]=perfcurve(Ytest,Yh5,5);
%[X6,Y6,T6,AUC6]=perfcurve(Ytest,Yh6,6);
%[X7,Y7,T7,AUC7]=perfcurve(Ytest,Yh7,7);

%plot(X1,Y1)
%hold on
%plot(X2,Y2)
%hold on
%plot(X3,Y3)
%hold on
%plot(X4,Y4)
%hold on
%plot(X5,Y5)
%hold on
%plot(X6,Y6)
%hold on
%plot(X7,Y7)
%legend('ROC Y=1','ROC Y=3','ROC Y=4','ROC Y=5','ROC Y=6','ROC Y=7')
%xlabel('False positive rate'); ylabel('True positive rate');
%title('ROC Curves')
hold off

testX = Xtest;
testY = Ytest;

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 6;
% function_type = 'c'; %'c' - classification, 'f' - regression  
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e_RBF,s_RBF,t_RBF] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);

kernel_type = 'lin_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e_lin,s_lin,t_lin] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);

kernel_type = 'poly_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e_poly,s_poly,t_poly] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);