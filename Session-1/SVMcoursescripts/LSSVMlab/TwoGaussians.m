X1=1+randn(50,2);
X2=-1+randn(51,2);
Y1=ones(50,1);
Y2=-ones(51,1);
X=[X1;X2];
Y=[Y1;Y2];
SVMModel = fitcsvm(X,Y,'KernelFunction','rbf','Boxconstraint',500)
figure;
hold on;
plot(X1(:,1),X1(:,2),'ro');
plot(X2(:,1),X2(:,2),'bo');
hold off;


sv = SVMModel.SupportVectors;
figure
gscatter(X(:,1),X(:,2),Y)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('versicolor','virginica','Support Vector')
hold off