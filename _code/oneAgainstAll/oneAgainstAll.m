clear all;
load _data/svmData;
kernel = 'rbf';
treshold = 1; verbose = 0;

classesStr = ["(N", "(AFIB", "(SBR", "(B" ];
numberClasses = length(classesStr);

svmData = normalize(svmData, 1);

[Xtrain, Xtest, Ytrain, Ytest] = loadData(svmData, labels, classesStr);

% Train    
modelsOAA = trainOAAModels(Xtrain, Ytrain, numberClasses, kernel, verbose);

% Predict
predictLabel = predictOAAModels(modelsOAA, Xtest);

[cmat, order] = confusionmat(Ytest,predictLabel);
acc = 100*sum(diag(cmat))./sum(cmat(:));
fprintf('SVM (1-against-A):\naccuracy = %.2f%%\n', acc);
fprintf('Confusion Matrix:\n'), disp(order'), disp(cmat)
disp(classesStr)

precision = 100*(diag(cmat)'./sum(cmat))
recall = 100*(diag(cmat)'./sum(cmat, 2)')

%save('_data/_result/oaaRbf.mat', 'modelsOAA', 'Xtest', 'predictLabel', 'Ytest', 'cmat', 'order')
