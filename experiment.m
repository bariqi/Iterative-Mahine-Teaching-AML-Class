clear all
clc

%% generate data
for i = 1:50
    f1 = randi(10);
    f2 = randi(20);
    f3 = 1;
    kelas1(i,:) = [f1 f2 f3]; 
end
for i = 1:50
    f1 = randi(20);
    f2 = randi(10);
    f3 = 2;
    kelas2(i,:) = [f1 f2 f3]; 
end
Atrans = [kelas1' kelas2'];
DATA = Atrans';

%% Plot Data
% class1 = DATA(:,3) == 1;
% x1 = DATA(:,1);
% x2 = DATA(:,2);
% scatter(x1(class1==1), x2(class1==1), 'b','*');
% hold on;
% scatter(x1(class1~=1), x2(class1~=1), 'r','*');
% axis([0 30 0 30])
% pause

%% Data Preprocessing and find the model
% predictor and target value
Xglobal = DATA(:,1:end-1);
Yglobal = DATA(:,end);

% dividing data training testing
cv = cvpartition(size(DATA,1),'holdout',0.3);
Xtrain = Xglobal(training(cv),:);
Ytrain = Yglobal(training(cv),:);
dataTrain = [Xtrain, Ytrain];
Xtest = Xglobal(test(cv),:);
Ytest = Yglobal(test(cv),:);
rng('default');

% obtain model (<w>)
fitmodel = linReg(Xtrain', Ytrain');
[yfit,sigma] = linRegPred(fitmodel,Xtest');
% plotCurveBar( Xtest', yfit, sigma );
correct = 0;
for i = 1:size(Ytest,1)
    dis(i) = norm(yfit(i)-Ytest(i));
    if(dis(i)<=sigma(i))
        correct = correct + 1;
    end
end
fitacc = correct/size(Ytest,1);

%% start teaching step
% choose data teaching for teach the model
randomValue1 = randi([1,10]);
randomValue2 = randi([size(find(dataTrain(:,3)==1),1)+1,70]);
randomValue3 = randi([11,size(find(dataTrain(:,3)==1),1)]);
% randomValue1 = 1;
% randomValue2 = 2;
% randomValue3 = size(find(dataTrain(:,3)==1),1);
% randomValue4 = size(find(dataTrain(:,3)==1),1)+1;


dataTeaching = [];
dataTeaching(1,:) = dataTrain(randomValue1,:);
dataTeaching(2,:) = dataTrain(randomValue2,:);
dataTeaching(3,:) = dataTrain(randomValue3,:);
% dataTeaching(4,:) = dataTrain(randomValue4,:);
dataTrain(randomValue1,:)=[];
dataTrain(randomValue2,:)=[];
dataTrain(randomValue3,:)=[];
% dataTrain(randomValue4,:)=[];

% model for data teaching
currentmodel = linReg(dataTeaching(:,1:end-1)', dataTeaching(:,end)');
currentnorm = norm([currentmodel.w' currentmodel.w0] - [fitmodel.w' fitmodel.w0]);

% start teaching
averageNow = mean(dataTeaching);
pos = 4;
while(currentnorm > 0.2 && size(dataTrain,1)~=0)
    indexChosen = 1;
    maxNorm = 0;
    
    % choose next data teaching
    for i = 1:size(dataTrain,1)
        lihat(i) = norm(dataTrain(i,1:end-1)-averageNow(:,1:end-1));
        if(lihat(i)>maxNorm)
            indexChosen = i;
        end
    end
    % add data teaching from data training
    dataTeaching(pos,:) = dataTrain(indexChosen,:);
    averageNow = mean(dataTeaching);
    dataTrain(indexChosen,:) = [];
    pos = pos + 1;
    
    currentmodel = linReg(dataTeaching(:,1:end-1)', dataTeaching(:,end)');
    currentnorm = norm([currentmodel.w' currentmodel.w0] - [fitmodel.w' fitmodel.w0]);
end

tic
%% Final model
finmodel = currentmodel;
[yfin,sigma] = linRegPred(currentmodel,Xtest');
% plotCurveBar( Xtest', yfin, sigma );
correct = 0;
for i = 1:size(Ytest,1)
    dis(i) = norm(yfin(i)-Ytest(i));
    if(dis(i)<=sigma(i))
        correct = correct + 1;
    end
end
finacc = correct/size(Ytest,1);
toc
fprintf('%d data training is used and obtain accuracy %f (with teacher).\n',size(dataTeaching,1),finacc);
fprintf('all data training is used and obtain accuracy %f (without teacher).\n',fitacc);