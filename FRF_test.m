function [FRF_pred, FRFL_pred] = FRF_test(Model_FRF, Xtest, D, Command)
% FRF performs Functional Random Forest which predicts drug sensitivity
% measure "Area under the Curve (AUC)" using functional data like dose
% response curve points
%     Input:
%         Model_FRF:Functional Random Forest model
%         Xtest:    N_test x P design matrix, where N_test is the number of samples for test
%                   and p is the number of predictor (same as training)
%         D:        Number of dose response points or, in general number of
%                   functional data points
%         Command:  There can be three kinds of Command,
%                   (i)"FRF_points" which uses dose response points for node cost calculation
%                   (ii)"FRF_dist_KL" which uses dose response distribution with KL divergence for node cost calculation
%                   (iii)"FRF_dist_Hell" which uses dose response distribution with Hellinger distance for node cost calculation
%                   Remember to perform command (ii) and (iii), standard deviation of response points is necessary.
% %   Output:
%         FRF_pred:  N_test x (1+D) matrix of the prediction of the testing samples using Functional Random Forest model
%         FRFL_pred: N_test x 1 matrix of the prediction of the testing samples using Functional Random Forest model
%                    with regular averaging at the leaf node
%
% %   Example:
%         Xtest= rand(50,1000);
%         [FRF_pred, FRFL_pred] = FRF_test(Model_FRF, Xtest, D, Command);

addpath(genpath(pwd))

forest = Model_FRF.forest;
n_tree = size(forest,2);

Y_Single_prediction_mean=cell(size(Xtest,1),n_tree);
for X_size = 1:size(Xtest,1)
    x = Xtest(X_size,:);
    for TT = 1:n_tree
        t = forest{1,TT};
        leaf_info = predict(x,t);
        Y_Single_pred_RF(X_size,TT)=mean(leaf_info(:,2));
        if Command=="FRF_points"
            Y_Single_prediction_mean{X_size,TT} = mean((leaf_info(:,(D+4):end)),1)';
        elseif Command=="FRF_dist_KL" || Command=="FRF_dist_Hell"
            Y_Single_prediction_mean{X_size,TT} = mean((leaf_info(:,(2*D+4):end)),1)';
        end
    end
    FRFL_pred(X_size,:)=mean(Y_Single_pred_RF(X_size,:));
    
    Prediction_temp=[Y_Single_prediction_mean{X_size,:}]';
    for jj=1:size(Prediction_temp,2)
        pd{jj} = fitdist(Prediction_temp(:,jj),'Normal');
        Max_prediction(X_size,jj)=pd{jj}.mu;
    end
    Y_hat2(X_size,:)=mean(0-min(0,Max_prediction(X_size,:)./100));
end
FRF_pred = [Y_hat2 Max_prediction(:,1:100:end)];