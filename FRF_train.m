function Model_FRF = FRF_train(Xtrain, Ytrain, n_tree,mtree,min_leaf, D, Command)
% FRF performs Functional Random Forest which predicts drug sensitivity
% measure "Area under the Curve (AUC)" using functional data like dose
% response curve points
%     Input:
%         Xtrain: N x P design matrix, where N is the number of samples for training
%                 and p is the number of predictor
%         Ytrain: N x Q response matrix, where Q is of length (1+2*D) and D is number of dose response points.
%                 In Ytrain, first column should contain drug sensitivity measure AUC,
%                 next D column contains median response points of D doses and
%                 last D column contains standard deviation of response
%                 points of D doses. If standard deviation of response
%                 points is unavailable, Q will be of length (1+D).
%         n_tree: Number of trees in the forest, for sample size 200 or less, it can be 50,
%                 for sample size 200 to 500, it can be 100 and
%                 for sample size higher than 500, it can be 150.
%         mtree:  Number of randomly picked features (m) out of all features (M)for node splitting.
%         min_leaf: Number of maximum samples in the leaf nodes of the trees.
%         D:      Number of dose response points or, in general number of
%                 functional data points
%         Command:There can be three kinds of Command,
%                 (i)"FRF_points" which uses dose response points for node cost calculation
%                 (ii)"FRF_dist_KL" which uses dose response distribution with KL divergence for node cost calculation
%                 (iii)"FRF_dist_Hell" which uses dose response distribution with Hellinger distance for node cost calculation
%                 Remember to perform command (ii) and (iii), standard deviation of response points is necessary.
% %   Output:
%         Model_FRF: Functional Random Forest model
%
% %   Example:
%         Xtrain= rand(100,1000);
%         Ytrain= rand(100,11);
%         n_tree= 20;
%         mtree=  10;
%         min_leaf= 5;
%         D=      5;
%         Command='FRF_points';
%         Model_FRF = FRF_train(Xtrain, Ytrain, n_tree,mtree,min_leaf, D, Command)

if isempty(n_tree)
    n_tree=min(150,floor(size(Xtrain,1)/3));
end
if isempty(mtree)
    mtree=min(20,floor(size(Xtrain,2)/1000));
end
if isempty(min_leaf)
    min_leaf=max(5,floor(size(Xtrain,1)/50));
end
if Command=="FRF_dist_KL" || Command=="FRF_dist_Hell"
    if size(Ytrain,2)<(1+2*D)
        error('standard deviation of response points is missing!!!\n')
    end
end

for SS=1:size(Ytrain,1)
    [Spline, ~]=least_square_spline2(1:D,Ytrain(SS,2:D+1),4); % Basis Order of B-spline curve is 4
    Curve_values(SS,:)=fnval(Spline,1:0.01:D);
    if Command=="FRF_points"
        Ytrain2(SS,:)=[Ytrain(SS,1) Ytrain(SS,2:D+1)  mean(0-min(0,Curve_values(SS,:)./100)) Curve_values(SS,:)];
    elseif Command=="FRF_dist_KL" || Command=="FRF_dist_Hell"
        Ytrain2(SS,:)=[Ytrain(SS,1) Ytrain(SS,2:D+1) Ytrain(SS,D+2:end) mean(0-min(0,Curve_values(SS,:)./100)) Curve_values(SS,:)];
    end
end

Model_FRF = build_forest(Xtrain, Ytrain2, n_tree, mtree, min_leaf,D, Command);