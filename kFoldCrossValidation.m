function avgMSE = kFoldCrossValidation(train_x, train_y, lambda, k)
% train_x - Input to the function to be regression. M (rows) * N (cols)
% train_y - Target value of train data. M (rows) * 1
% lambda - regularization parameter
% k - number of fold

errors = zeros(1, k);
fold_size = ceil(size(train_x, 1)/ k);
for i = 1: k
    % split data
    n1 = ((i-1)*fold_size+1);
    n2 = min(i*fold_size, size(train_x, 1));
    testx = train_x(n1:n2, :);
    trainx = [train_x(1:n1-1,:); train_x(n2+1:end, :)];
    testy = train_y(n1:n2,:);
    trainy = [train_y(1:n1-1,:);train_y(n2+1:end,:)];
    % predict data
    predy = RidgeRegression(trainx, trainy, testx, lambda);
    % compute MSE
    MSE = 0;
    for j = 1:n2-n1
        MSE = MSE + (predy(j)-testy(j))^2;
    end
    MSE = MSE/(n2-n1);
    errors(i) = MSE;
end
avgMSE = sum(errors)/k;

end
