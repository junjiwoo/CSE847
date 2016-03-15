function pred = RidgeRegression(train_x, train_y, test_x, lambda)
% train_x - Input to the function to be regression. M (rows) * N (cols)
% train_y - Target value of train data. M (rows) * 1
% test_x - Test data. D (rows) * N (cols)
% lambda - regularization parameter

%% test dimensions
if size(train_x, 2) ~= size(test_x, 2)
    fprintf('\nColumns of train data and test data are unequal\n');
    return
if size(train_x, 1) ~= size(train_y, 1)
    fprintf('\nRows of train data and target value are unequal\n');
    return

%% training
M = size(train_x, 1);
N = size(train_x, 2);
[U, D, V] = svd(train_x);
w = zeros(1, M);

for i = 1: M
    sigma_i = D(i, i);
    u_i = U(:, i);
    v_i = V(:, i);
    w = w + ((sigma_i * u_i * train_y)/(sigma_i^2 + lambda)) * v_i
end

%% predicting
pred = test_x * w

end
