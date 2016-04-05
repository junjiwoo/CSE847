%Problem1:"Logistic Regression: Experiment"

data = load('data.txt');
labels = load('labels.txt');

data = [data , ones(size(data,1),1)]; %Add bias

train_data = data(1:2000, :);
train_labels = labels(1:2000, :);
test_data = data(2001:4601, :);
test_labels = labels(2001:4601, :);

train_sizes = [200; 500; 800; 1000; 1500; 2000];
test_accuracy = zeros(6,1);

for k=1:6
    train_size = train_sizes(k,1);
    
    [weights] = logistic_train(train_data(1:train_size, :), train_labels(1:train_size,1));
    
    test_outcome= test_data*weights;
    
    for i=1:size(test_outcome,1)
        if(test_outcome(i,1)>0)
            test_outcome(i,1)=1;
        else
            test_outcome(i,1)=0;
        end
    end
    
    erro= sum(abs(test_outcome- test_labels));
    
    test_accuracy(k,1)= (size(test_labels,1)-erro)/size(test_labels,1);
 
end

plot(train_sizes, test_accuracy,'-d');
title('Problem 1');
ylabel('Classification Accuracy');
xlabel('Train Size');
