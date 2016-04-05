%Problem1:"Sparse Logistic Regression: Experiment"

clear;

load('ad_data.mat');

opts.rFlag= 1;
opts.tol= 1e-6;
opts.tFlag= 4;
opts.maxIter= 5000;


pars= [1e-8; 0.01; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
pars_num_selected= zeros(size(pars,1),1);

AUCs= zeros(size(pars,1),1);

for i=1:size(pars,1)
    num_selected=0;
    [w,c]= LogisticR(X_train, y_train, pars(i,1), opts);
    for j=1:size(w,1)
        if (w(j,1)~=0)
            num_selected= num_selected+1;
        end
    end
    pars_num_selected(i,1)= num_selected;
    
    y_outcome= X_test*w + c;
    for k=1:size(y_outcome,1)
        if(y_outcome(k,1)>0)
            y_outcome(k,1)=1;
        else
            y_outcome(k,1)=-1;
        end
    end
    
    [X, Y, T, AUC]= perfcurve(y_test, y_outcome, 1);
    AUCs(i,1)=AUC;
end

plot(pars, AUCs, '-d');
title('AUC and Regularization Parameters');
xlabel('Regularization Parameters'); 
ylabel('AUC');


