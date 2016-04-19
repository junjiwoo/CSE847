load 'USPS.mat';
p = [10 50 100 200];
k = length(p);

[coeff,score,latent] = pca(A);

error = zeros(k,1);

figure;
hold on;

for i=1:k
    [~,reconstructed] = pcares(A,p(i));
   
    error(i) = sum(sum((reconstructed-A).^2, 2));
   
    A2 = reshape(reconstructed(1,:), 16, 16);
    subplot(k,3,i*3-1);
    imshow(A2');
    
    A2 = reshape(reconstructed(2,:), 16, 16);
    subplot(k,3,i*3);
    imshow(A2');
end
hold off;

figure;
plot(p, error, 'b-');
title('Reconstruction Error for different P');
xlabel('P');
ylabel('Reconstruction Error');
