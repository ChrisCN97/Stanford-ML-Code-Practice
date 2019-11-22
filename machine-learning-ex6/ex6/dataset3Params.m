function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
cl = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sl = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
C = 1;
sigma = 0.3;
pre = 0;
count = 1;

for ct = 1:length(cl)
  for st = 1:length(sl)
    printf("%d/64 C=%d sigma=%d pre=%d", count,C,sigma,pre);
    count = count+1;
    model = svmTrain(X, y, cl(ct), @(x1, x2) gaussianKernel(x1, x2, sl(st)));
    pred = svmPredict(model, Xval);
    pret = mean(double(pred ~= yval));
    if (count==2)
      pre = pret;
    elseif (pret<pre)
      C = cl(ct);
      sigma = sl(st);
      pre = pret;
    endif
  endfor
endfor

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
