
% DEn här funkar !!!

g1=GaussD('Mean',[1.0, 5.0],'Covariance',[2, 1;  1, 4]);  % 2X2
g2=GaussD('Mean',[1.0, 5.0],'Covariance',eye(2,2));       % 2x2
mc=MarkovChain([0.75;0.25], [0.99 0.01;0.03 0.97]);
h=HMM(mc, [g1; g2]);

% Den här funkar inte  !!!

g1=GaussD('Mean',[1.0, 5.0],'Covariance',[2, 1;  1, 4]);
g2=GaussD('Mean',[1.0, 5.0,5.0],'Covariance',eye(3,3));  % 3x3
mc=MarkovChain([0.75;0.25], [0.99 0.01;0.03 0.97]);  % 3x3
h=HMM(mc, [g1; g2]);
