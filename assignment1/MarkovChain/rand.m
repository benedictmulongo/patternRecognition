function S=rand(mc,T)
%S=rand(mc,T) returns a random state sequence from given MarkovChain object.
%
%Input:
%mc=    a single MarkovChain object
%T= scalar defining maximum length of desired state sequence.
%   An infinite-duration MarkovChain always generates sequence of length=T
%   A finite-duration MarkovChain may return shorter sequence,
%   if END state was reached before T samples.
%
%Result:
%S= integer row vector with random state sequence,
%   NOT INCLUDING the END state,
%   even if encountered within T samples
%If mc has INFINITE duration,
%   length(S) == T
%If mc has FINITE duration,
%   length(S) <= T
%
%---------------------------------------------
%Code Authors:
%---------------------------------------------

S=zeros(1,T);%space for resulting row vector
nS=mc.nStates;

init_prob = mc.InitialProb';
A = mc.TransitionProb;
[~,state0] = max(init_prob);

terminate = finiteDuration(mc);
S(1) = state0;

% Allocate and initiales Discrete random states 
% for each states
discrete_state = cell(1,nS);
for j=1:nS
    Dr =DiscreteD(A(j,:));
    discrete_state{j} = Dr;
end 

for i=2:T
    temp =  discrete_state{S(i-1)};
    S(i) = temp.rand(1);
    
    if terminate && (S(i) == nS + 1)
       S = S(1:end_state - 1);
       break 
    end
end 

% error('Method not yet implemented');
% %continue code from here, and erase the error message........
% S = cell2mat(X{i});
end 


