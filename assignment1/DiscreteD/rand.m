function R=rand(pD,nData)

%R=rand(pD,nData) returns random scalars drawn from given Discrete Distribution.
%
%Input:
%pD=    DiscreteD object
%nData= scalar defining number of wanted random data elements
%
%Result:
%R= row vector with integer random data drawn from the DiscreteD object pD
%   (size(R)= [1, nData]
%
%----------------------------------------------------
%Code Authors:
%----------------------------------------------------

if numel(pD)>1
    error('Method works only for a single DiscreteD object');
end

%*** Insert your own code here and remove the following error message 
% change pD to probability mass
[m,n] = size(pD.ProbMass);
prob = reshape(pD.ProbMass,n,m);
Pnorm=[0 prob]/sum(prob);
Pcum=cumsum(Pnorm);

N=round(1);
M=round(nData);
Uniform =rand(1,N*M);

V=1:length(prob);
[~,inds] = histc(Uniform,Pcum); 

R = V(inds);
R =reshape(R,N,M);

