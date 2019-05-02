
function [] = soundAnalysis()
% TODO Compute all fatures with their parameters and estimate the 
% % plot play 
% plotPlay();
% % plot Frequency and intensity features 
% plotFreqIntensity()

% IntegrationWithOuputDistribution()
InsensibilityToTransposition()

end 

function [] = InsensibilityToTransposition()

[signal,fs] = audioread('melody_1.wav');
info = audioinfo('melody_1.wav');
w_size = 0.03 ;

feature_mat = GetMusicFeatures(signal,fs,0.03);
frequency = feature_mat(1,:);
correlation = feature_mat(2,:);
intensity = feature_mat(3,:);

% Plot of the energy feature for the Original intensty 
% and energy feature of intensity
[fft_mag, fft_angle] = feat_fft(frequency,20, 5);
[C,S] = fft_centroid(frequency,20, 5,fs);
E = ShortTimeEnergy(intensity, 2,1);
loud = loudness(intensity,20,5);
plotframes(0.5*w_size*(1:length(intensity)),intensity, 'Intensity rate', 'Intensity Melody 1');
plotframes(0.5*w_size*(1:length(E)),E, 'Energy rate', 'Energy Melody 1');

% The transposed intensity 
intensity = transpose(intensity);
% Plot of the energy feature for the transposed intensity
[fft_mag, fft_angle] = feat_fft(frequency,20, 5);
[C,S] = fft_centroid(frequency,20, 5,fs);
E = ShortTimeEnergy(intensity, 2,1);
loud = loudness(intensity,20,5);
plotframes(0.5*w_size*(1:length(E)),E, 'Energy rate', 'Energy transposed melody 1');

end

function [] = IntegrationWithOuputDistribution()

% This shows a possible intengration with the output distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,coeffs,delta,deltaDelta, z] = Extract_all_features('melody_3.wav');

[Mean1, Covar1] = mean_cov(coeffs);
[Mean2, Covar2] = mean_cov(delta);
[Mean3, Covar3] = mean_cov(deltaDelta);
[Mean4, Covar4] = mean_cov(z);

g1 = GaussD('Mean',Mean1,'Covariance',Covar1);
g2 = GaussD('Mean',Mean2,'Covariance',Covar2);
g3 = GaussD('Mean',Mean3,'Covariance',Covar3);
g4 = GaussD('Mean',Mean4,'Covariance',Covar4);
[pi, A] = RandomMarkovChain(4);
mc=MarkovChain(pi, A);
h =HMM(mc, [g1;g2;g3;g4]);
[X,S]=rand(h,5)

% F1 = size(coeffs)
% F2 = size(delta)
% F3 = size(deltaDelta)
% F4 = size(z)

end

function [feats_out] = transpose(feats_in)
%  Compute the transpostion 
% of a pitch in input parameters feats_in
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute average frequency
avg_freq = mean(feats_in);
% Align average to C3 (or choose any other tone)
shift = 130.83/avg_freq;
feats_in = shift * feats_in;
feats_out = feats_in;

end

function [pi, A] = RandomMarkovChain(n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create a random markov chain 
% where n = numberOfStates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
state_distrubition = randi([1,n*4],1,n);
pi = state_distrubition / sum(state_distrubition);

transition_mat = randi([1,n*4],n,n);
for i=1:n 
    transition_mat(i,:) = transition_mat(i,:) / sum(transition_mat(i,:));
end 

A = transition_mat;

end

function [Mean, Covar] = mean_cov(data)
Mean = mean(data);
Covar = cov(data);
end

function [feats,coeffs,delta,deltaDelta, feats_part] = Extract_all_features(sound)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extraction of all the features where : sound is a wave file with
% extensions ('.wav'). 
% Feats -> All the feaures from 
% Coeffs -> coefficintes features for MCFF
% Delta -> difference for coeffs
% deltaDelta -> difference for Delta
% feats_part -> feats redimensioned in order to create features with same
% dimension
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[signal,fs] = audioread(sound);
% info = audioinfo(sound);

feature_mat = GetMusicFeatures(signal,fs,0.03);
frequency = feature_mat(1,:);
correlation = feature_mat(2,:);
intensity = feature_mat(3,:);

[fft_mag, fft_angle] = feat_fft(frequency,20, 5);
[C,S] = fft_centroid(frequency,20, 5,fs);
E = ShortTimeEnergy(intensity, 20,5);
loud = loudness(intensity,20,5);
[coeffs,delta,deltaDelta,~] = mfcc(signal,fs);

[~,n] = size(coeffs);

feats = [E,loud,C,fft_mag,fft_angle,S];
feats_part = partition(feats, n); 

end 

function [z]= partition(v,nparties)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function partition a matrix v with NxM where the matrix is divided 
% in nparties {N1xM, N2xM, ..., Nnparties x M}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% v = rand(8812,1);
[N,~] = size(v);
% N = numel(v)
b = ceil((N /nparties )-0.5); % block size
c = mat2cell(v,diff([0:b:N-1,N]));
% z = cellfun(@mean,c);
z = cellfun(@mean,c,'UniformOutput',false);
z = cell2mat(z)';
z = z(:,1:nparties);

end 

function [] = plotfeat_features_evolution()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function shows the time evolution of all features 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Melody 1 
[signal,fs] = audioread('melody_1.wav');
info = audioinfo('melody_1.wav');

feature_mat = GetMusicFeatures(signal,fs,0.03);
Mel1 = size(feature_mat)
frequency = feature_mat(1,:);
correlation = feature_mat(2,:);
intensity = feature_mat(3,:);

[fft_mag, fft_angle] = feat_fft(frequency,20, 5);
[C,S] = fft_centroid(frequency,20, 5,fs);
E = ShortTimeEnergy(intensity, 20,5);
loud = loudness(intensity,20,5);
plotframes((1:length(E)),E, 'Energy rate', 'Energy Melody 1');
plotframes((1:length(loud)),loud, 'Loudness rate', 'Loudness Melody 1');
plotframes((1:length(C)),C, 'Centroid', 'Spectrum centroid Melody 1');
plotframes((1:length(S)),S, 'Spread', 'Spectrum spread Melody 1');
plotframes((1:length(fft_mag)),fft_mag, 'Spread', 'FFT freq Melody 1');
plotframes((1:length(fft_angle)),fft_angle, 'Spread', 'FFT angle Melody 1');

% Melody 2
[signal,fs] = audioread('melody_2.wav');
info = audioinfo('melody_2.wav');

feature_mat = GetMusicFeatures(signal,fs,0.03);
Mel2 = size(feature_mat)
frequency = feature_mat(1,:);
correlation = feature_mat(2,:);
intensity = feature_mat(3,:);

[fft_mag, fft_angle] = feat_fft(frequency,20, 5);
[C,S] = fft_centroid(frequency,20, 5,fs);
E = ShortTimeEnergy(intensity, 20,5);
loud = loudness(intensity,20,5);
plotframes((1:length(E)),E, 'Energy rate', 'Energy Melody 2');
plotframes((1:length(loud)),loud, 'Loudness rate', 'Loudness Melody 2');
plotframes((1:length(C)),C, 'Centroid', 'Spectrum centroid Melody 2');
plotframes((1:length(S)),S, 'Spread', 'Spectrum spread Melody 2');
plotframes((1:length(fft_mag)),fft_mag, 'Spread', 'FFT freq Melody 2');
plotframes((1:length(fft_angle)),fft_angle, 'Spread', 'FFT angle Melody 2');


% Melody 3
[signal,fs] = audioread('melody_3.wav');
info = audioinfo('melody_3.wav');

feature_mat = GetMusicFeatures(signal,fs,0.03);
Mel3 = size(feature_mat)
frequency = feature_mat(1,:);
correlation = feature_mat(2,:);
intensity = feature_mat(3,:);

[fft_mag, fft_angle] = feat_fft(frequency,20, 5);
[C,S] = fft_centroid(frequency,20, 5,fs);
E = ShortTimeEnergy(intensity, 20,5);
loud = loudness(intensity,20,5);
plotframes((1:length(E)),E, 'Energy rate', 'Energy Melody 3');
plotframes((1:length(loud)),loud, 'Loudness rate', 'Loudness Melody 3');
plotframes((1:length(C)),C, 'Centroid', 'Spectrum centroid Melody 3');
plotframes((1:length(S)),S, 'Spread', 'Spectrum spread Melody 3');
plotframes((1:length(fft_mag)),fft_mag, 'Spread', 'FFT freq Melody 3');
plotframes((1:length(fft_angle)),fft_angle, 'Spread', 'FFT angle Melody 3');

end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------- Intensity Related features -------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function E = ShortTimeEnergy(signal, windowLength,step)
signal = signal / max(max(signal));
curPos = 1;
L = length(signal);
numOfFrames = floor((L-windowLength)/step) + 1;
E = zeros(numOfFrames,1);
for (i=1:numOfFrames)
    window = (signal(curPos:curPos+windowLength-1));
    E(i) = (1/(windowLength)) * sum(abs(window.^2));
    curPos = curPos + step;
end

end 

function loud = loudness(signal, windowLength, step)

signal = signal / max(max(signal));
curPosition = 1;
L = length(signal);
numOfFrames = floor((L-windowLength)/step) + 1;
%H = hamming(windowLength);
loud = zeros(numOfFrames,1);
for (i=1:numOfFrames)
    window = (signal(curPosition:curPosition+windowLength-1));
    
    loud(i) = sum(abs(window.^2));
    loud(i) = loud(i).^0.67;
    
    curPosition = curPosition + step;
end

end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------ Frequency domain Related features ---------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [FFT,f] = dftGet(signal, fs)

N = length(signal);
FFT = (1/N)*abs(fft(signal));
f = (1:N)/fs;

end 

function [C,S] = fft_centroid(signal, windowLength, step,fs)

signal = signal / max(max(signal));
curPos = 1;
L = length(signal);
numOfFrames = floor((L-windowLength)/step) + 1;
%H = hamming(windowLength);
C = zeros(numOfFrames,1);
S = zeros(numOfFrames,1);
for (i=1:numOfFrames)
    window = (signal(curPos:curPos+windowLength-1));
    
    [w_FFT,~] = dftGet(window, fs) ;
    Wl= length(w_FFT);

    m = ((fs/(2*Wl))*[1:Wl])';

    w_FFT = w_FFT / max(w_FFT);
    Ci =  sum(m.*w_FFT)/ (sum(w_FFT)+eps) ;
    Si =  sqrt(sum(((m-Ci).^2).*w_FFT)/ (sum(w_FFT)+eps));
    C(i) = sum(Ci) / (fs/2);
    S(i) = sum(Si) / (fs/2);
    
    curPos = curPos + step;
end

end

function [fft_mag, fft_angle] = feat_fft(signal, windowLength,step)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature for fourier transform magnetude and angle.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

signal = signal / max(max(signal));
curPos = 1;
L = length(signal);
numOfFrames = floor((L-windowLength)/step) + 1;
fft_mag = zeros(numOfFrames,1);
fft_angle = zeros(numOfFrames,1);
for (i=1:numOfFrames)
    window = (signal(curPos:curPos+windowLength-1));
    fft_mag(i) = sum(abs(fft(window))); %Identifies pitch
    fft_angle(i) = sum(angle(fft(window)));%Identifies pauses
    curPos = curPos + step;
end

end

function [] = plotFreqIntensity()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of the Frequency (Pitch) and intensity profiles for all the three
% melodies in the dataset.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Melody 1 
[signal,fs] = audioread('melody_1.wav');
info = audioinfo('melody_1.wav');

w_size = 0.03 ;
feature_mat = GetMusicFeatures(signal,fs,0.03);
frequency = feature_mat(1,:);
% Change Frames (samples) to time (sec)
x = 0.5*w_size*(1:length(frequency));
plotframes(x,frequency, 'Frequency [Hz]', 'Frequency Melody 1');
correlation = feature_mat(2,:);
intensity = feature_mat(3,:);
plotframes(x,intensity, 'Intensity [W/m^2]','Intensity Melody 1');

% Melody 2
[signal,fs] = audioread('melody_2.wav');
info = audioinfo('melody_2.wav');
w_size = 0.03 ;
feature_mat = GetMusicFeatures(signal,fs,0.03);
frequency = feature_mat(1,:);
% Change Frames (samples) to time (sec)
x = 0.5*w_size*(1:length(frequency));
plotframes(x,frequency, 'Frequency [Hz]', 'Frequency Melody 2');
correlation = feature_mat(2,:);
intensity = feature_mat(3,:);
plotframes(x,intensity, 'Intensity [W/m^2]','Intensity Melody 2');

% Melody 3
[signal,fs] = audioread('melody_3.wav');
info = audioinfo('melody_3.wav');
w_size = 0.03 ;
feature_mat = GetMusicFeatures(signal,fs,0.03);
frequency = feature_mat(1,:);
% Change Frames (samples) to time (sec)
x = 0.5*w_size*(1:length(frequency));
plotframes(x,frequency, 'Frequency [Hz]', 'Frequency Melody 3');
correlation = feature_mat(2,:);
intensity = feature_mat(3,:);
plotframes(x,intensity, 'Intensity [W/m^2]','Intensity Melody 3');

end 

function [] = plotPlay()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The plot of each melody signal, the real signal with extensions .wav
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read first song Medody 1
[x1,Fs1] = audioread('melody_1.wav');
info = audioinfo('melody_1.wav');
plotSamples(x1,Fs1, info.Duration,info.Title );
% sound(x1,Fs1);

% Read first song Medody 2
[x2,Fs2] = audioread('melody_2.wav');
info = audioinfo('melody_2.wav');
plotSamples(x2,Fs2, info.Duration,info.Title );

% Read first song Medody 3
[x3,Fs3] = audioread('melody_3.wav');
info = audioinfo('melody_3.wav');
plotSamples(x3,Fs3, info.Duration,info.Title );
sound(x3,Fs3);

end 

function [] = plotframes(x,y, ylab, titl)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the frames or time signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
% fig = gcf;
% fig.Color = randi([0,1],1,3);
s = randi([0,1],1,3);
s = randi([0,1],1,3);
plot(x,y, 'Color',s);
xlabel('Time (Sec) ');
ylabel(ylab);
title(titl);
% titl = titl(find(~isspace(titl)));
% saveas(gcf,strcat(titl,'.png'))
end 

function [] = plotSamples(Xs,Fs, durat, titl)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the audio signal used in function plotPlay() above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ts = 1 / Fs;
time = 0:Ts:durat;
[alpha, omega] = size(Xs');

figure;
% plot(time(1:end-1),Xs');
plot(time(1:omega),Xs');
axis([0 time(end) -1 1]);
xlabel('Time (sec)');
ylabel('Signal Ampl.');
title(titl);

end 