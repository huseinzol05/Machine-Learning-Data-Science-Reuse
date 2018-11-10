import speechpy
import scipy.io.wavfile as wav
import numpy as np

def extract_features(signal, fs):
    frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, filter=lambda x: np.ones((x,)),zero_padding=True)
    power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=1)
    logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,num_filters=1, fft_length=512, low_frequency=0, high_frequency=None)
    mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,num_filters=1, fft_length=512, low_frequency=0, high_frequency=None)
    mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)
    mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
    return np.hstack([power_spectrum[:,0],logenergy[:,0],mfcc_cmvn[:,0],mfcc_feature_cube[:,0,1]])

fs, signal = wav.read(sound_file_wav)
print(extract_features(signal, fs))
