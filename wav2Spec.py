import librosa
import matplotlib.pyplot as plt
import numpy as np

def wav2Spec(filepath, framelength):
    # load audio
    y, fs = librosa.load(filepath, sr=None, mono=False)

    # time of audio
    L = len(y[0,:])
    print('Time:', L/fs)
    # number of fft
    nfft = int(framelength * fs)
    print('NFFT:', nfft)

    window = np.hamming(M=nfft)

    plt.specgram(y[0,:], NFFT=nfft, Fs=fs, window=window)
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    plt.title('Spectrogram')
    plt.show()


if __name__ == '__main__':
    # length of each frame
    framelength = 0.025
    path = 'data/BVCGender_AgeData/one_sentence/one_sentence/S_01_4001_VE.wav'
    wav2Spec(path, framelength)