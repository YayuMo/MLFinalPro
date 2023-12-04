import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def wav2Mel(filepath, framelength):
    # load audio
    y, fs = librosa.load(filepath, sr=None, mono=False)

    # time of audio
    L = len(y[0,:])
    print('Time:', L/fs)
    # number of fft
    nfft = int(framelength * fs)
    print('NFFT:', nfft)

    # extract MEL feature
    mel_spect = librosa.feature.melspectrogram(y[0,:], sr=fs, n_fft=nfft)
    # convert to dB
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    # draw mel
    librosa.display.specshow(mel_spect, sr=fs, x_axis='time', y_axis='mel')
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time(s)')
    plt.title('Mel Spectrogram')
    plt.show()


if __name__ == '__main__':
    # length of each frame
    framelength = 0.025
    path = 'data/BVCGender_AgeData/one_sentence/one_sentence/S_01_4001_VE.wav'
    wav2Mel(path, framelength)