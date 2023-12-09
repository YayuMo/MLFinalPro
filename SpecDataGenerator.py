import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# constants
CLASS_FILE = 'data/BVCGender_AgeData/BVC_Voice_Bio_Public.xlsx'
BASE_DIR = [
    'data/BVCGender_AgeData/multiple_sentences/multiple_sentences/',
    'data/BVCGender_AgeData/one_sentence/one_sentence/',
    'data/BVCGender_AgeData/S_02_voice/S_02/multiple_sentences/'
]
OUTPUT_DIR = 'dataset/BVCData/'

# convert wav file to Spectrogram
def wav2Spectrogram(filepath, outpath, framelength, type):
    # type -- 'General': Spec, 'MEL': MEL Spec
    # load audio
    y, fs = librosa.load(filepath, sr=None, mono=False)
    # number of fft
    nfft = int(framelength * fs)
    if(type == 'General'):
        # determine window using Hamming function
        window = np.hamming(M=nfft)
        # generate spectrogram
        plt.specgram(y[0,:], NFFT=nfft, Fs=fs, window=window)
        plt.axis('off')
        plt.savefig(outpath)
        # save image according to categories
    elif(type == 'MEL'):
        # extract MEL feature
        mel_spec = librosa.feature.melspectrogram(y[0,:], sr=fs, n_fft=nfft)
        # convert to dB
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # draw MEL
        librosa.display.specshow(mel_spec, sr=fs, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(outpath)

# generate Audio-Sex dict
def dataArr2Dic(basedir, classfile):
    # get dataArr
    dataArr = []
    for dir in basedir:
        dataArr.append(os.listdir(dir))
    # construct Audio-Sex dict
    df = pd.read_excel(classfile)
    df.New_ID = (df['New_ID']).astype(str)
    df.Sex = df['Sex'].apply(lambda x:x.strip("'") if isinstance(x, str) else x)
    dataDic = df.set_index('New_ID')['Sex'].to_dict()
    return dataArr, dataDic

# generate spectrogram dataset
def dataGenerate(dataArr, dataDic, type):
    for index in tqdm(range(len(dataArr))):
        for filename in tqdm(dataArr[index],position=0, leave=True):
            category = filename[5:9]
            filepath = os.path.join(BASE_DIR[index],filename)
            outdir = os.path.join(OUTPUT_DIR + type + '/',dataDic[category])
            imgname = filename.split('.')[0] + '.png'
            outpath = os.path.join(outdir+'/', imgname)
            wav2Spectrogram(
                filepath,
                outpath,
                framelength=0.025,
                type=type
            )
    pass

if __name__ == '__main__':
    dataArr,dataDic = dataArr2Dic(BASE_DIR, CLASS_FILE)
    dataGenerate(dataArr, dataDic, type='General')
    dataGenerate(dataArr, dataDic, type='MEL')