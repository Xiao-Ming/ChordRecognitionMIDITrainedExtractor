# Automatic Audio Chord Recognition with MIDI-Traind Deep Feature and BLSTM-CRF Sequence Decoding Model

The source codes used for the experiments presented in our chord recognition work are presented here. Please refer to the following instructions if you want to reproduce the experiments.

##Quick navigations to some important codes

If you are interested in implementation details of the presented model, you may want to see the following source codes:

networks.py -- Definitions of the MIDI-trained feature extractor model and BLSTM-CRF decoder model are written here.
training.py -- Codes for training loops.
const.py  --  Some hyper-parameters are specified here.
CalcHarmonicCQT.py  --  Generates the input spectrograms of the audio data. The shape of the input data is specified here.
ChordVocabulary.py  --  Describes the specifications on the chord vocabulary (how complex chords are reduced to more simple chords, etc).


##Dependencies
The experiments were performed on Python 3.6 and the following libraries were used:

Chainer 4.2.0
Cupy 4.2.0
librosa 0.6.2
mir_eval 0.4

It is ok to use the latest versions of those libraries since (as we know) currently there are no major changes in API from the above versions. 

##To perform chord recognition from raw audio files

Put your audio files (.wav files with sample rate 44.1kHz) in Datas/audios_estimation, and run the script Estimate_Chords_From_Audio.py. 
The pre-trained model will estimate the chord label and results are exported as .lab files in Datas/labs_estimated.

##To train the decoder by yourself (GPU environment required)

##1.Prepare and preprocess the data
Store the audio and annotation files in the following directories:

Data/audios_train  --  Audio files (.wav) for training.
Data/labels_train  --  Annotation files (.lab) for training.

Make sure the file names of audios and annotations are strictly matched(exept for the extensions).

After the data is prepared, run Save_HCQT.py to generate the Harmonic CQT spectrogram of each audio file.

##2.Run training loop
Run Train_Decoder.py.

