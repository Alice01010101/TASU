# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import glob
from tqdm import tqdm

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from audioldm.audio import wav_to_fbank, read_wav_file,TacotronSTFT
import soundfile

def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)
import pdb

class SoundSpacesSpeechDataset(Dataset):
    def __init__(self, split, normalize_whole=True, normalize_segment=False,
                 use_real_imag=False, use_rgb=False, use_depth=False, limited_fov=False,
                 remove_oov=False, hop_length=160, use_librispeech=False, convolve_random_rir=False, use_da=False,
                 read_mp4=False):
        self.split = split
        self.data_dir = os.path.join('/home/pantianrui/data1/soundspace-speech', split)
        self.files = glob.glob(self.data_dir + '/**/*.pkl', recursive=True)
        # self.lrs3_files = []
        # with open('/home/pantianrui/lrs3/lrs3/433h_data/train.tsv', 'r') as f:
        #     for line in f.readlines():
        #         self.lrs3_files.append(line.split('\t')[2])

        np.random.shuffle(self.files)
        self.normalize_whole = normalize_whole
        self.normalize_segment = normalize_segment
        self.use_real_imag = use_real_imag
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.limited_fov = limited_fov
        self.hop_length = hop_length

        self.fn_STFT = TacotronSTFT(
            filter_length=1024,
            hop_length=160,
            win_length=1024,
            n_mel_channels=64,
            sampling_rate=16000,
            mel_fmin=0,
            mel_fmax=8000
        )

        if remove_oov:
            """
            Remove cases where the speaker is out of view (or there is no direct sound).
            """
            os.makedirs('data/soundspaces_speech/remove_oov', exist_ok=True)
            tgt_file = os.path.join('data/soundspaces_speech/remove_oov', split + '.pkl')
            if os.path.exists(tgt_file):
                with open(tgt_file, 'rb') as fo:
                    self.files = pickle.load(fo)
                print(f'Read from file {tgt_file}')
            else:
                new_files = []
                for file in tqdm(self.files):
                    with open(file, 'rb') as fo:
                        file_data = pickle.load(fo)
                    speaker_in_view = np.sum(
                        np.concatenate([x == 41 for x in file_data['semantic']], axis=1)) >= 50
                    if speaker_in_view and np.argmax(file_data['rir']) == 0:
                        new_files.append(file)
                self.files = new_files
                with open(tgt_file, 'wb') as fo:
                    pickle.dump(new_files, fo)
                    print(f'Write to file {tgt_file}')
            print(f"Number of files after removing out-of-view cases: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        ####################################################################################
        file = self.files[item]

        with open(file, 'rb') as fo:
            data = pickle.load(fo)

        receiver_audio = data['receiver_audio']
        rir = data['rir']
        source_audio = data['source_audio']
    
        src_wav, recv_wav = self.process_audio(source_audio, receiver_audio)

        src_fbank, src_log_magnitudes_stft, src_energy = self.get_mel_from_wav(src_wav)
        recv_fbank, recv_log_magnitudes_stft, recv_energy = self.get_mel_from_wav(recv_wav)  

        sample = dict()
        sample['src_wav'] = src_wav
        sample['recv_wav'] = recv_wav
        sample['src_fbank'] = src_wav
        sample['recv_fbank'] = recv_wav
        rir_wav = process_rir(rir)
        sample['rir_wav'] = rir_wav

        # stitch images into panorama
        visual_sensors = []
        if self.use_rgb:
            sample['rgb'] = to_tensor(np.concatenate([x / 255.0 for x in data['rgb']], axis=1)).permute(2, 0, 1)
            visual_sensors.append('rgb')
        if self.use_depth:
            sample['depth'] = to_tensor(np.concatenate(data['depth'], axis=1)).unsqueeze(0)
            visual_sensors.append('depth')

        if len(visual_sensors) > 0:
            if self.split == 'train':
                # data augmentation, randomly shifting the panorama image
                width_shift = None
                for visual_sensor in visual_sensors:
                    if width_shift is None:
                        width_shift = np.random.randint(0, sample[visual_sensor].shape[-1])
                    sample[visual_sensor] = torch.roll(sample[visual_sensor], width_shift, dims=-1)

            if self.limited_fov:
                # crop image to normal FOV with size 384 * 256
                if self.split == 'train':
                    offset = None
                    for visual_sensor in visual_sensors:
                        if offset is None:
                            offset = np.random.randint(0, sample[visual_sensor].shape[-1] - 256)
                        sample[visual_sensor] = sample[visual_sensor][:, :, offset: offset + 256]
                else:
                    for visual_sensor in visual_sensors:
                        sample[visual_sensor] = sample[visual_sensor][:, :, :256]

            else:
                for visual_sensor in visual_sensors:
                    sample[visual_sensor] = torchvision.transforms.Resize((192, 576))(sample[visual_sensor])

        return sample
    
    def get_mel_from_wav(self,audio):
        audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec, log_magnitudes_stft, energy = self.fn_STFT.mel_spectrogram(audio)
        return melspec, log_magnitudes_stft, energy

    # def process_audio(self, source_audio, receiver_audio):
    #     # normalize the intensity before padding
    #     waveform_length = 41600

    #     if self.normalize_whole:
    #         receiver_audio = normalize(receiver_audio, normalize_whole=True, norm='peak')
    #         source_audio = normalize(source_audio, normalize_whole=True, norm='peak')

    #     # pad audio
    #     if waveform_length > source_audio.shape[0]:
    #         #receiver_audio = np.pad(receiver_audio, (0, max(0, waveform_length - receiver_audio.shape[0])))
    #         source_audio = np.pad(source_audio, (0, waveform_length - source_audio.shape[0]))
    #     if waveform_length > receiver_audio.shape[0]:
    #         receiver_audio = np.pad(receiver_audio, (0, waveform_length - receiver_audio.shape[0]))

    #     if self.split != 'train':
    #         start_index = 0
    #     else:
    #         min_length = min(source_audio.shape[0], receiver_audio.shape[0])
    #         start_index = np.random.randint(0, min_length - waveform_length) \
    #                         if min_length > waveform_length else 0

    #     source_clip = source_audio[start_index: start_index + waveform_length]
    #     receiver_clip = receiver_audio[start_index: start_index + waveform_length]

    #     # normalize the short clip
    #     if self.normalize_segment:
    #         source_clip = normalize(source_clip, normalize_whole=False, norm='peak')
    #         receiver_clip = normalize(receiver_clip, normalize_whole=False, norm='peak')

    #     return source_clip, receiver_clip
    
    def process_audio(self, source_audio, receiver_audio):
        # normalize the intensity before padding
        waveform_length = 41600

        if self.normalize_whole:
            receiver_audio = normalize(receiver_audio, normalize_whole=True, norm='peak')
            source_audio = normalize(source_audio, normalize_whole=True, norm='peak')

        # pad audio
        if waveform_length > source_audio.shape[0]:
            receiver_audio = np.pad(receiver_audio, (0, max(0, waveform_length - receiver_audio.shape[0])))
            source_audio = np.pad(source_audio, (0, waveform_length - source_audio.shape[0]))

        if self.split != 'train':
            start_index = 0
        else:
            start_index = np.random.randint(0, source_audio.shape[0] - waveform_length) \
                            if source_audio.shape[0] != waveform_length else 0
        source_clip = source_audio[start_index: start_index + waveform_length]
        receiver_clip = receiver_audio[start_index: start_index + waveform_length]

        # normalize the short clip
        if self.normalize_segment:
            source_clip = normalize(source_clip, normalize_whole=False, norm='peak')
            receiver_clip = normalize(receiver_clip, normalize_whole=False, norm='peak')

        return source_clip, receiver_clip



def normalize(audio, normalize_whole=False, norm='peak'):
    if norm == 'peak':
        if normalize_whole == True:
            audio = audio - audio.mean()
            peak = abs(audio).max()
            audio = audio / (peak + 1e-8)
            return audio * 0.5
        else:
            peak = abs(audio).max()
            audio = audio / (peak + 1e-8)
            return audio * 0.5
        peak = abs(audio).max()
        if peak != 0:
            return audio / peak
        else:
            return audio
        
    elif norm == 'rms':
        if torch.is_tensor(audio):
            audio = audio.numpy()
        audio_without_padding = np.trim_zeros(audio, trim='b')
        rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
        if rms != 0:
            return audio / rms
        else:
            return audio
    else:
        raise NotImplementedError


def process_rir(rir):
    # normalize the intensity before padding
    rir = normalize(np.trim_zeros(rir, 'fb'), norm='peak')
    max_length = 17536

    # pad audio
    if max_length > rir.shape[0]:
        rir = np.pad(rir, (0, max(0, max_length - rir.shape[0])))

    rir = to_tensor(rir)
    return rir
