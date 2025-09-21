import os
import pdb
import torch
import torchaudio
import numpy as np
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

EPS = 1e-12
class MFDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        input_dir, 
        chunk_size=6400,    # 0.38 sec. encoder_hop = 256, enc-dec size mismatch: 6080->6400
        train=True,
        valid=False,
        eval_t01=False,
        eval_t02=False,
        eval_t04=False,
        eval_t03=False,
        spoof=False,
        disc=False,
        sample_rate=16000,
        data_type='myst'):
        self.path = input_dir
        self.train = train
        self.valid = valid
        self.eval_t01 = eval_t01
        self.eval_t02 = eval_t02
        self.eval_t04 = eval_t04
        self.eval_t03 = eval_t03
        self.spoof = spoof
        self.disc = disc
        self.chunk_size = chunk_size
        self.track_list = self._get_tracklist()
        self.sample_rate = sample_rate
        self.data_type = data_type
        self.hop_size = 256
        
    def _get_tracklist(self):
        path = str(self.path)
        # names = glob(os.path.join(path,'*.flac'))
        if self.train:
            names = glob(os.path.join(path, '*_Training_*'))    # recon_0_Training_1006.wav, # 0: bonafide, 1: spoof
        elif self.valid:
            names = glob(os.path.join(path, '*_Validation_*'))
        elif self.eval_t01:
            names = glob(os.path.join(path, '*_T01_*'))
        elif self.eval_t02:
            names = glob(os.path.join(path, '*_T02_*'))
        elif self.eval_t04:
            names = glob(os.path.join(path, '../mixtures_T04_only_per', '*_T04_*'))
        elif self.eval_t03:
            names = glob(os.path.join(path, '*_T03_*'))
            
        return sorted(names)

    def _get_chunk_indices(self, track_name):
        mix_path = Path(track_name)
        num_frames_total = torchaudio.info(mix_path).num_frames
        
        start_frame = 0
        num_frames_to_read = self.chunk_size
            
        if num_frames_total <= self.chunk_size:
            num_frames_to_read = -1
        else:
            if self.train:
                start_frame = int(torch.randint(0, num_frames_total - self.chunk_size, (1,)))
        return start_frame, num_frames_to_read
    
    def _read_audio(self, path: Path, frame_offset: Optional[int] = 0, num_frames: Optional[int] = -1) -> torch.Tensor:
        y, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        assert sr == self.sample_rate, "audio samplerate of data does not match requested samplerate"
        return y

    def _load_track(self, index):
        track_list = self.track_list

        track_name = track_list[index]
        frame_offset, num_frames = self._get_chunk_indices(track_name)
        track_path = Path(track_name)
        
        sig = self._read_audio(track_path, frame_offset, num_frames)
        if len(sig) > 1:
            if index % 2 == 0:
                sig = sig[0]
            else:
                sig = sig[1]

        sig_len = sig.size(-1)
        if self.train and (sig_len < self.chunk_size):
            num_repeats = int(self.chunk_size / sig_len) + 1
            sig = torch.tile(sig, [num_repeats])[:self.chunk_size]

        elif sig_len < self.chunk_size:
            sig = torch.nn.functional.pad(sig, (0, self.chunk_size-sig_len))
        elif sig_len % self.hop_size != 0:
            padding_size = ((sig_len // self.hop_size)+1) * self.hop_size - sig_len
            sig = torch.nn.functional.pad(sig, (0, padding_size))

        if 'recon' in track_name:
            key = int(os.path.basename(track_name).split("_")[1])   # recon_0_Training_1006.wav
        else:
            key = int(os.path.basename(track_name).split("_")[0])   # 0_Training_1006.wav
        
        return sig, key, track_name.split('/')[-1]

    def __getitem__(self, index):
        return self._load_track(index)


    def __len__(self):
        return len(self.track_list)
    
    
