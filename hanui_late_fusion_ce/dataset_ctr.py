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
        sample_rate=16000,
        data_type='myst'):
        self.path = input_dir
        self.chunk_size = chunk_size
        self.track_list = self._get_tracklist()
        self.sample_rate = sample_rate
        self.data_type = data_type
        self.hop_size = 256
        
    def _get_tracklist(self):
        path = str(self.path)
        names = glob(os.path.join(path,'*.flac'))
        
        return sorted(names)

    def _get_chunk_indices(self, track_name):
        mix_path = Path(track_name)
        num_frames_total = torchaudio.info(mix_path).num_frames
        
        start_frame = 0
        num_frames_to_read = self.chunk_size
            
        if num_frames_total <= self.chunk_size:
            num_frames_to_read = -1
        return start_frame, num_frames_to_read
    
    def _read_audio(self, path: Path, frame_offset: Optional[int] = 0, num_frames: Optional[int] = -1) -> torch.Tensor:
        y, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        assert sr == self.sample_rate, "audio samplerate of data does not match requested samplerate"
        if sr != self.sample_rate:
            print("audio samplerate of data does not match requested samplerate")
            breakpoint()
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
        if sig_len < self.chunk_size:
            sig = torch.nn.functional.pad(sig, (0, self.chunk_size-sig_len))
        elif sig_len % self.hop_size != 0:
            padding_size = ((sig_len // self.hop_size)+1) * self.hop_size - sig_len
            sig = torch.nn.functional.pad(sig, (0, padding_size))

        key = int(os.path.basename(track_name).split('.')[0].split("_")[-1])   # **_1.flac
        
        return sig, key, track_name.split('/')[-1]

    def __getitem__(self, index):
        return self._load_track(index)


    def __len__(self):
        return len(self.track_list)
    
    
