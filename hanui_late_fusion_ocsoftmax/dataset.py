import os
import pdb
import torch
import torchaudio
import numpy as np
from glob import glob
from pathlib import Path
# from data_utils import *
from typing import List, Optional, Tuple, Union

EPS = 1e-12
class MFDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        input_dir, 
        chunk_size=64600, 
        train=True,
        valid=False,
        eval_t01=False,
        eval_t02=False,
        eval_t03=False,
        eval_t04=False,
        ctr_test=False,
        ctr_dev=False,
        # random_start=False, 
        sample_rate=16000,
        data_type='myst'):
        self.path = input_dir
        self.train = train
        self.valid = valid
        self.eval_t01 = eval_t01
        self.eval_t02 = eval_t02
        self.eval_t03 = eval_t03
        self.eval_t04 = eval_t04
        self.ctr_test = ctr_test
        self.ctr_dev = ctr_dev
        self.chunk_size = chunk_size
        # self.random_start = random_start
        self.track_list = self._get_tracklist()
        self.sample_rate = sample_rate
        self.data_type = data_type
        
    def _get_tracklist(self):
        path = str(self.path)
        if self.train:
            names = glob(os.path.join(path, '0_Training_*')) + glob(os.path.join(path, '1_Training_*'))
        elif self.valid:
            #print(os.path.join(path, 'dev/0_Validation_*'))
            names = glob(os.path.join(path, 'dev/0_Validation_*')) + glob(os.path.join(path, 'dev/1_Validation_*'))
        elif self.eval_t01:
            names = glob(os.path.join(path, 'test/0_T01_*')) + glob(os.path.join(path, 'test/1_T01_*'))
        elif self.eval_t02:
            names = glob(os.path.join(path, 'additional_test/0_T02_*')) + glob(os.path.join(path, 'additional_test/1_T02_*'))
        elif self.eval_t03:
            names = glob(os.path.join(path, 'mixtures_T03', 'aac_64k', '*_T03_*')) + glob(os.path.join(path, 'mixtures_T03', 'mp3_128k', '*_T03_*'))
        elif self.eval_t04:
            names = glob(os.path.join(path, 'mixtures_T04_only_per', '*_T04_*'))
        elif self.ctr_test:
            names = glob('/ssd2/Database/ctrsvdd/test_set/*.flac')
        elif self.ctr_dev:
            names = glob('/ssd2/Database/ctrsvdd/dev_set/*.flac')
        # elif self.eval_t01:
        #     # names = path + glob(os.path.join(path, '1_T01_*'))
        #     names = glob(os.path.join(path, '1_T01_*'))
        # elif self.eval_t02:
        #     path_tmp = path.replace('/test', '/additional_test')
        #     names = glob(os.path.join(path_tmp, '1_T02_*'))
        # else:
        #     path_tmp = path.replace('/test', '/dev')
        #     names = glob(os.path.join(path_tmp, '1_Validation_*'))
        
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
                nums_frames_to_read = start_frame + self.chunk_size
        return start_frame, num_frames_to_read
    
    def _read_audio(self, path: Path, frame_offset: Optional[int] = 0, num_frames: Optional[int] = -1) -> torch.Tensor:
        try:
            y, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
            return y
        except RuntimeError as e:
            print(f"Warning: failed to read {path}, skipping. Error: {e}")
            return None
        #assert sr == self.sample_rate, "audio samplerate of data does not match requested samplerate"
        #return y

    def _load_track(self, index):
        track_list = self.track_list

        while(1):
            track_name = track_list[index]
            frame_offset, num_frames = self._get_chunk_indices(track_name)
            track_path = Path(track_name)

            name = track_name.split('/')[-1]

            sig = self._read_audio(track_path, frame_offset, num_frames)
            if sig is not None:
                if len(sig) == 1:
                    sig = sig[0]
                else:
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
                
                if self.ctr_dev or self.ctr_test:
                    key = int(os.path.basename(track_name).split("_")[-1][:-5])
                else:
                    key = int(os.path.basename(track_name).split("_")[0])
                return sig, key, name
            else:        
                index = (index+1) % len(track_list)
                print(name)
            

    def __getitem__(self, index):
        return self._load_track(index)


    def __len__(self):
        return len(self.track_list)
    
    

