import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EMGDataset(Dataset):

    def __init__(
        self,
        data_dir,
        tokenizer,
        window=2000,
        stride=None,
        context_left=0,
        context_right=0,
        files=None,
        norm_mean=None,
        norm_std=None,
        augment=False,
        noise_std=0.0,
        channel_dropout_prob=0.0,
    ):
        self.tokenizer = tokenizer
        self.window = window
        self.stride = window if stride is None else int(stride)
        self.context_left = int(context_left)
        self.context_right = int(context_right)
        self.augment = augment
        self.noise_std = float(noise_std)
        self.channel_dropout_prob = float(channel_dropout_prob)

        if self.stride <= 0:
            raise ValueError('stride must be > 0')
        if self.context_left < 0 or self.context_right < 0:
            raise ValueError('context_left/context_right must be >= 0')

        self.norm_mean = None
        self.norm_std = None
        if norm_mean is not None and norm_std is not None:
            self.norm_mean = torch.tensor(norm_mean, dtype=torch.float32)
            self.norm_std = torch.tensor(norm_std, dtype=torch.float32)

        if files is None:
            self.files = []
            for file in sorted(os.listdir(data_dir)):
                if file.endswith('.hdf5'):
                    self.files.append(os.path.join(data_dir, file))
        else:
            self.files = [str(file) for file in files]

        self.samples = []

        # Build window indices and labels from emg2qwerty format.
        for file in self.files:
            with h5py.File(file, 'r') as f:
                group = f['emg2qwerty']
                ts = group['timeseries']
                t = ts.shape[0]

                keystrokes = json.loads(group.attrs['keystrokes'])
                ks_starts = np.asarray([k['start'] for k in keystrokes], dtype=np.float64)
                timestamps = ts['time']

                starts = list(range(0, t - window + 1, self.stride))
                if starts and starts[-1] != (t - window):
                    starts.append(t - window)

                for start in starts:
                    end = start + window
                    start_t = timestamps[start]
                    end_t = timestamps[end - 1]

                    i0 = int(np.searchsorted(ks_starts, start_t, side='left'))
                    i1 = int(np.searchsorted(ks_starts, end_t, side='right'))
                    text = self._keystrokes_to_text(keystrokes[i0:i1])

                    tokens = self.tokenizer.encode(text)
                    if len(tokens) == 0:
                        continue

                    self.samples.append((file, start, tokens, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file, start, tokens, total_len = self.samples[idx]

        with h5py.File(file, 'r') as f:
            ext_start = max(0, start - self.context_left)
            ext_end = min(total_len, start + self.window + self.context_right)
            window = f['emg2qwerty']['timeseries'][ext_start:ext_end]

            left = window['emg_left']
            right = window['emg_right']
            emg = np.concatenate([left, right], axis=1)

        x = torch.tensor(emg, dtype=torch.float32)
        if self.norm_mean is not None and self.norm_std is not None:
            x = (x - self.norm_mean) / self.norm_std

        if self.augment:
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std

            if self.channel_dropout_prob > 0:
                channel_mask = (
                    torch.rand(x.shape[1], device=x.device) < self.channel_dropout_prob
                )
                x[:, channel_mask] = 0.0

        y = torch.tensor(tokens, dtype=torch.long)

        return x, y

    @staticmethod
    def compute_channel_stats(files, chunk_size=200000):
        channel_sum = np.zeros(32, dtype=np.float64)
        channel_sq_sum = np.zeros(32, dtype=np.float64)
        count = 0

        for file in files:
            with h5py.File(file, 'r') as f:
                ts = f['emg2qwerty']['timeseries']
                total = ts.shape[0]

                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    chunk = ts[start:end]
                    left = chunk['emg_left'].astype(np.float64, copy=False)
                    right = chunk['emg_right'].astype(np.float64, copy=False)
                    emg = np.concatenate([left, right], axis=1)

                    channel_sum += emg.sum(axis=0)
                    channel_sq_sum += np.square(emg).sum(axis=0)
                    count += emg.shape[0]

        if count == 0:
            raise ValueError('Cannot compute stats: no EMG samples found.')

        mean = channel_sum / count
        var = channel_sq_sum / count - np.square(mean)
        var = np.maximum(var, 1e-8)
        std = np.sqrt(var)

        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def _keystrokes_to_text(keystrokes):
        chars = []

        for key in keystrokes:
            raw_code = key.get('ascii', -1)
            if raw_code is None:
                continue

            try:
                code = int(raw_code)
            except (TypeError, ValueError):
                continue

            if code == 9003:  # backspace
                if chars:
                    chars.pop()
                continue

            if code == 9166:  # enter
                chars.append(' ')
                continue

            if 32 <= code <= 126:
                chars.append(chr(code))

        return ''.join(chars)
