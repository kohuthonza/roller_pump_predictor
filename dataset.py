import os
import numpy as np

from torch.utils.data import Dataset


class WaveDataset(Dataset):
    def __init__(self, wave_directory_path, max_pressure=60000, max_speed=1.6):
        self.wave_directory_path = wave_directory_path
        self.max_pressure = max_pressure
        self.max_speed = max_speed
        self.waves = []
        self.waves_paths = []
        for wave_path in os.listdir(self.wave_directory_path):
            input_pressure_wave, target_speed_wave = self.read_wave(os.path.join(self.wave_directory_path, wave_path))
            self.waves.append((input_pressure_wave, target_speed_wave))
            self.waves_paths.append(wave_path)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        return {'input_pressure_wave': self.waves[idx][0],
                'target_speed_wave': self.waves[idx][1],
                'path': self.waves_paths[idx]}

    def read_wave(self, wave_path):
        with open(wave_path) as f:
            lines = f.readlines()
        input_pressure_wave = []
        target_speed_wave = []
        for l in lines:
            _, pressure, speed = l.split(",")
            input_pressure_wave.append(float(pressure))
            target_speed_wave.append(float(speed))
        input_pressure_wave = np.asarray(input_pressure_wave)
        target_speed_wave = np.asarray(target_speed_wave)
        input_pressure_wave = input_pressure_wave / (self.max_pressure / 2.0)
        target_speed_wave = target_speed_wave / (self.max_speed / 2.0)
        input_pressure_wave = input_pressure_wave - 1.0
        target_speed_wave = target_speed_wave - 1.0
        return input_pressure_wave, target_speed_wave
