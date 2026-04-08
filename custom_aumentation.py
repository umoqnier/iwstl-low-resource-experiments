from audiomentations import Compose, AddGaussianNoise, PitchShift, Aliasing,ApplyImpulseResponse, BandPassFilter,AddBackgroundNoise,PolarityInversion

import numpy as np

T1 = AddBackgroundNoise(
    sounds_path="ESC-50/audio",
    min_snr_db=3.0,
    max_snr_db=30.0,
    noise_transform=PolarityInversion(),
    p=0.1
)

T2 = AddGaussianNoise(
    min_amplitude=0.001,
    max_amplitude=0.015,
    p=0.1
)

T3 = Aliasing(min_sample_rate=8000, max_sample_rate=30000, p=0.1)

T4 = ApplyImpulseResponse(ir_path=["room-impulse-responses/mit/MIT_Survey",
                                   "room-impulse-responses/rwcp/RIRS_NOISES/real_rirs_isotropic_noises"], p=1)

T5 = BandPassFilter(min_center_freq=100.0, max_center_freq=6000, p=0.1)

T6 = PitchShift(
    min_semitones=-5.0,
    max_semitones=5.0,
    p=0.1
)

augment = Compose([T1,T2,T3,T4,T5,T6])
