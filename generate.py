import sys
import os
import glob

import numpy as np
import librosa
import chainer

from models import VAE
from utils import mu_law
from utils import Preprocess
import opt

if opt.speaker_cond:
    speakers = glob.glob(os.path.join(opt.root, 'wav48/*'))
    n_speaker = len(speakers)
    speaker_dic = {
        os.path.basename(speaker): i for i, speaker in enumerate(speakers)}
else:
    n_speaker = None
    speaker_dic = None
model = VAE(opt.d, opt.k, opt.n_loop, opt.n_layer, opt.n_filter, opt.mu,
            opt.n_channel1, opt.n_channel2, opt.n_channel3, opt.beta, n_speaker)
chainer.serializers.load_npz(sys.argv[1], model)

n = 1
x = np.expand_dims(Preprocess(
    opt.data_format, opt.sr, opt.mu, opt.sr * 3, speaker_dic, False)(sys.argv[2])[0], 0)
if opt.speaker_cond:
    output = model.generate(x, np.array([sys.argv[3]], dtype=np.int32))
else:
    output = model.generate(x)

wave = mu_law(opt.mu).itransform(output)
np.save('result.npy', wave)
librosa.output.write_wav('result.wav', wave, opt.sr)
