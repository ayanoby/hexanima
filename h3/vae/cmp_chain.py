"""python cmp_chain.py X.f32 K : compare the phone's rows after K blocks with the fp16 torch reference"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys, numpy as np, torch; sys.path.insert(0, (ROOT+'/h3' + '/vae'))
from mkvae_case import *
from safetensors import safe_open
st = safe_open(f'{P}/vae/minimax_h3_video_vae_fp16.safetensors', 'pt'); K = int(sys.argv[2]); T, LH, LW = 2, 16, 16
ids = torch.cat([ids3(T, LH, LW), torch.zeros(5, 3)]); cs = rope_cs(ids)
with torch.no_grad():
    x = embed(st, f'{P}/out/lat_fox50.f32', T, LH, LW)
    for i in range(K): x = block(x, st, i, cs)
o = torch.from_numpy(np.fromfile(sys.argv[1], np.float32).reshape(-1, D)); print('K', K, 'rel err %.4f' % float((o - x).pow(2).mean().sqrt() / x.pow(2).mean().sqrt()), 'ref rms %.3f' % float(x.pow(2).mean().sqrt()))
