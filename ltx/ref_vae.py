"""Torch decode of LTX video latents with the ltx-core VideoDecoder (the 'conv' VAE file): python ref_vae.py LATENT_TOKENS.f32 FL HL WL OUT_PREFIX  -> OUT_PREFIX_NN.png frames (+ .npy)"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, json, numpy as np, torch
sys.path.insert(0, (ROOT+'/ltx'))
import ltxstub
import types, os
for name in ('ltx_core.guidance',):
    m = types.ModuleType(name); m.__path__ = [os.path.join((os.environ.get('LTX_C', 'ltxc')), *name.split('.'))]; sys.modules[name] = m
from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator
from safetensors import safe_open
P = (os.environ.get('LTX_W', 'ltx_weights') + '/vae/ltx-2.5-video-vae-conv-bf16.safetensors')
st = safe_open(P, 'pt'); cfg = json.loads(st.metadata()['config'])
dec = VideoDecoderConfigurator.from_config(cfg)
sd = {}
for k in st.keys():
    if k.startswith('decoder.'): sd[k[len('decoder.'):]] = st.get_tensor(k).float()
    elif k.startswith('per_channel_statistics.'): sd[k] = st.get_tensor(k).float()
print(dec.load_state_dict(sd, strict=False)); dec = dec.float().eval()
if __name__ == '__main__':
    z = np.fromfile(sys.argv[1], dtype=np.float32); Fl, Hl, Wl = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); z = torch.from_numpy(z).reshape(1, Fl, Hl, Wl, 128).permute(0, 4, 1, 2, 3).contiguous()
    with torch.no_grad(): vid = dec(z)
    print('decoded', tuple(vid.shape), 'range', float(vid.min()), float(vid.max()))
    v = ((vid[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).numpy(); np.save(sys.argv[5] + '.npy', v)
    from PIL import Image
    for i in range(v.shape[0]): Image.fromarray(v[i]).save(f'{sys.argv[5]}_{i:02d}.png')
