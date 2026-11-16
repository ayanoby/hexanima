"""ONNX of the tile decoder: python export_tile.py NAME T H W (latent frames, latent h, w) -> NAME.onnx (input z [T,128,H,W] normalised latent, output [8(T-1)+1, 3, 32H, 32W])"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys, torch
sys.path.insert(0, (ROOT+'/ltx' + '/vae'))
import tiledec as D
name, T, H, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
import gc
td = D.TileDecoder(D.V.dec).eval(); del D.V.dec; gc.collect(); z = torch.randn(T, 128, H, W)   # free the ltx-core decoder (1.6 GB) before tracing
with torch.no_grad(): y = td(z); print('out', tuple(y.shape))
torch.onnx.export(td, (z,), f'{name}.onnx', input_names=['z'], output_names=['y'], opset_version=17, dynamo=False)
torch.save({'z': z, 'y': y}, f'{name}.ref.pt'); print('exported', name)
