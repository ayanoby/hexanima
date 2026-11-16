"""ONNX of the one-image tile encoder: python export_enc.py NAME H W (pixels, multiples of 32) -> NAME.onnx (input img [1,3,H,W] in [-1,1], output normalised latent [1,128,H/32,W/32])"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys, gc, torch
sys.path.insert(0, (ROOT+'/ltx' + '/vae'))
import enc2d as E
name, H, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ref = E.load_ref(); m = E.Enc(ref).eval(); E.rescale(m); del ref; gc.collect(); x = torch.rand(1, 3, H, W) * 2 - 1
with torch.no_grad(): y = m(x); print('out', tuple(y.shape))
torch.onnx.export(m, (x,), f'{name}.onnx', input_names=['img'], output_names=['z'], opset_version=17, dynamo=False)
torch.save({'x': x, 'y': y}, f'{name}.ref.pt'); print('exported', name)
