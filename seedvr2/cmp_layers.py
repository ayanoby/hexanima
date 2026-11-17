import numpy as np, sys
D = sys.argv[1]; T = int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
for l in range(n):
    a = np.fromfile(f'{D}/x_{l:02d}.f32', np.float32).reshape(T + 58, 2560); b = np.fromfile(f'{D}/dsp/x_{l:02d}.f32', np.float32).reshape(T + 58, 2560)
    e = lambda sl: np.sqrt(((a[sl] - b[sl]) ** 2).mean()) / np.sqrt((a[sl] ** 2).mean())
    print(f'layer {l:2d} vid {e(slice(0, T)):.4f} txt {e(slice(T, T + 58)):.4f}  vid max |x| {np.abs(a[:T]).max():.1f}')
