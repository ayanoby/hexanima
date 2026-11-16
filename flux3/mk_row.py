import json, textwrap
from PIL import Image, ImageDraw, ImageFont
ids = ['1794677923714', '1794678129547', '1794678597291']
W = 800; PAD = 30; BG = (18, 18, 22); FG = (235, 235, 240); DIM = (150, 155, 165)
fb = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
ft = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
cols = []
for i in ids:
    im = Image.open(f'out/h3test/{i}.png').convert('RGB').resize((W, W), Image.LANCZOS)
    j = json.load(open(f'out/h3test/{i}.json'))
    head = f"{j['w']}x{j['h']} | 6 steps | {j['sec']:.0f} s"
    lines = textwrap.wrap(j['prompt'], 78)
    cols.append((im, head, lines))
tmax = max(len(c[2]) for c in cols)
H = PAD + W + 20 + 40 + tmax * 27 + PAD
out = Image.new('RGB', (3 * W + 4 * PAD, H), BG); d = ImageDraw.Draw(out)
for k, (im, head, lines) in enumerate(cols):
    x = PAD + k * (W + PAD); out.paste(im, (x, PAD))
    y = PAD + W + 18; d.text((x, y), head, font=fb, fill=FG); y += 44
    for ln in lines: d.text((x, y), ln, font=ft, fill=DIM if False else FG); y += 27
out.save('out/h3_row.png'); out.convert('RGB').save('out/h3_row.jpg', quality=90)
print(out.size)
