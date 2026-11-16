# reference text encoding on CPU (bf16): prompt -> (L,20480) tokens saved to out/te.pt
import sys, math, torch
sys.path.insert(0, 'ref/src')
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
OUT = (4, 8, 12, 16, 20, 24, 28, 32); PAD = 80
P = 'klein/text_encoder'
prompt = open(sys.argv[1]).read().strip(); outp = sys.argv[2]
m = Qwen3VLForConditionalGeneration.from_pretrained(P, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
tok = AutoTokenizer.from_pretrained(P); tok.padding_side = 'right'
text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
real = tok(text, return_tensors='pt')['input_ids'].shape[1]
L = math.ceil(real / PAD) * PAD
inp = tok(text, return_tensors='pt', padding='max_length', truncation=True, max_length=L)
mask = inp['attention_mask']
pos = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
with torch.inference_mode():
    o = m(input_ids=inp['input_ids'], attention_mask=mask, position_ids=pos[None].expand(4, -1, -1), output_hidden_states=True, use_cache=False)
st = torch.stack([o.hidden_states[k] for k in OUT], dim=1)
tokens = st.permute(0, 2, 1, 3).reshape(L, -1).to(torch.bfloat16)
torch.save({'tokens': tokens, 'real': real, 'ids': inp['input_ids']}, outp)
print('real', real, 'L', L, tokens.shape, tokens.float().std().item())
