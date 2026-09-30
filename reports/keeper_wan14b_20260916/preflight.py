import json, socket, sys
from pathlib import Path
import torch
from pdd.data import PromptEmbeddings, WeightedPromptCache
from diffusers import WanTransformer3DModel
p='/mnt/data/butong/pipeline/test_prompts/wan22_sft_high_motion_proportional_32_extended_umt5.pt'
d=PromptEmbeddings(p)
assert len(d)==32, len(d)
for row in d:
 assert row['context'].shape == (512,4096), row['context'].shape
 assert torch.isfinite(row['context']).all()
assert torch.cuda.device_count()==8
x=torch.ones(4,device='cuda'); assert x.sum().item()==4
c=json.loads(Path('configs/wan21_14b_480p_450k_phased_dmd_pdd.json').read_text())
root=Path(c['checkpoint']); idx=json.loads((root/'diffusion_pytorch_model.safetensors.index.json').read_text())
assert all((root/f).is_file() for f in set(idx['weight_map'].values()))
w=WeightedPromptCache(c['weight_index']); item=next(iter(w)); assert item['context'].shape[-1]==4096
with socket.socket() as s:s.bind(('0.0.0.0',29616))
print(json.dumps({'preflight':'PASS','host':socket.gethostname(),'python':sys.executable,'gpus':8,'gpu':torch.cuda.get_device_name(0),'fixed_prompts':len(d),'shape':list(d[0]['context'].shape),'weighted_cache_entries':len(w.entries)}),flush=True)
