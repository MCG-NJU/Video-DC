import torch

H = 16
D = 64 * H
L = 1000

torch.set_float32_matmul_precision("highest")
Wq = torch.randn(D, D).cuda()
Wv = torch.randn(D, D).cuda()
Wo = torch.randn(D, D).cuda()
data = torch.randn(L, D).cuda()
# attention
q = torch.mm(data, Wq).view(L, H, D // H).permute(1, 0, 2)
k = torch.mm(data, Wq).view(L, H, D // H).permute(1, 0, 2)
attn = torch.bmm(q, k.transpose(1, 2))
attn = torch.softmax(attn, dim=-1)

v = torch.mm(data, Wv).view(L, H, D // H).permute(1, 0, 2)
attn_v = torch.bmm(attn, v).permute(1, 0, 2).reshape(L, D)
out = torch.mm(attn_v, Wo)

# fused
Wvo = torch.mm(Wv, Wo)
data2 = data.view(L, H, D // H).permute(1, 0, 2)
attn_x = torch.bmm(attn, data2).permute(1, 0, 2).reshape(L, D)
fused_out = torch.mm(attn_x, Wvo)
# fused_out = torch.mm(attn_x, Wv)
# fused_out = torch.mm(fused_out, Wo)

# cos sim between fusedout and out
nfo = torch.nn.functional.normalize(fused_out.view(1, -1))
no = torch.nn.functional.normalize(out.view(1, -1))
cos = (nfo * no).sum()
print(cos)

print(((fused_out - out).abs().mean()))

