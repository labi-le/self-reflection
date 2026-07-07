import hqq.core.quantize as q
import shutil

p = q.__file__
src = open(p).read()


def show(name: str) -> str:
    i = src.find(name)
    return src[i : i + 320] if i >= 0 else "NOT FOUND"


print("=== BEFORE: matmul ===")
print(show("def matmul"))
print("=== BEFORE: forward_pytorch ===")
print(show("def forward_pytorch"))

report = []
new = src

pat1_old = "weight = self.dequantize()\n        return torch.matmul(x, weight.t()"
pat1_new = "weight = self.dequantize()\n        x = x.to(weight.dtype)\n        return torch.matmul(x, weight.t()"
if pat1_old in new:
    new = new.replace(pat1_old, pat1_new)
    report.append("matmul PATCHED")
elif "x.to(weight.dtype)" in new:
    report.append("matmul ALREADY-CAST")
else:
    report.append("matmul PATTERN-NOT-FOUND")

pat2_old = "w = self.dequantize()\n        out = torch.matmul(x, w.t())"
pat2_new = "w = self.dequantize()\n        out = torch.matmul(x.to(w.dtype), w.t())"
if pat2_old in new:
    new = new.replace(pat2_old, pat2_new)
    report.append("forward_pytorch PATCHED")
elif "x.to(w.dtype)" in new:
    report.append("forward_pytorch ALREADY-CAST")
else:
    report.append("forward_pytorch PATTERN-NOT-FOUND")

if new != src:
    shutil.copy2(p, p + ".bak")
    open(p, "w").write(new)

print("REPORT:", report)
print("FILE:", p)
