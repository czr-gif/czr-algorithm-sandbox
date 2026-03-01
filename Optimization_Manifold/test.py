import torch
import numpy as np

print("✅ PyTorch version:", torch.__version__)
print("✅ Numpy version:", np.__version__)

# 检查苹果 M 系列芯片的 MPS 引擎是否激活
if torch.backends.mps.is_available():
    print("🚀 苹果 MPS (Metal) 硬件加速已就绪！")
    # 我们可以建一个张量直接塞进 Mac 的 GPU 里
    x = torch.ones(1, device=torch.device("mps"))
    print(f"测试张量已分配至: {x.device}")
else:
    print("⚠️ 未检测到 MPS。将使用纯 CPU 模式运行（如果你用的是老款 Intel Mac，这是正常的）。")