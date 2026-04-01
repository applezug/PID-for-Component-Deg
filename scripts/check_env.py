#!/usr/bin/env python
"""
MBD_PINN 环境依赖检查
用法：conda activate MBD_PINN 后运行
      python scripts/check_env.py
结果保存到 results/env_check_YYYYMMDD_HHMMSS.txt
"""
import sys
import os
from datetime import datetime

# 项目根目录（脚本在 scripts/ 下）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
out_path = os.path.join(ROOT, "results", f"env_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

lines = []
def log(s=""):
    print(s)
    lines.append(s)

def check(name, fn):
    """执行检查，返回 (ok, msg)"""
    try:
        result = fn()
        return True, str(result) if result is not None else "OK"
    except Exception as e:
        return False, str(e)

log("=" * 60)
log("MBD_PINN 环境依赖检查")
log(f"时间: {datetime.now().isoformat()}")
log(f"Python: {sys.executable}")
log("=" * 60)
log()

# 1. Python 版本
v = sys.version_info
ok, msg = check("Python", lambda: f"{v.major}.{v.minor}.{v.micro}")
log(f"1. Python 版本: {msg}")
if v.major < 3 or (v.major == 3 and v.minor < 9):
    log("   [警告] 推荐 Python 3.9+")
log()

# 2-10. 依赖包
checks = [
    ("torch", lambda: f"{__import__('torch').__version__}"),
    ("numpy", lambda: __import__('numpy').__version__),
    ("pandas", lambda: __import__('pandas').__version__),
    ("scipy", lambda: __import__('scipy').__version__),
    ("matplotlib", lambda: __import__('matplotlib').__version__),
    ("tqdm", lambda: __import__('tqdm').__version__),
    ("yaml (pyyaml)", lambda: getattr(__import__('yaml'), '__version__', 'OK')),
    ("sklearn", lambda: __import__('sklearn').__version__),
]
for i, (name, fn) in enumerate(checks, start=2):
    ok, msg = check(name, fn)
    status = "[OK]" if ok else "[失败]"
    log(f"{i}. {name}: {status} {msg}")
log()

# PyTorch 详情
log("11. PyTorch 详情:")
try:
    import torch
    log(f"    版本: {torch.__version__}")
    log(f"    CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"    CUDA 版本: {torch.version.cuda}")
        log(f"    GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        log("    [提示] 未检测到 GPU，将使用 CPU（速度较慢）")
except Exception as e:
    log(f"    [失败] {e}")
log()

# 12. 项目模块导入测试
log("12. 项目模块导入:")
proj_imports = [
    ("models.diffusion", "MBDDegradationImputation, LinearDegradationModel"),
    ("models.physics", "CompressorPhysicsNorm"),
    ("datasets.cmapss_dataset", "CMAPSSDataset"),
    ("utils.rul_estimator", "load_rul_labels, estimate_rul_from_trajectory"),
]
# 需在项目根目录执行
try:
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    for mod, _ in proj_imports:
        ok, msg = check(mod, lambda m=mod: __import__(m))
        status = "[OK]" if ok else "[失败]"
        log(f"    {mod}: {status} {msg}")
except Exception as e:
    log(f"    [失败] {e}")
log()

log("=" * 60)
log("检查完成，请将本文件内容发给协助方")
log("=" * 60)

# 写入文件
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n结果已保存至: {out_path}")
