"""
RUL 预测与评估实验

流程：1) 加载测试集与 RUL 真值
      2) 加载或拟合 MBD 模型
      3) 对每个测试发动机生成未来轨迹
      4) 从轨迹估计 RUL
      5) 计算 RMSE、PHM Score 并保存
"""

import os
import sys
import json
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from models.diffusion import (
    MBDDegradationImputation,
    LinearDegradationModel,
    ExponentialDegradationModel,
    PowerLawDegradationModel,
)
from datasets.cmapss_dataset import CMAPSSDataset
from utils.rul_estimator import (
    load_rul_labels,
    load_test_sequences_last_window,
    estimate_rul_from_trajectory,
    evaluate_rul,
)
from utils.metrics import picp, mean_interval_width
from utils.io_utils import load_yaml_config
from experiments.run_hpc import fit_degradation_from_dataset, fit_coeff_norm_for_strategy2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/hpc_cmapss_paper.yaml")
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--n_test", type=int, default=5, help="评估的测试发动机数量（0=全部）")
    parser.add_argument("--n_samples", type=int, default=1, help="每发动机采样轨迹数；>1 时得到 RUL 预测区间并计算 PICP")
    parser.add_argument("--interval_quantile", type=float, default=0.9, help="预测区间置信水平，如 0.9 表示 90%% 区间")
    parser.add_argument("--degradation_model", type=str, default=None, choices=["linear", "exponential", "power_law"],
                        help="MBD 退化模型类型；默认从 config diffusion.degradation_model 读取")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    cfg = load_yaml_config(args.config)
    data_cfg = cfg.get("data", {})
    diff_cfg = cfg.get("diffusion", {})
    rul_cfg = cfg.get("rul", {})
    out_cfg = cfg.get("output", {})
    train_cfg = cfg.get("training", {})

    data_root = data_cfg.get("data_root", "data/cmapss")
    dataset = args.dataset
    seq_length = data_cfg.get("seq_length", 256)
    sensor_indices = data_cfg.get("sensor_indices", [2, 3, 7, 14])
    feature_size = len(sensor_indices)
    failure_threshold = rul_cfg.get("failure_threshold", 0.8)
    deg_feature_idx = rul_cfg.get("deg_feature_idx", 1)  # 退化特征列索引，HPC=1(T30)，风扇=0(T2)，涡轮=0(T50)，燃烧室=3(Wf/Ps30)
    max_rul = rul_cfg.get("max_rul", 125.0)  # 轨迹步数→RUL(cycles)的映射上界，与 C-MAPSS/PHM08 截断一致
    save_dir = out_cfg.get("save_dir", "results")
    exp_name = out_cfg.get("exp_name", "hpc_cmapss")
    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = train_cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. 加载测试集与 RUL 真值
    try:
        test_seqs, last_cycles, engine_ids = load_test_sequences_last_window(
            data_root, dataset, seq_length, sensor_indices
        )
        true_rul = load_rul_labels(data_root, dataset)
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        return

    n_test = args.n_test if args.n_test > 0 else len(test_seqs)
    n_test = min(n_test, len(test_seqs))

    # 2. 准备训练集（用于拟合退化模型）与归一化
    train_ds = CMAPSSDataset(
        data_root=data_root,
        dataset=dataset,
        seq_length=seq_length,
        period="train",
        sensor_indices=sensor_indices,
        seed=seed,
        normalize=True,
    )
    min_v, max_v = train_ds.global_norm_stats

    # 归一化测试序列
    test_norm = 2.0 * (test_seqs - min_v) / (max_v - min_v + 1e-8) - 1.0
    test_norm = test_norm.astype(np.float32)

    # 3. 加载或拟合 MBD（退化模型类型：linear / exponential / power_law）
    deg_type = (args.degradation_model or diff_cfg.get("degradation_model", "linear")).lower()
    if deg_type == "exponential":
        deg_model = ExponentialDegradationModel(device=device)
    elif deg_type == "power_law":
        deg_model = PowerLawDegradationModel(device=device)
    else:
        deg_model = LinearDegradationModel(device=device)
    ckpt_path = os.path.join(save_dir, f"{exp_name}_degradation_{dataset}_{deg_type}.pt")
    fallback_path = os.path.join(save_dir, f"{exp_name}_degradation_{dataset}.pt")
    load_path = ckpt_path if os.path.exists(ckpt_path) else (fallback_path if os.path.exists(fallback_path) else None)
    if load_path:
        ckpt = torch.load(load_path, map_location=device, weights_only=False)
        deg_model.params = ckpt.get("degradation_params", deg_model.params)
        print("Loaded degradation params from", load_path)
    else:
        fit_degradation_from_dataset(deg_model, train_ds, n_samples=min(50, len(train_ds)), device=device, dataset_name=dataset)

    diffusion_model = MBDDegradationImputation(
        seq_length=seq_length,
        feature_size=feature_size,
        degradation_model=deg_model,
        timesteps=diff_cfg.get("timesteps", 200),
        sampling_timesteps=diff_cfg.get("sampling_timesteps", 50),
        Nsample=diff_cfg.get("Nsample", 512),
        temp_sample=diff_cfg.get("temp_sample", 0.1),
        beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
        device=device,
    )
    diffusion_model.eval()

    # 4. 对每个测试发动机生成未来轨迹并估计 RUL（可选：多采样得预测区间）
    gen_length = int(seq_length * 0.5)
    n_samples = max(1, args.n_samples)
    use_interval = n_samples > 1
    pred_rul_list = []
    pred_lower_list = []
    pred_upper_list = []

    # 轨迹 RUL 为“步数”(0~gen_length)，需转换为与 true_rul 一致的 cycles；C-MAPSS 常用 max_rul=125
    def steps_to_cycles(steps: np.ndarray) -> np.ndarray:
        return (steps.astype(np.float64) / float(gen_length)) * max_rul

    for i in tqdm(range(n_test), desc="RUL prediction"):
        obs = torch.from_numpy(test_norm[i]).float().to(device)
        t = torch.linspace(0, 1, seq_length, device=device)
        mask = torch.ones(seq_length, feature_size, dtype=torch.bool, device=device)
        mask[seq_length - gen_length :, :] = False

        rul_samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                imputed = diffusion_model.sample(obs, t, mask, clip_denoised=True)
            future = imputed[seq_length - gen_length :].cpu().numpy()
            r_steps = estimate_rul_from_trajectory(
                future,
                deg_feature_idx=deg_feature_idx,
                failure_threshold=failure_threshold,
                normalize=True,
            )
            rul_samples.append(r_steps)

        rul_samples = np.array(rul_samples)
        rul_cycles = steps_to_cycles(rul_samples)
        pred_rul_list.append(float(np.median(rul_cycles)))
        if use_interval:
            alpha = 1.0 - args.interval_quantile
            pred_lower_list.append(float(np.percentile(rul_cycles, alpha / 2 * 100)))
            pred_upper_list.append(float(np.percentile(rul_cycles, (1 - alpha / 2) * 100)))

    pred_rul = np.array(pred_rul_list)
    true_rul_sub = true_rul[:n_test]

    # 5. 评估
    rmse = float(np.sqrt(np.mean((pred_rul - true_rul_sub) ** 2)))
    phm = float(np.sum(np.where(
        pred_rul - true_rul_sub < 0,
        10 * (np.exp(-(pred_rul - true_rul_sub) / 10) - 1),
        10 * (np.exp((pred_rul - true_rul_sub) / 13) - 1),
    )))

    metrics = {"rul_rmse": rmse, "rul_phm_score": phm, "n_test": n_test}
    if use_interval:
        pred_lower = np.array(pred_lower_list)
        pred_upper = np.array(pred_upper_list)
        metrics["rul_picp"] = float(picp(pred_lower, pred_upper, true_rul_sub))
        metrics["rul_interval_quantile"] = args.interval_quantile
        metrics["rul_mean_interval_width"] = float(mean_interval_width(pred_lower, pred_upper))
        np.save(os.path.join(save_dir, f"{exp_name}_rul_predictions_{dataset}_{deg_type}_lower.npy"), pred_lower)
        np.save(os.path.join(save_dir, f"{exp_name}_rul_predictions_{dataset}_{deg_type}_upper.npy"), pred_upper)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{exp_name}_rul_predictions_{dataset}_{deg_type}.npy"), pred_rul)
    with open(os.path.join(save_dir, f"{exp_name}_rul_metrics_{dataset}_{deg_type}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"RUL RMSE: {rmse:.4f}, PHM Score: {phm:.4f}")
    if use_interval:
        print(f"PICP ({args.interval_quantile*100:.0f}% interval): {metrics['rul_picp']:.4f}, mean width: {metrics['rul_mean_interval_width']:.2f}")
    print(f"Results saved to {save_dir}/")


if __name__ == "__main__":
    main()
