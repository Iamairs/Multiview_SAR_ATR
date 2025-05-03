#!/usr/bin/env python3
# asc_ga_afem.py
# ============================================================
#  ASC  attributed‑scattering‑center extraction from a
#  complex SAR / ISAR image  (image‑domain version)
#
#  – 基础框架采用 Code‑A（整幅染色体携带多 ASC 的做法）
#  – 引入 Code‑B 的 AFEM 思想：一次估 1 个，再松弛
#  – 依赖: numpy, scipy, matplotlib, tqdm
# ============================================================
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy.linalg as la
from scipy.signal import convolve2d
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, time, math, copy, functools

# -------------------------- 0.  常数区 ------------------------------
# 可根据传感器实参替换
C0 = 3.0e8                     # 光速
PIXEL_SIZE_M = 0.3            # 每像素代表多少米 (行、列相同假设)
MAX_AFEM_STEPS = 15            # 试图提取的最大 ASC 数

# -------- 图像域“散射类型词典”  S_t(x,y)  --------------------------
def make_freq_grid(H, W):
    fy = np.fft.fftfreq(H, d=PIXEL_SIZE_M)   # cycles / m
    fx = np.fft.fftfreq(W, d=PIXEL_SIZE_M)
    FX, FY = np.meshgrid(fx, fy)
    return FX, FY

# === 你可以在这里把 Code‑B 的 α,φ,L 解析模型换进来 ===
def kernel_isotropic(H, W):
    FX, FY = make_freq_grid(H, W)
    gamma = np.ones_like(FX)
    S = ifft2(ifftshift(gamma))
    return S / np.sqrt(np.sum(np.abs(S)**2))

def kernel_dihedral(H, W, phi=0):
    FX, FY = make_freq_grid(H, W)
    theta = np.arctan2(FY, FX)
    gamma = np.cos(theta - phi)
    S = ifft2(ifftshift(gamma))
    return S / np.sqrt(np.sum(np.abs(S)**2))

DICT_KERNELS = None    # 运行时构造
def get_kernel(type_id, H, W):
    global DICT_KERNELS
    if DICT_KERNELS is None:
        # 预先生成 3 种方向的双面体 + 1 个等向
        DICT_KERNELS = [
            kernel_isotropic(H, W),
            kernel_dihedral(H, W, 0),
            kernel_dihedral(H, W, np.pi/4),
            kernel_dihedral(H, W, np.pi/2)
        ]
    return DICT_KERNELS[type_id % len(DICT_KERNELS)]

# -------------------- 1. 单 ASC 前向函数 ----------------------------
def place_kernel(canvas, kern, xc, yc, amp):
    """把 kernel(复数) 平移到 (xc,yc) 并乘以 amp 后累加到 canvas"""
    H, W = canvas.shape
    kh, kw = kern.shape
    top = int(np.round(yc - kh/2))
    left = int(np.round(xc - kw/2))
    y0 = max(0, -top); y1 = min(kh, H-top)
    x0 = max(0, -left); x1 = min(kw, W-left)
    if y1<=y0 or x1<=x0:
        return
    canvas[top+y0:top+y1, left+x0:left+x1] += amp * kern[y0:y1, x0:x1]

def model_single(params, H, W):
    """params = [x, y, Re(a), Im(a), type_id]"""
    xc, yc, ar, ai, tp = params
    amp = ar + 1j*ai
    kern = get_kernel(int(round(tp)), H, W)
    canv = np.zeros((H, W), np.complex128)
    place_kernel(canv, kern, xc, yc, amp)
    return canv

# -------------------- 2.  适应度 --------------------
def fitness_single(params, target_img, lam_N=0.0):
    H, W = target_img.shape
    Ihat = model_single(params, H, W)
    resid = target_img - Ihat
    err2 = la.norm(resid.ravel())**2
    mdl = np.log(H*W)*1 + err2     # N=1 here, lam_N 可选
    return -mdl, err2

# -------------------- 3.  GA 类：针对“一个 ASC” ----------------------
class GAOneASC:
    def __init__(self, pop, pc, pm, bounds, target):
        self.pop = pop          # (P, D)
        self.pc, self.pm = pc, pm
        self.bounds = np.array(bounds)
        self.target = target
        self.H, self.W = target.shape
        self.fitness = np.zeros(len(pop))
        self.best = None
        self.best_fit = -np.inf

    # ------- 基本算子 -------
    def eval_fit(self):
        for i in range(len(self.pop)):
            f, _ = fitness_single(self.pop[i], self.target)
            self.fitness[i] = f
            if f > self.best_fit:
                self.best_fit = f; self.best = self.pop[i].copy()

    def tournament(self, k=3):
        ids = np.random.randint(0, len(self.pop), k)
        return self.pop[ids[np.argmax(self.fitness[ids])]].copy()

    def sbx(self, p1, p2, eta=2):
        c1, c2 = p1.copy(), p2.copy()
        for i in range(len(p1)):
            if np.random.rand()>self.pc: continue
            u = np.random.rand()
            beta = (2*u)**(1/(eta+1)) if u<=.5 else (1/(2*(1-u)))**(1/(eta+1))
            c1[i] = .5*((1+beta)*p1[i] + (1-beta)*p2[i])
            c2[i] = .5*((1-beta)*p1[i] + (1+beta)*p2[i])
        return c1, c2

    def mutate(self, ind):
        for j in range(len(ind)):
            if np.random.rand() < self.pm/len(ind):
                rng = self.bounds[j,1]-self.bounds[j,0]
                ind[j] += (np.random.rand()-0.5)*0.2*rng
        self.bound(ind)

    def bound(self, ind):
        for j in range(len(ind)):
            lo, hi = self.bounds[j]
            ind[j] = np.clip(ind[j], lo, hi)
            if j==4: ind[j]=round(ind[j])   # type id

    # ------- 主循环 -------
    def run(self, n_gen=150, elite=2):
        self.eval_fit()
        for g in range(n_gen):
            new = []
            elite_idx = self.fitness.argsort()[-elite:]
            new.extend(self.pop[elite_idx].copy())
            while len(new)<len(self.pop):
                p1 = self.tournament(); p2 = self.tournament()
                c1,c2 = self.sbx(p1,p2)
                self.mutate(c1); self.mutate(c2)
                new.extend([c1,c2])
            self.pop = np.array(new[:len(self.pop)])
            self.eval_fit()
        return self.best, self.best_fit

# -------------------- 4.  AFEM‑GA 多 ASC 提取 ------------------------
def afem_ga_extract(img,
                    pop_size=120, n_gen=250,
                    pc=.9, pm=.2,
                    max_k=MAX_AFEM_STEPS,
                    type_choices=4,  # 词典里核的种类数
                    accept_tol=1e-4):
    H,W = img.shape
    # ----------- 基础边界 ------------
    bounds = np.array([
        [0, W-1],          # x
        [0, H-1],          # y
        [-1, 1],            # Re(a)
        [-1, 1],            # Im(a)
        [0, type_choices-1]# type id
    ], dtype=float)

    residual = img.copy()
    asc_params = []       # 已接受的 ASC
    err2_prev = la.norm(residual.ravel())**2
    print(f"初始误差² = {err2_prev:.3e}")

    for step in range(max_k):
        print(f"\n====  提取第 {step+1} 个 ASC  ====")
        # --1. GA 初始种群
        pop0 = np.zeros((pop_size, 5))
        for j in range(5):
            lo,hi=bounds[j]
            if j==4:
                pop0[:,j]=np.random.randint(lo,hi+1,pop_size)
            else:
                pop0[:,j]=np.random.uniform(lo,hi,pop_size)

        ga = GAOneASC(pop0, pc, pm, bounds, residual)
        best, best_fit = ga.run(n_gen)
        contri = model_single(best, H, W)
        err2_new = la.norm((residual-contri).ravel())**2
        print(f"找到候选 ASC, 残差² 从 {err2_prev:.3e} → {err2_new:.3e}")

        # --2. 接受判决
        if err2_prev - err2_new < accept_tol*err2_prev:
            print("改善不足，终止提取。")
            break
        asc_params.append(best)
        residual -= contri
        err2_prev = err2_new

        # --3. Relaxation：重新优化已接受的所有 ASC
        print("  ↪  松弛优化已接受的 ASC ...")
        for idx in range(len(asc_params)):
            # 构造“其它 ASC 重建”的图像
            other = sum(model_single(p, H,W) for k,p in enumerate(asc_params) if k!=idx)
            target_for_idx = img - other
            # GA 微调，每次资源减半
            pop_half = np.tile(asc_params[idx], (pop_size//2,1))
            pop_half += 0.05*np.random.randn(*pop_half.shape)
            pop_half[:,4] = asc_params[idx][4]   # type 不乱动
            ga2 = GAOneASC(pop_half, pc, pm, bounds, target_for_idx)
            best2,_ = ga2.run(max(50, n_gen//4))
            # 若更好则替换
            before = model_single(asc_params[idx],H,W)
            after  = model_single(best2, H,W)
            resid_try = residual + before - after
            if la.norm(resid_try.ravel())**2 < err2_prev:
                print(f"    ASC {idx+1} 继续改进 √")
                asc_params[idx]=best2
                residual = resid_try
                err2_prev = la.norm(residual.ravel())**2

        print(f"松弛结束，当前残差² = {err2_prev:.3e}")
    return asc_params, residual

# --------------------- 5.  CLI Demo ------------------------------
def vis(img, asc_params, residual):
    H,W = img.shape
    recon = sum(model_single(p,H,W) for p in asc_params)
    fig,ax = plt.subplots(1,3,figsize=(15,4))
    ax[0].imshow(np.abs(img),cmap='gray'); ax[0].set_title('Orig |I|')
    ax[1].imshow(np.abs(recon),cmap='gray');ax[1].set_title('Recon |Î|')
    ax[2].imshow(np.abs(residual),cmap='gray');ax[2].set_title('Residue')
    for p in asc_params:
        ax[0].plot(p[0], p[1],'rx'); ax[0].text(p[0],p[1],f'{int(p[4])}',color='y')
    plt.tight_layout();plt.show()

def main():
    # if len(sys.argv)<2:
    #     print("用法: python extract_asc.py  xxx.npy")
    #     return
    # img_path = sys.argv[1]
    img_path = r"E:\code\objectDetection\AConvNet-pytorch-main\dataset\soc\train\ZSU234\HB20001-10.npy"
    img = np.load(img_path)

    # --------- 新增: 三维 (phase,amp) 自动转换 ----------
    if img.ndim == 3 and img.shape[2] >= 2:
        phase = img[..., 0]
        amp   = img[..., 1]
        # 如果你的幅度是 dB，就把下一行改成  amp = 10**(amp/20)
        img = amp * np.exp(1j*phase)
        print("检测到 (H,W,C) 输入，已转换为复数矩阵。")

    amp_max = np.max(np.abs(img))
    img /= amp_max

    asc_params, resid = afem_ga_extract(img,
                                        pop_size=100,
                                        n_gen=300,
                                        max_k=10)
    print("\n======  结果  ======")
    for i,p in enumerate(asc_params,1):
        print(f"ASC{i}: (x,y)=({p[0]:.1f},{p[1]:.1f})  amp={p[2]:+.2f}{p[3]:+.2f}j  type={int(p[4])}")
    np.save('asc_params.npy', np.array(asc_params))
    vis(img, asc_params, resid)

if __name__ == '__main__':

    main()

    # import numpy as np
    # import matplotlib.pyplot as plt
    # import os
    #
    #
    # def visualize_sar_npy(npy_file_path):
    #     """
    #     加载并可视化 .npy 格式的 SAR 图像数据。
    #
    #     假定数据格式为 (height, width, channels)，其中：
    #     - 第一个通道 (索引 0) 是相位值。
    #     - 第二个通道 (索引 1) 是振幅值。
    #
    #     参数:
    #         npy_file_path (str): .npy 文件的路径。
    #     """
    #     # --- 1. 检查文件是否存在 ---
    #     if not os.path.exists(npy_file_path):
    #         print(f"错误：文件 '{npy_file_path}' 不存在。")
    #         return
    #     if not npy_file_path.lower().endswith('.npy'):
    #         print(f"警告：文件 '{npy_file_path}' 可能不是 npy 格式。")
    #
    #     # --- 2. 加载数据 ---
    #     try:
    #         print(f"正在加载数据从: {npy_file_path}")
    #         sar_data = np.load(npy_file_path)
    #         print(f"数据加载成功，形状为: {sar_data.shape}")
    #     except Exception as e:
    #         print(f"错误：加载 npy 文件时出错: {e}")
    #         return
    #
    #     # --- 3. 验证数据形状 ---
    #     if len(sar_data.shape) != 3:
    #         print(f"错误：期望的数据维度为 3 (h, w, c)，但获取到 {len(sar_data.shape)} 维。")
    #         return
    #     if sar_data.shape[2] != 2:
    #         print(f"错误：期望最后一个维度（通道数）为 2，但获取到 {sar_data.shape[2]}。")
    #         print("请确保第一个通道是相位，第二个通道是振幅。")
    #         return
    #
    #     height, width, channels = sar_data.shape
    #     print(f"图像尺寸: 高={height}, 宽={width}")
    #
    #     # --- 4. 提取通道 ---
    #     phase_channel = sar_data[:, :, 0]
    #     amplitude_channel = sar_data[:, :, 1]
    #
    #     print(f"振幅范围: min={np.min(amplitude_channel)}, max={np.max(amplitude_channel)}")
    #     print(f"相位范围: min={np.min(phase_channel)}, max={np.max(phase_channel)}")
    #
    #     # --- 5. 可视化 ---
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 创建一个包含1行2列子图的画布
    #
    #     # 可视化振幅通道 (通常使用灰度图)
    #     ax = axes[0]
    #     im_amp = ax.imshow(amplitude_channel, cmap='gray')
    #     ax.set_title('振幅通道 (Amplitude)')
    #     ax.set_xlabel('宽度 (pixels)')
    #     ax.set_ylabel('高度 (pixels)')
    #     fig.colorbar(im_amp, ax=ax, label='振幅值')
    #
    #     # 可视化相位通道
    #     # 相位通常在 [-π, π] 或 [0, 2π] 范围内循环
    #     # 可以使用循环色图 (如 'hsv', 'twilight') 或灰度图
    #     ax = axes[1]
    #     # cmap_phase = 'hsv' # 循环色图，适合表示角度
    #     cmap_phase = 'gray'  # 或者使用灰度图查看原始相位结构
    #     im_phase = ax.imshow(phase_channel, cmap=cmap_phase)
    #     ax.set_title(f'相位通道 (Phase - cmap={cmap_phase})')
    #     ax.set_xlabel('宽度 (pixels)')
    #     # ax.set_ylabel('高度 (pixels)') # Y轴标签通常在最左侧图中显示即可
    #     fig.colorbar(im_phase, ax=ax, label='相位值 (通常为弧度)')
    #
    #     plt.tight_layout()  # 调整子图间距
    #     plt.suptitle(f'SAR 数据可视化: {os.path.basename(npy_file_path)}', y=1.02)  # 添加总标题
    #     plt.show()
    #
    #     # --- (可选) 振幅对数缩放 ---
    #     # SAR 振幅通常具有很宽的动态范围，对数缩放可以增强低强度区域的对比度
    #     # 注意：需要处理零值或负值（如果存在的话）
    #     amplitude_log = np.log1p(np.abs(amplitude_channel))  # 使用 log1p 处理 0 值 (log(1+x))
    #
    #     plt.figure(figsize=(6, 5))
    #     plt.imshow(amplitude_log, cmap='gray')
    #     plt.title('振幅通道 (对数缩放)')
    #     plt.xlabel('宽度 (pixels)')
    #     plt.ylabel('高度 (pixels)')
    #     plt.colorbar(label='Log(1 + 振幅值)')
    #     plt.suptitle(f'对数缩放: {os.path.basename(npy_file_path)}')
    #     plt.show()
    #
    #
    # # --- 使用示例 ---
    # img_path = r"E:\code\objectDetection\AConvNet-pytorch-main\dataset\soc\train\ZSU234\HB20001-10.npy"
    #
    # # 运行可视化函数
    # visualize_sar_npy(img_path)

