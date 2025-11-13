import os
import h5py
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (启用 3D 投影)
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# ========= 配置 =========
HDF5_PATH = "/datasets/data/input_data/benchmark1_0/h5_ur_1rgb/bread_on_table/success_episodes/train/1014_165432/data/trajectory.hdf5"
OUT_DIR   = "/mnt/aceluo/h5_ur_1rgb"
OUT_MP4   = os.path.join(OUT_DIR, "end_effector_xyz.mp4")
OUT_TXT   = os.path.join(OUT_DIR, "end_effector_xyz.txt")   # 保存 (t,x,y,z)
FPS       = 30
MAX_FRAMES = 3000
DPI       = 150
FIGSIZE   = (6, 6)
DECIMALS  = 3
NBINS     = 5
# =======================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_xyz(h5_path):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"❌ HDF5 文件不存在：{h5_path}")
    with h5py.File(h5_path, "r") as f:
        if "puppet/end_effector" not in f:
            raise KeyError("❌ 未找到数据集 'puppet/end_effector'")
        data = f["puppet/end_effector"][:]  # (T, >=3)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"❌ end_effector 形状异常，期望 (T, >=3)，实际 {data.shape}")
    xyz = data[:, :3].astype(np.float64)
    return xyz, xyz.shape[0]

def compute_ranges(xyz, pad_ratio=0.05, min_span=1e-6):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    span = np.maximum(maxs - mins, min_span)
    pad  = pad_ratio * span
    lo = mins - pad
    hi = maxs + pad
    return (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])

def fig_to_rgb_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        buf = fig.canvas.tostring_rgb()
        rgb = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
    except Exception:
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
        rgb = argb[:, :, 1:]
    return rgb

def print_and_save_txt(xyz, out_txt_path):
    T = xyz.shape[0]
    t_col = np.arange(T).reshape(-1, 1)
    out_arr = np.hstack([t_col, xyz])
    print("==== (t, x, y, z) 全量数据 ====")
    for row in out_arr:
        print(f"{int(row[0])} {row[1]:.9f} {row[2]:.9f} {row[3]:.9f}")
    print("================================")
    header = "t x y z"
    np.savetxt(out_txt_path, out_arr, fmt=["%d", "%.9f", "%.9f", "%.9f"], header=header, comments="")
    print(f"📝 TXT 已保存：{out_txt_path}")

# —— 修复后的轴格式化（不访问 _axinfo） ——
def format_axes(ax, decimals=3, nbins=5):
    fmt_major = FormatStrFormatter(f'%.{decimals}f')  # 禁用科学计数法，固定小数位
    ax.xaxis.set_major_formatter(fmt_major)
    ax.yaxis.set_major_formatter(fmt_major)
    ax.zaxis.set_major_formatter(fmt_major)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))

    ax.tick_params(pad=2, length=3, width=0.8)
    ax.grid(True, linewidth=0.6, alpha=0.6)

def main():
    ensure_dir(OUT_DIR)

    xyz, T = load_xyz(HDF5_PATH)
    print(f"✅ Loaded (x,y,z). T = {T} frames.")
    print(f"   输出视频：{OUT_MP4}")

    print_and_save_txt(xyz, OUT_TXT)

    stride = max(1, T // MAX_FRAMES)
    xyz_used = xyz[::stride]
    frames = xyz_used.shape[0]
    xs, ys, zs = xyz_used[:, 0], xyz_used[:, 1], xyz_used[:, 2]

    x_range, y_range, z_range = compute_ranges(xyz)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("End-Effector (x, y, z) over Time")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    format_axes(ax, decimals=DECIMALS, nbins=NBINS)

    writer = imageio.get_writer(
        OUT_MP4, fps=FPS, codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )

    try:
        for i in range(frames):
            ax.cla()
            ax.set_title("End-Effector")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            format_axes(ax, decimals=DECIMALS, nbins=NBINS)

            ax.plot(xs[:i+1], ys[:i+1], zs[:i+1], lw=2)
            ax.scatter(xs[i], ys[i], zs[i], s=30)

            orig_idx = i * stride
            text_lines = (
                f"t: {orig_idx} / {T-1}\n"
                f"x: {xs[i]:.{DECIMALS}f}\n"
                f"y: {ys[i]:.{DECIMALS}f}\n"
                f"z: {zs[i]:.{DECIMALS}f}"
            )
            ax.text2D(
                0.02, 0.95, text_lines,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.75),
                family="monospace", fontsize=10, va="top"
            )

            frame_rgb = fig_to_rgb_array(fig)
            writer.append_data(frame_rgb)

            if (i + 1) % 100 == 0:
                print(f"... 已渲染 {i+1}/{frames} 帧")

    finally:
        writer.close()
        plt.close(fig)

    print("🎬 MP4 已保存：", OUT_MP4)
    print("✅ 处理完成。")

if __name__ == "__main__":
    main()

