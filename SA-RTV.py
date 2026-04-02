import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator
import scipy.ndimage as ndimage
from scipy.ndimage import binary_erosion
import trimesh
# [新增导入] 用于设置坐标轴刻度间距
from matplotlib.ticker import MultipleLocator

# ==========================================
# 0. IEEE 论文绘图全局设置 (大字体，粗边框)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 28,        # 增大坐标轴标签字体
    'axes.titlesize': 26,        # 增大标题字体
    'font.size': 24,             # 增大全局基础字体
    'legend.fontsize': 24,       # 增大图例字体
    'xtick.labelsize': 24,       # 增大X轴刻度数字字体
    'ytick.labelsize': 24,       # 增大Y轴刻度数字字体
    'axes.linewidth': 2.0,       # 边框粗细
    'lines.linewidth': 3.0       # 线条粗细
})

# ==========================================
# 1. 批量参数与基础配置
# ==========================================
BASE_OUTPUT_DIR = "SA_RTV_Batch_Results_Targeted"

# --- 核心批量测试网格 (按图表分布特性拼接) ---
part1 = np.logspace(-6, -5, 1, endpoint=False)
part2 = np.logspace(-5, -2, 3, endpoint=False)
part3 = np.logspace(-2, -1, 1)
LAMBDAS_TO_TEST = np.concatenate([part1, part2, part3]).tolist()

FIXED_EPSILON = 1e-6     
WEIGHT_SCALE = 1e-3      
BETA_SMOOTH  = 1e-8      
Z_CUTOFF     = 10.0      

DIMS = (40, 40, 100)     
VOXEL_SIZE = 0.5         
ROI_ORIGIN = [-10, -10, 0] 
OUTER_ITER = 5           
INNER_MAX_ITER = 200     

# ==========================================
# 2. 核心算法函数 
# ==========================================

def calculate_weighted_tv(P, weights, xv, yv, zv, beta):
    P_reshaped = P.reshape((xv, yv, zv))
    w_reshaped = weights.reshape((xv, yv, zv))
    gx = np.roll(P_reshaped, -1, axis=0) - P_reshaped
    gy = np.roll(P_reshaped, -1, axis=1) - P_reshaped
    gz = np.roll(P_reshaped, -1, axis=2) - P_reshaped
    gm = np.sqrt(gx**2 + gy**2 + gz**2 + beta**2)
    return np.sum(w_reshaped * gm)

def gradient_weighted_tv(P, weights, xv, yv, zv, beta):
    P_reshaped = P.reshape((xv, yv, zv))
    w_reshaped = weights.reshape((xv, yv, zv))
    gx = np.roll(P_reshaped, -1, axis=0) - P_reshaped
    gy = np.roll(P_reshaped, -1, axis=1) - P_reshaped
    gz = np.roll(P_reshaped, -1, axis=2) - P_reshaped
    denom = np.sqrt(gx**2 + gy**2 + gz**2 + beta**2)
    vx, vy, vz = w_reshaped*(gx/denom), w_reshaped*(gy/denom), w_reshaped*(gz/denom)
    div = (vx - np.roll(vx, 1, axis=0)) + (vy - np.roll(vy, 1, axis=1)) + (vz - np.roll(vz, 1, axis=2))
    return -div.flatten()

def loss_function(P, L, D, weights, lam, dims, beta):
    LP = np.maximum(L.dot(P), 1e-10)
    data_loss = np.sum(LP - D * np.log(LP)) / D.size
    reg_loss = lam * calculate_weighted_tv(P, weights, *dims, beta)
    return data_loss + reg_loss

def gradient_function(P, L, D, weights, lam, dims, beta):
    LP = np.maximum(L.dot(P), 1e-10)
    d_grad = L.T.dot(1.0 - D / LP) / D.size
    r_grad = lam * gradient_weighted_tv(P, weights, *dims, beta)
    return d_grad + r_grad

def update_weights(P, dims, eps_w, scale):
    P_r = P.reshape(dims)
    gx = np.roll(P_r, -1, axis=0) - P_r
    gy = np.roll(P_r, -1, axis=1) - P_r
    gz = np.roll(P_r, -1, axis=2) - P_r
    gm = np.sqrt(gx**2 + gy**2 + gz**2 + 1e-12)
    return scale / (gm + eps_w)

# ==========================================
# 3. 数据加载与预处理
# ==========================================

def load_and_prepare_data():
    print(">>> Loading measurement data and matrix...")
    try:
        processed_vectors = []
        for i in range(1, 6):
            vec = np.loadtxt(f"Extracted-Oout-{i}.txt").flatten()
            processed_vectors.append(vec)
        D = np.concatenate(processed_vectors) / 100.0
        D = np.maximum(D, 1e-9)

        L_sparse = load_npz('LengthMatrix.npz')
        L = LinearOperator(L_sparse.shape, matvec=L_sparse.dot, rmatvec=L_sparse.T.dot)
        return L, D
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def create_bf_mask_solid_filled(stl_path, translation_vector, dims_tuple, roi_origin, voxel_size_val):
    print(">>> 正在从STL文件创建几何掩码...")
    try:
        mesh = trimesh.load_mesh(stl_path)
    except Exception as e:
        print(f"错误：无法加载STL文件 '{stl_path}'. {e}")
        return None

    mesh.fill_holes()
    mesh.apply_scale(0.001)
    mesh.apply_translation(translation_vector)
    
    voxel_grid = mesh.voxelized(pitch=voxel_size_val)
    full_roi_mask = np.zeros(dims_tuple, dtype=bool) 
    origin_world = voxel_grid.transform[:3, 3]
    
    half_voxel_shift = 0.5 * voxel_size_val
    roi_origin_center = np.array(roi_origin) + half_voxel_shift
    start_index = np.round((origin_world - roi_origin_center) / voxel_size_val).astype(int)
    
    shape_vox = voxel_grid.matrix.shape
    end_index = start_index + shape_vox
    
    safe_start = np.maximum(0, start_index)
    safe_end = np.minimum(dims_tuple, end_index)

    crop_start = safe_start - start_index
    crop_end = crop_start + (safe_end - safe_start)
        
    full_roi_mask[safe_start[0]:safe_end[0], safe_start[1]:safe_end[1], safe_start[2]:safe_end[2]] = \
    voxel_grid.matrix[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]

    solid_mask_3d = ndimage.binary_fill_holes(full_roi_mask)
    return solid_mask_3d.flatten()

def generate_bounds_with_z_cutoff(bf_mask_flat, interior_mask_flat, min_density_shell, min_density_core, max_density, dims, origin, vs, z_cut):
    N = len(bf_mask_flat)
    bounds = np.empty((N, 2))
    bounds[~bf_mask_flat] = (0.0, 0.0)
    
    mask_true = bf_mask_flat
    shell_mask = mask_true & (~interior_mask_flat)
    bounds[shell_mask] = (min_density_shell, max_density)
    
    core_mask = mask_true & interior_mask_flat
    bounds[core_mask] = (min_density_core, max_density)
    
    xv, yv, zv = dims
    z_indices = np.arange(N) % zv
    z_phys = origin[2] + z_indices * vs
    bounds[z_phys < z_cut] = (0.0, 0.0)
    
    return list(map(tuple, bounds))

# ==========================================
# 4. 批量执行流程
# ==========================================

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    L, D = load_and_prepare_data()
    if L is None: exit()

    translation_vector = [0, -7.255554475, 3.1117]
    bf_mask_flat = create_bf_mask_solid_filled('full.stl', translation_vector, DIMS, ROI_ORIGIN, VOXEL_SIZE)
    if bf_mask_flat is None: exit()

    interior_mask_flat = binary_erosion(bf_mask_flat.reshape(DIMS), iterations=1).flatten()
    bounds = generate_bounds_with_z_cutoff(bf_mask_flat, interior_mask_flat, 0.0, 0.0, 5.5, DIMS, ROI_ORIGIN, VOXEL_SIZE, Z_CUTOFF)
    
    total_runs = len(LAMBDAS_TO_TEST)
    print("\n" + "="*50)
    print(f"开始批量处理：总计 {total_runs} 组 Lambda (固定 Eps={FIXED_EPSILON:.1e})")
    print("="*50)

    z_max = ROI_ORIGIN[2] + DIMS[2] * VOXEL_SIZE

    for i, lam in enumerate(LAMBDAS_TO_TEST):
        suffix = f"Lam_{lam:.2e}"
        print(f"\n[{i+1}/{total_runs}] 正在运行: {suffix} ...")
        start_time = time.time()
        
        P_curr = np.full(np.prod(DIMS), 0.1)
        P_curr[~bf_mask_flat] = 0.0
        weights = np.ones(np.prod(DIMS))

        for k in range(OUTER_ITER):
            print(f"    - Outer Iteration {k+1}/{OUTER_ITER}", end='\r')
            res = minimize(
                fun=loss_function, x0=P_curr,
                args=(L, D, weights, lam, DIMS, BETA_SMOOTH),
                method='L-BFGS-B', jac=gradient_function, bounds=bounds,
                options={'maxiter': INNER_MAX_ITER, 'ftol': 1e-10, 'disp': False}
            )
            P_curr = res.x
            if k < OUTER_ITER - 1:
                weights = update_weights(P_curr, DIMS, FIXED_EPSILON, WEIGHT_SCALE).flatten()
                
        print(f"    - 完成！耗时: {(time.time() - start_time)/60:.2f} 分钟.")

        P_3d = P_curr.reshape(DIMS)
        mid_x, mid_y = DIMS[0] // 2, DIMS[1] // 2

        # ----------------------------------------
        # 1. 处理 XZ 截面
        # ----------------------------------------
        slice_xz = P_3d[:, mid_y, :].T
        np.savetxt(os.path.join(BASE_OUTPUT_DIR, f"density_XZ_{suffix}.txt"), slice_xz, delimiter='\t', fmt='%.6e')

        fig_xz = plt.figure(figsize=(10, 8))
        ax_xz = fig_xz.add_subplot(111)
        im_xz = ax_xz.imshow(slice_xz, cmap='jet', origin='lower', vmin=0, vmax=5,
                             extent=[ROI_ORIGIN[0], ROI_ORIGIN[0]+DIMS[0]*VOXEL_SIZE, 
                                     ROI_ORIGIN[2], z_max])
        
        cb_xz = plt.colorbar(im_xz, ax=ax_xz)
        cb_xz.set_label(r"Density (g/cm$^3$)", size=28) 
        cb_xz.ax.tick_params(labelsize=24) 
        
        ax_xz.set_title(f"Lam: {lam:.2e}", pad=15) 
        ax_xz.set_xlabel("X (m)")
        ax_xz.set_ylabel("Z (m)")
        
        ax_xz.set_ylim(10.0, z_max)
        
        ax_xz.yaxis.set_major_locator(MultipleLocator(10)) 
        ax_xz.set_xticks([-5, 0, 5]) 
        
        # [修改点] 将 direction 设为 'out' 使刻度线朝外
        ax_xz.tick_params(direction='out', top=True, right=True, length=6, width=1.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_OUTPUT_DIR, f"plot_XZ_{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ----------------------------------------
        # 2. 处理 YZ 截面
        # ----------------------------------------
        slice_yz = P_3d[mid_x, :, :].T
        np.savetxt(os.path.join(BASE_OUTPUT_DIR, f"density_YZ_{suffix}.txt"), slice_yz, delimiter='\t', fmt='%.6e')

        fig_yz = plt.figure(figsize=(10, 8))
        ax_yz = fig_yz.add_subplot(111)
        im_yz = ax_yz.imshow(slice_yz, cmap='jet', origin='lower', vmin=0, vmax=5,
                             extent=[ROI_ORIGIN[1], ROI_ORIGIN[1]+DIMS[1]*VOXEL_SIZE, 
                                     ROI_ORIGIN[2], z_max])
        
        cb_yz = plt.colorbar(im_yz, ax=ax_yz)
        cb_yz.set_label(r"Density (g/cm$^3$)", size=28)
        cb_yz.ax.tick_params(labelsize=24)
        
        ax_yz.set_title(f"Lam: {lam:.2e}", pad=15)
        ax_yz.set_xlabel("Y (m)")
        ax_yz.set_ylabel("Z (m)")
        
        ax_yz.set_ylim(10.0, z_max)
        
        ax_yz.yaxis.set_major_locator(MultipleLocator(10))
        ax_yz.set_xticks([-5, 0, 5])
        
        # [修改点] 将 direction 设为 'out' 使刻度线朝外
        ax_yz.tick_params(direction='out', top=True, right=True, length=6, width=1.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_OUTPUT_DIR, f"plot_YZ_{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print("\n\n>>> 批量测试全部完成！")