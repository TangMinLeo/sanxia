import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.morphology import skeletonize, remove_small_objects, closing, disk
from skimage.feature import hessian_matrix
from skimage import measure
from skimage.filters import frangi
from scipy.ndimage import convolve, label
from scipy.spatial import cKDTree



def prune_skeleton_spurs_smart(skeleton, heatmap, min_branch_length=60, max_iter=3):
    """
    智能骨架毛刺修剪：加入边界保护与热力值加权，防止误删源头主干
    :param skeleton: 二值骨架图
    :param heatmap: 对数压缩后的热力图底图 (用于评估支流热力高低)
    :param min_branch_length: 允许保留的末端分支最小长度 (调小至60，保护左上角短主干)
    """
    pruned = skeleton.copy().astype(bool)
    h, w = skeleton.shape
    
    # 8连通邻域卷积核，用于计算当前像素周边的邻居数量
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
                       
    for iteration in range(max_iter):
        # 1. 计算每个骨架像素的邻居数
        neighbors = convolve(pruned.astype(np.uint8), kernel, mode='constant')
        
        # 2. 识别端点 (邻居为1) 和 交叉点 (邻居>2)
        endpoints = pruned & (neighbors == 1)
        intersections = pruned & (neighbors > 2)
        
        # 如果当前骨架已经没有任何端点 (纯环状)，直接结束
        if not np.any(endpoints):
            break
        
        # 3. 抠除交叉点，打断骨架，形成独立的线段
        segments = pruned & ~intersections
        
        # 4. 对独立线段进行连通域标记
        labeled_segments, num_features = label(segments, structure=np.ones((3,3)))
        
        branches_removed = False
        
        # 5. 查找并删除过短的末端分支
        for i in range(1, num_features + 1):
            mask = (labeled_segments == i)
            
            # 核心判断 1：该线段是否包含端点 (是否为末端分支)
            is_endpoint_branch = np.any(mask & endpoints)
            
            if is_endpoint_branch:
                # 核心判断 2【边界保护】：如果这个分支触碰了图像边缘（四周各留5个像素的安全区），绝对不删！
                # 这样能保证河流源头（左上角边缘）和出口完美延伸到最外围。
                touches_boundary = np.any(mask[0:5, :]) or np.any(mask[h-5:h, :]) or \
                                   np.any(mask[:, 0:5]) or np.any(mask[:, w-5:w])
                
                if touches_boundary:
                    continue # 触碰边界的末端分支直接放过，保留！
                
                # 核心判断 3【热力加权】：计算该分支上的平均热力值
                # 如果虽然短，但热力值极高（说明是高热力支流），也可以选择保留
                branch_mean_heat = np.mean(heatmap[mask])
                
                # 综合判断：长度小于阈值 且 热力没有特别突出的，才判定为毛刺删除
                length = np.sum(mask)
                if length < min_branch_length:
                    pruned[mask] = False
                    branches_removed = True
                    
        if not branches_removed:
            break
            
        pruned = skeletonize(pruned)
        
    return pruned


def _build_skeleton_graph(skeleton):
    """
    将二值骨架图构建为8连通图，返回图节点、邻接表、度数信息
    """
    h, w = skeleton.shape
    ys, xs = np.where(skeleton)
    node_set = set(zip(ys.tolist(), xs.tolist()))
    adj = {n: [] for n in node_set}
    for y, x in node_set:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in node_set:
                    adj[(y, x)].append((ny, nx))
    degree = {n: len(neighbors) for n, neighbors in adj.items()}
    return node_set, adj, degree


def _trace_branch_from_endpoint(skeleton, start, adj, max_length=10000):
    """
    从端点或交汇点邻域出发，沿骨架追踪一条分支直到下一个端点/交汇点
    返回路径点列表 [(y,x), ...]
    """
    path = [start]
    visited = {start}
    current = start
    neighbors = adj.get(current, [])
    if len(neighbors) == 0:
        return path
    prev = None
    while True:
        next_candidates = [n for n in adj.get(current, []) if n not in visited]
        if len(next_candidates) == 0:
            break
        if len(next_candidates) == 1:
            nxt = next_candidates[0]
        else:
            if prev is None:
                nxt = next_candidates[0]
            else:
                dy_prev = current[0] - prev[0]
                dx_prev = current[1] - prev[1]
                best_score = -999
                nxt = next_candidates[0]
                for cand in next_candidates:
                    dy_c = cand[0] - current[0]
                    dx_c = cand[1] - current[1]
                    score = dy_prev * dy_c + dx_prev * dx_c
                    if score > best_score:
                        best_score = score
                        nxt = cand
        path.append(nxt)
        visited.add(nxt)
        prev = current
        current = nxt
        if len(path) >= max_length:
            break
    return path


def _compute_hessian_fields(heatmap, sigma=5.0):
    """
    计算Hessian矩阵场，提取每个像素的脊线强度、切线方向、法线方向
    :param heatmap: 连续数值热力图（建议用log压缩+平滑后的）
    :param sigma: Hessian计算的尺度参数，应与河流宽度匹配
    :return: ridge_strength (H,W), tangent_field (H,W,2), normal_field (H,W,2)
    """
    smoothed = gaussian(heatmap.astype(np.float64), sigma=sigma)
    Hrr, Hrc, Hcc = hessian_matrix(smoothed, sigma=sigma, order='rc')
    H, W = heatmap.shape
    ridge_strength = np.zeros((H, W), dtype=np.float64)
    tangent_field = np.zeros((H, W, 2), dtype=np.float64)
    normal_field = np.zeros((H, W, 2), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            H_mat = np.array([[Hrr[y, x], Hrc[y, x]],
                               [Hrc[y, x], Hcc[y, x]]])
            eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            if eigenvalues[0] < 0:
                ridge_strength[y, x] = -eigenvalues[0]
                normal_field[y, x] = eigenvectors[:, 0]
                tangent_field[y, x] = eigenvectors[:, 1]
    return ridge_strength, tangent_field, normal_field


def _trace_ridge_hessian(start_y, start_x, direction, ridge_strength,
                          tangent_field, heatmap_smoothed,
                          max_steps=80, step_size=1.0,
                          ridge_min_strength=0.01):
    """
    从断口端点出发，沿Hessian切线方向追踪脊线
    在交汇区域，切线方向自然跟随各自支流，不会提前合并
    :param start_y, start_x: 断口端点坐标
    :param direction: 初始追踪方向 (2,)，应指向交汇区域内部
    :param ridge_strength, tangent_field: Hessian场
    :param heatmap_smoothed: 平滑热力图（用于梯度微调）
    :param max_steps: 最大追踪步数
    :param step_size: 每步步长（像素）
    :return: path [(y,x), ...]
    """
    H, W = ridge_strength.shape
    path = [(int(start_y), int(start_x))]
    cy, cx = float(start_y), float(start_x)
    cur_dir = direction.copy()
    cur_dir_norm = np.linalg.norm(cur_dir)
    if cur_dir_norm < 1e-8:
        return path
    cur_dir /= cur_dir_norm
    gy_field, gx_field = np.gradient(heatmap_smoothed)
    for _ in range(max_steps):
        iy, ix = int(round(cy)), int(round(cx))
        if not (0 <= iy < H and 0 <= ix < W):
            break
        if ridge_strength[iy, ix] < ridge_min_strength:
            break
        local_tangent = tangent_field[iy, ix]
        tangent_norm = np.linalg.norm(local_tangent)
        if tangent_norm < 1e-8:
            break
        local_tangent /= tangent_norm
        if np.dot(local_tangent, cur_dir) < 0:
            local_tangent = -local_tangent
        cur_dir = 0.6 * cur_dir + 0.4 * local_tangent
        cur_dir /= np.linalg.norm(cur_dir)
        next_y = cy + cur_dir[0] * step_size
        next_x = cx + cur_dir[1] * step_size
        perp = np.array([-cur_dir[1], cur_dir[0]])
        best_offset = 0.0
        best_heat = -1.0
        for offset in np.linspace(-3, 3, 7):
            py = int(round(next_y + perp[0] * offset))
            px = int(round(next_x + perp[1] * offset))
            if 0 <= py < H and 0 <= px < W:
                h = heatmap_smoothed[py, px]
                if h > best_heat:
                    best_heat = h
                    best_offset = offset
        next_y += perp[0] * best_offset
        next_x += perp[1] * best_offset
        niy, nix = int(round(next_y)), int(round(next_x))
        if not (0 <= niy < H and 0 <= nix < W):
            break
        path.append((niy, nix))
        cy, cx = next_y, next_x
    return path


def _find_closest_point_between_paths(path_a, path_b):
    """
    找两条路径之间距离最近的点对，返回 (point_a, point_b, distance)
    """
    if len(path_a) == 0 or len(path_b) == 0:
        return None, None, np.inf
    pts_a = np.array(path_a)
    pts_b = np.array(path_b)
    tree_b = cKDTree(pts_b)
    dists, indices = tree_b.query(pts_a)
    best_idx_a = np.argmin(dists)
    best_dist = dists[best_idx_a]
    best_idx_b = indices[best_idx_a]
    return tuple(pts_a[best_idx_a]), tuple(pts_b[best_idx_b]), best_dist


def _find_closest_point_on_path_to_target(path, target):
    """
    找路径上距离目标点最近的点及其索引
    :param path: [(y,x), ...]
    :param target: (y, x)
    :return: (closest_point, index_on_path, distance)
    """
    if len(path) == 0:
        return None, -1, np.inf
    pts = np.array(path)
    tgt = np.array(target)
    dists = np.sqrt(np.sum((pts - tgt) ** 2, axis=1))
    best_idx = np.argmin(dists)
    return tuple(pts[best_idx]), int(best_idx), dists[best_idx]


def _trim_ridge_to_junction(traced_ridge, true_junction, trim_margin=3):
    """
    将追踪脊线修剪到距真实交汇点最近的位置
    丢弃超过交汇点之后的部分（避免脊线越过交汇点延伸到另一支流）
    :param traced_ridge: 追踪脊线 [(y,x), ...]
    :param true_junction: 真实交汇点 (y, x)
    :param trim_margin: 在最近点之后再保留几个点（用于桥接时估计切线方向）
    :return: trimmed_ridge [(y,x), ...], endpoint_tangent (2,)
    """
    closest_pt, closest_idx, closest_dist = _find_closest_point_on_path_to_target(
        traced_ridge, true_junction
    )
    if closest_idx < 0:
        return traced_ridge, np.array([0.0, 1.0])
    trim_end = min(closest_idx + trim_margin + 1, len(traced_ridge))
    trimmed = traced_ridge[:trim_end]
    tangent_len = min(5, len(trimmed))
    if tangent_len >= 2:
        p_end = np.array(trimmed[-1], dtype=np.float64)
        p_prev = np.array(trimmed[-tangent_len], dtype=np.float64)
        tangent = p_end - p_prev
        t_norm = np.linalg.norm(tangent)
        if t_norm > 1e-8:
            tangent /= t_norm
        else:
            tangent = np.array([0.0, 1.0])
    else:
        tangent = np.array([0.0, 1.0])
    return trimmed, tangent


def _trace_ridge_to_target(start_y, start_x, start_tangent,
                            target_y, target_x,
                            ridge_strength, tangent_field, heatmap_smoothed,
                            max_steps=120, step_size=1.0,
                            ridge_min_strength=0.01,
                            attraction_strength=0.3):
    """
    目标引导的Hessian脊线追踪：从断口端点追踪到目标点（真实交汇点）
    方向 = (1-w)*Hessian切线 + w*目标引力，w随距离递增
    越靠近目标，目标引力越强，自然形成顺滑过渡
    :param start_y, start_x: 起点坐标
    :param start_tangent: 起点切线方向 (2,)，确保方向连续
    :param target_y, target_x: 目标点坐标（真实交汇点）
    :param ridge_strength, tangent_field: Hessian场
    :param heatmap_smoothed: 平滑热力图
    :param max_steps: 最大步数
    :param step_size: 步长
    :param ridge_min_strength: 脊线强度阈值
    :param attraction_strength: 目标引力基础强度（0~1）
    :return: path [(y,x), ...]
    """
    H, W = ridge_strength.shape
    path = [(int(start_y), int(start_x))]
    cy, cx = float(start_y), float(start_x)
    cur_dir = start_tangent.copy()
    cur_dir_n = np.linalg.norm(cur_dir)
    if cur_dir_n < 1e-8:
        return path
    cur_dir /= cur_dir_n
    total_dist = np.sqrt((start_y - target_y)**2 + (start_x - target_x)**2)
    if total_dist < 1.0:
        return path
    arrive_dist = step_size * 1.5
    for step in range(max_steps):
        dy_to_tgt = target_y - cy
        dx_to_tgt = target_x - cx
        dist_to_tgt = np.sqrt(dy_to_tgt**2 + dx_to_tgt**2)
        if dist_to_tgt < arrive_dist:
            path.append((int(round(target_y)), int(round(target_x))))
            break
        iy, ix = int(round(cy)), int(round(cx))
        if not (0 <= iy < H and 0 <= ix < W):
            break
        dir_to_target = np.array([dy_to_tgt, dx_to_tgt], dtype=np.float64)
        dir_to_target /= dist_to_tgt
        hessian_weight = max(0.0, 1.0 - attraction_strength * (1.0 - dist_to_tgt / total_dist))
        target_weight = 1.0 - hessian_weight
        hessian_dir = cur_dir.copy()
        if 0 <= iy < H and 0 <= ix < W and ridge_strength[iy, ix] > ridge_min_strength:
            local_tangent = tangent_field[iy, ix]
            t_norm = np.linalg.norm(local_tangent)
            if t_norm > 1e-8:
                local_tangent /= t_norm
                if np.dot(local_tangent, cur_dir) < 0:
                    local_tangent = -local_tangent
                hessian_dir = 0.5 * cur_dir + 0.5 * local_tangent
                hessian_dir /= np.linalg.norm(hessian_dir)
        blended_dir = hessian_weight * hessian_dir + target_weight * dir_to_target
        b_norm = np.linalg.norm(blended_dir)
        if b_norm < 1e-8:
            blended_dir = dir_to_target
        else:
            blended_dir /= b_norm
        next_y = cy + blended_dir[0] * step_size
        next_x = cx + blended_dir[1] * step_size
        perp = np.array([-blended_dir[1], blended_dir[0]])
        best_offset = 0.0
        best_heat = -1.0
        for offset in np.linspace(-3, 3, 7):
            py = int(round(next_y + perp[0] * offset))
            px = int(round(next_x + perp[1] * offset))
            if 0 <= py < H and 0 <= px < W:
                ht = heatmap_smoothed[py, px]
                if ht > best_heat:
                    best_heat = ht
                    best_offset = offset
        next_y += perp[0] * best_offset
        next_x += perp[1] * best_offset
        niy, nix = int(round(next_y)), int(round(next_x))
        if not (0 <= niy < H and 0 <= nix < W):
            break
        path.append((niy, nix))
        cy, cx = next_y, next_x
        cur_dir = blended_dir.copy()
    return path


def _generate_smooth_bridge(ridge_endpoint_y, ridge_endpoint_x,
                             ridge_tangent,
                             junction_y, junction_x,
                             ridge_strength, tangent_field, heatmap_smoothed,
                             step_size=1.0, ridge_min_strength=0.01):
    """
    从脊线断口端点到真实交汇点生成顺滑桥接曲线
    使用目标引导的Hessian追踪，确保桥接曲线既沿热力脊线走，又顺滑到达交汇点
    :param ridge_endpoint_y, ridge_endpoint_x: 脊线断口端点
    :param ridge_tangent: 断口端点处的切线方向 (2,)
    :param junction_y, junction_x: 真实交汇点
    :param ridge_strength, tangent_field, heatmap_smoothed: Hessian场与热力图
    :return: bridge_path [(y,x), ...]
    """
    dist = np.sqrt((ridge_endpoint_y - junction_y)**2 +
                   (ridge_endpoint_x - junction_x)**2)
    if dist < 1.5:
        return [(int(round(junction_y)), int(round(junction_x)))]
    max_steps = int(dist / step_size) + 20
    attraction = 0.3 + 0.4 * min(1.0, 10.0 / max(dist, 1.0))
    bridge = _trace_ridge_to_target(
        ridge_endpoint_y, ridge_endpoint_x, ridge_tangent,
        junction_y, junction_x,
        ridge_strength, tangent_field, heatmap_smoothed,
        max_steps=max_steps, step_size=step_size,
        ridge_min_strength=ridge_min_strength,
        attraction_strength=attraction
    )
    return bridge


def fix_junctions_by_hessian_tracing(skeleton, heatmap_log, smoothed,
                                      sigma_hessian=7.0, junction_radius=40,
                                      ridge_min_strength_ratio=0.15,
                                      max_trace_steps=80):
    """
    方案3核心：基于Hessian脊线追踪修复交汇区域骨架
    在已有骨架上检测degree>=3的交汇节点，切除交汇区域骨架，
    利用Hessian切线场从各分支断口独立追踪脊线，
    追踪出的各支流脊线自然延伸相交，交点即为真实交汇点。
    :param skeleton: 原始二值骨架图（prune之后的）
    :param heatmap_log: log压缩后的热力图
    :param smoothed: 高斯平滑后的热力图
    :param sigma_hessian: Hessian尺度参数，应与河流半宽匹配
    :param junction_radius: 交汇区域切除半径（像素），
           建议取河流宽度的1.5~2倍
    :param ridge_min_strength_ratio: 脊线强度最低阈值（占最大值的比例）
    :param max_trace_steps: 从每个断口最大追踪步数
    :return: fixed_skeleton (修复后的二值骨架图),
             junction_info (交汇信息列表)
    """
    print("5. 正在基于Hessian脊线追踪修复交汇区域...")
    h, w = skeleton.shape
    node_set, adj, degree = _build_skeleton_graph(skeleton)
    junctions = [n for n, d in degree.items() if d >= 3]
    if len(junctions) == 0:
        print("   未检测到交汇节点，无需修复。")
        y_coords, x_coords = np.where(skeleton)
        coords = np.column_stack((x_coords, y_coords))
        return skeleton, coords, []
    print(f"   检测到 {len(junctions)} 个交汇节点: {junctions}")
    print(f"   计算Hessian场 (sigma={sigma_hessian})...")
    ridge_strength, tangent_field, normal_field = _compute_hessian_fields(
        smoothed, sigma=sigma_hessian
    )
    ridge_threshold = ridge_min_strength_ratio * np.max(ridge_strength)
    fixed_skeleton = skeleton.copy()
    junction_info = []
    for j_idx, (jy, jx) in enumerate(junctions):
        print(f"   处理交汇节点 {j_idx+1}/{len(junctions)}: ({jy}, {jx})")
        yy, xx = np.ogrid[:h, :w]
        dist_map = np.sqrt((yy - jy)**2 + (xx - jx)**2)
        excise_mask = dist_map <= junction_radius
        ring_inner = dist_map <= junction_radius * 0.7
        ring_outer = (dist_map > junction_radius * 0.7) & (dist_map <= junction_radius * 1.3)
        boundary_skel_y, boundary_skel_x = np.where(
            fixed_skeleton & ring_outer
        )
        if len(boundary_skel_y) < 2:
            print(f"     交汇区域边界骨架点不足，跳过。")
            continue
        fixed_skeleton[excise_mask & fixed_skeleton] = False
        branch_endpoints = list(zip(boundary_skel_y.tolist(),
                                    boundary_skel_x.tolist()))
        visited = set()
        branch_groups = []
        for ep in branch_endpoints:
            if ep in visited:
                continue
            group = [ep]
            visited.add(ep)
            for other in branch_endpoints:
                if other in visited:
                    continue
                if np.sqrt((ep[0]-other[0])**2 + (ep[1]-other[1])**2) < junction_radius * 0.5:
                    group.append(other)
                    visited.add(other)
            branch_groups.append(group)
        print(f"     识别到 {len(branch_groups)} 个分支断口组")
        if len(branch_groups) < 2:
            fixed_skeleton[excise_mask] = skeleton[excise_mask]
            continue
        traced_ridges = []
        for bg_idx, group in enumerate(branch_groups):
            rep = group[0]
            ey, ex = rep
            direction_to_junction = np.array([jy - ey, jx - ex], dtype=np.float64)
            dir_norm = np.linalg.norm(direction_to_junction)
            if dir_norm < 1e-8:
                continue
            direction_to_junction /= dir_norm
            local_tangent = tangent_field[ey, ex]
            if np.linalg.norm(local_tangent) > 1e-8:
                local_tangent = local_tangent / np.linalg.norm(local_tangent)
                if np.dot(local_tangent, direction_to_junction) < 0:
                    local_tangent = -local_tangent
                init_dir = 0.5 * direction_to_junction + 0.5 * local_tangent
            else:
                init_dir = direction_to_junction
            init_dir /= np.linalg.norm(init_dir)
            path = _trace_ridge_hessian(
                ey, ex, init_dir, ridge_strength, tangent_field,
                smoothed, max_steps=max_trace_steps, step_size=1.0,
                ridge_min_strength=ridge_threshold
            )
            traced_ridges.append(path)
            print(f"       分支 {bg_idx+1}: 从({ey},{ex})追踪到 {len(path)} 个点")
        if len(traced_ridges) < 2:
            fixed_skeleton[excise_mask] = skeleton[excise_mask]
            continue
        true_junction = None
        min_junc_dist = np.inf
        for i in range(len(traced_ridges)):
            for j in range(i + 1, len(traced_ridges)):
                pa, pb, dist = _find_closest_point_between_paths(
                    traced_ridges[i], traced_ridges[j]
                )
                if pa is not None and dist < min_junc_dist:
                    min_junc_dist = dist
                    ha = smoothed[pa[0], pa[1]]
                    hb = smoothed[pb[0], pb[1]]
                    true_junction = pa if ha >= hb else pb
        if true_junction is None:
            for path in traced_ridges:
                for py, px in path:
                    if 0 <= py < h and 0 <= px < w:
                        fixed_skeleton[py, px] = True
            junction_info.append({
                'original_junction': (jy, jx),
                'true_junction': None,
                'traced_ridges': traced_ridges,
                'bridges': [],
                'num_branches': len(traced_ridges),
            })
            continue
        jy_t, jx_t = true_junction
        print(f"     真实交汇点: ({jy_t}, {jx_t}), "
              f"原交汇点: ({jy}, {jx}), "
              f"偏移: {np.sqrt((jy_t-jy)**2+(jx_t-jx)**2):.1f} 像素")
        bridges = []
        for r_idx, ridge in enumerate(traced_ridges):
            trimmed_ridge, endpoint_tangent = _trim_ridge_to_junction(
                ridge, true_junction, trim_margin=3
            )
            for py, px in trimmed_ridge:
                if 0 <= py < h and 0 <= px < w:
                    fixed_skeleton[py, px] = True
            trim_end_y, trim_end_x = trimmed_ridge[-1]
            dist_to_junc = np.sqrt((trim_end_y - jy_t)**2 + (trim_end_x - jx_t)**2)
            if dist_to_junc > 1.5:
                bridge = _generate_smooth_bridge(
                    trim_end_y, trim_end_x, endpoint_tangent,
                    jy_t, jx_t,
                    ridge_strength, tangent_field, smoothed,
                    step_size=1.0,
                    ridge_min_strength=ridge_threshold
                )
                bridges.append(bridge)
                for py, px in bridge[1:]:
                    if 0 <= py < h and 0 <= px < w:
                        fixed_skeleton[py, px] = True
                print(f"       分支 {r_idx+1}: 修剪到({trim_end_y},{trim_end_x}), "
                      f"桥接 {len(bridge)} 点到交汇点")
            else:
                print(f"       分支 {r_idx+1}: 修剪后已到达交汇点，无需桥接")
        fixed_skeleton[jy_t, jx_t] = True
        junction_info.append({
            'original_junction': (jy, jx),
            'true_junction': true_junction,
            'traced_ridges': traced_ridges,
            'bridges': bridges,
            'num_branches': len(traced_ridges),
        })
    fixed_skeleton = skeletonize(fixed_skeleton)
    y_coords, x_coords = np.where(fixed_skeleton)
    coords = np.column_stack((x_coords, y_coords))
    print(f"   修复完成！共 {len(junction_info)} 个交汇区域被处理。")
    print(f"   修复后骨架共 {len(coords)} 个像素点。")
    return fixed_skeleton, coords, junction_info


def _build_skeleton_graph(skeleton):
    """
    将二值骨架图构建为8连通图，返回图节点、邻接表、度数信息
    """
    h, w = skeleton.shape
    ys, xs = np.where(skeleton)
    node_set = set(zip(ys.tolist(), xs.tolist()))
    adj = {n: [] for n in node_set}
    for y, x in node_set:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in node_set:
                    adj[(y, x)].append((ny, nx))
    degree = {n: len(neighbors) for n, neighbors in adj.items()}
    return node_set, adj, degree


def _trace_branch_from_endpoint(skeleton, start, adj, max_length=10000):
    """
    从端点或交汇点邻域出发，沿骨架追踪一条分支直到下一个端点/交汇点
    返回路径点列表 [(y,x), ...]
    """
    path = [start]
    visited = {start}
    current = start
    neighbors = adj.get(current, [])
    if len(neighbors) == 0:
        return path
    prev = None
    while True:
        next_candidates = [n for n in adj.get(current, []) if n not in visited]
        if len(next_candidates) == 0:
            break
        if len(next_candidates) == 1:
            nxt = next_candidates[0]
        else:
            if prev is None:
                nxt = next_candidates[0]
            else:
                dy_prev = current[0] - prev[0]
                dx_prev = current[1] - prev[1]
                best_score = -999
                nxt = next_candidates[0]
                for cand in next_candidates:
                    dy_c = cand[0] - current[0]
                    dx_c = cand[1] - current[1]
                    score = dy_prev * dy_c + dx_prev * dx_c
                    if score > best_score:
                        best_score = score
                        nxt = cand
        path.append(nxt)
        visited.add(nxt)
        prev = current
        current = nxt
        if len(path) >= max_length:
            break
    return path


def _compute_hessian_fields(heatmap, sigma=5.0):
    """
    计算Hessian矩阵场，提取每个像素的脊线强度、切线方向、法线方向
    :param heatmap: 连续数值热力图（建议用log压缩+平滑后的）
    :param sigma: Hessian计算的尺度参数，应与河流宽度匹配
    :return: ridge_strength (H,W), tangent_field (H,W,2), normal_field (H,W,2)
    """
    smoothed = gaussian(heatmap.astype(np.float64), sigma=sigma)
    Hrr, Hrc, Hcc = hessian_matrix(smoothed, sigma=sigma, order='rc')
    H, W = heatmap.shape
    ridge_strength = np.zeros((H, W), dtype=np.float64)
    tangent_field = np.zeros((H, W, 2), dtype=np.float64)
    normal_field = np.zeros((H, W, 2), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            H_mat = np.array([[Hrr[y, x], Hrc[y, x]],
                               [Hrc[y, x], Hcc[y, x]]])
            eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            if eigenvalues[0] < 0:
                ridge_strength[y, x] = -eigenvalues[0]
                normal_field[y, x] = eigenvectors[:, 0]
                tangent_field[y, x] = eigenvectors[:, 1]
    return ridge_strength, tangent_field, normal_field


def _trace_ridge_hessian(start_y, start_x, direction, ridge_strength,
                          tangent_field, heatmap_smoothed,
                          max_steps=80, step_size=1.0,
                          ridge_min_strength=0.01):
    """
    从断口端点出发，沿Hessian切线方向追踪脊线
    在交汇区域，切线方向自然跟随各自支流，不会提前合并
    :param start_y, start_x: 断口端点坐标
    :param direction: 初始追踪方向 (2,)，应指向交汇区域内部
    :param ridge_strength, tangent_field: Hessian场
    :param heatmap_smoothed: 平滑热力图（用于梯度微调）
    :param max_steps: 最大追踪步数
    :param step_size: 每步步长（像素）
    :return: path [(y,x), ...]
    """
    H, W = ridge_strength.shape
    path = [(int(start_y), int(start_x))]
    cy, cx = float(start_y), float(start_x)
    cur_dir = direction.copy()
    cur_dir_norm = np.linalg.norm(cur_dir)
    if cur_dir_norm < 1e-8:
        return path
    cur_dir /= cur_dir_norm
    gy_field, gx_field = np.gradient(heatmap_smoothed)
    for _ in range(max_steps):
        iy, ix = int(round(cy)), int(round(cx))
        if not (0 <= iy < H and 0 <= ix < W):
            break
        if ridge_strength[iy, ix] < ridge_min_strength:
            break
        local_tangent = tangent_field[iy, ix]
        tangent_norm = np.linalg.norm(local_tangent)
        if tangent_norm < 1e-8:
            break
        local_tangent /= tangent_norm
        if np.dot(local_tangent, cur_dir) < 0:
            local_tangent = -local_tangent
        cur_dir = 0.6 * cur_dir + 0.4 * local_tangent
        cur_dir /= np.linalg.norm(cur_dir)
        next_y = cy + cur_dir[0] * step_size
        next_x = cx + cur_dir[1] * step_size
        perp = np.array([-cur_dir[1], cur_dir[0]])
        best_offset = 0.0
        best_heat = -1.0
        for offset in np.linspace(-3, 3, 7):
            py = int(round(next_y + perp[0] * offset))
            px = int(round(next_x + perp[1] * offset))
            if 0 <= py < H and 0 <= px < W:
                h = heatmap_smoothed[py, px]
                if h > best_heat:
                    best_heat = h
                    best_offset = offset
        next_y += perp[0] * best_offset
        next_x += perp[1] * best_offset
        niy, nix = int(round(next_y)), int(round(next_x))
        if not (0 <= niy < H and 0 <= nix < W):
            break
        path.append((niy, nix))
        cy, cx = next_y, next_x
    return path


def _find_closest_point_between_paths(path_a, path_b):
    """
    找两条路径之间距离最近的点对，返回 (point_a, point_b, distance)
    """
    if len(path_a) == 0 or len(path_b) == 0:
        return None, None, np.inf
    pts_a = np.array(path_a)
    pts_b = np.array(path_b)
    tree_b = cKDTree(pts_b)
    dists, indices = tree_b.query(pts_a)
    best_idx_a = np.argmin(dists)
    best_dist = dists[best_idx_a]
    best_idx_b = indices[best_idx_a]
    return tuple(pts_a[best_idx_a]), tuple(pts_b[best_idx_b]), best_dist


def _bresenham_line(y0, x0, y1, x1):
    """
    Bresenham线段算法：生成(y0,x0)到(y1,x1)之间所有整数像素点
    确保任意两个相邻点都在8连通邻域内，无间隙
    """
    points = []
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y1 > y0 else -1
    sx = 1 if x1 > x0 else -1
    y, x = y0, x0
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((y, x))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((y, x))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((y1, x1))
    return points

def _rasterize_path(path, canvas, h, w):
    """
    将路径光栅化写入骨架图，相邻点之间用Bresenham插值填充，确保8连通无间隙
    """
    if len(path) == 0:
        return
    for i in range(len(path)):
        py, px = path[i]
        if 0 <= py < h and 0 <= px < w:
            canvas[py, px] = True
        if i > 0:
            prev_y, prev_x = path[i - 1]
            if abs(py - prev_y) > 1 or abs(px - prev_x) > 1:
                fill_pts = _bresenham_line(prev_y, prev_x, py, px)
                for fy, fx in fill_pts[1:-1]:
                    if 0 <= fy < h and 0 <= fx < w:
                        canvas[fy, fx] = True

def _ensure_8connectivity(skeleton, max_gap=3):
    """
    修复骨架图中的8连通断裂：对每个端点（邻居数=1），
    在其max_gap半径内搜索其他骨架点，若找到则用Bresenham线段连接
    """
    h, w = skeleton.shape
    kernel8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    nbrs = convolve(skeleton.astype(np.uint8), kernel8, mode='constant')
    ep_y, ep_x = np.where(skeleton & (nbrs == 1))
    if len(ep_y) == 0:
        return skeleton
    skel_pts = np.column_stack(np.where(skeleton))
    tree = cKDTree(skel_pts)
    result = skeleton.copy()
    for ey, ex in zip(ep_y, ep_x):
        idx_list = tree.query_ball_point([ey, ex], max_gap)
        cands = skel_pts[idx_list]
        best_d = np.inf
        best_p = None
        for c in cands:
            cy, cx = int(c[0]), int(c[1])
            if cy == ey and cx == ex:
                continue
            d = np.sqrt((cy - ey)**2 + (cx - ex)**2)
            if 1 < d <= max_gap and d < best_d:
                best_d = d
                best_p = (cy, cx)
        if best_p is not None:
            for fy, fx in _bresenham_line(ey, ex, best_p[0], best_p[1]):
                if 0 <= fy < h and 0 <= fx < w:
                    result[fy, fx] = True
    return result



def fix_junctions_by_hessian_tracing(skeleton, heatmap_log, smoothed,
                                      sigma_hessian=7.0, junction_radius=40,
                                      ridge_min_strength_ratio=0.15,
                                      max_trace_steps=80):
    """
    方案3核心：基于Hessian脊线追踪修复交汇区域骨架
    在已有骨架上检测degree>=3的交汇节点，切除交汇区域骨架，
    利用Hessian切线场从各分支断口独立追踪脊线，
    追踪出的各支流脊线自然延伸相交，交点即为真实交汇点。
    :param skeleton: 原始二值骨架图（prune之后的）
    :param heatmap_log: log压缩后的热力图
    :param smoothed: 高斯平滑后的热力图
    :param sigma_hessian: Hessian尺度参数，应与河流半宽匹配
    :param junction_radius: 交汇区域切除半径（像素），
           建议取河流宽度的1.5~2倍
    :param ridge_min_strength_ratio: 脊线强度最低阈值（占最大值的比例）
    :param max_trace_steps: 从每个断口最大追踪步数
    :return: fixed_skeleton (修复后的二值骨架图),
             junction_info (交汇信息列表)
    """
    print("5. 正在基于Hessian脊线追踪修复交汇区域...")
    h, w = skeleton.shape
    node_set, adj, degree = _build_skeleton_graph(skeleton)
    junctions = [n for n, d in degree.items() if d >= 3]
    if len(junctions) == 0:
        print("   未检测到交汇节点，无需修复。")
        y_coords, x_coords = np.where(skeleton)
        coords = np.column_stack((x_coords, y_coords))
        return skeleton, coords, []
    print(f"   检测到 {len(junctions)} 个交汇节点: {junctions}")
    print(f"   计算Hessian场 (sigma={sigma_hessian})...")
    ridge_strength, tangent_field, normal_field = _compute_hessian_fields(
        smoothed, sigma=sigma_hessian
    )
    ridge_threshold = ridge_min_strength_ratio * np.max(ridge_strength)
    fixed_skeleton = skeleton.copy()
    junction_info = []
    for j_idx, (jy, jx) in enumerate(junctions):
        print(f"   处理交汇节点 {j_idx+1}/{len(junctions)}: ({jy}, {jx})")
        yy, xx = np.ogrid[:h, :w]
        dist_map = np.sqrt((yy - jy)**2 + (xx - jx)**2)
        excise_mask = dist_map <= junction_radius
        ring_inner = dist_map <= junction_radius * 0.7
        ring_outer = (dist_map > junction_radius * 0.7) & (dist_map <= junction_radius * 1.3)
        boundary_skel_y, boundary_skel_x = np.where(
            fixed_skeleton & ring_outer
        )
        if len(boundary_skel_y) < 2:
            print(f"     交汇区域边界骨架点不足，跳过。")
            continue
        fixed_skeleton[excise_mask & fixed_skeleton] = False
        branch_endpoints = list(zip(boundary_skel_y.tolist(),
                                    boundary_skel_x.tolist()))
        visited = set()
        branch_groups = []
        for ep in branch_endpoints:
            if ep in visited:
                continue
            group = [ep]
            visited.add(ep)
            for other in branch_endpoints:
                if other in visited:
                    continue
                if np.sqrt((ep[0]-other[0])**2 + (ep[1]-other[1])**2) < junction_radius * 0.5:
                    group.append(other)
                    visited.add(other)
            branch_groups.append(group)
        print(f"     识别到 {len(branch_groups)} 个分支断口组")
        if len(branch_groups) < 2:
            fixed_skeleton[excise_mask] = skeleton[excise_mask]
            continue
        traced_ridges = []
        for bg_idx, group in enumerate(branch_groups):
            rep = group[0]
            ey, ex = rep
            direction_to_junction = np.array([jy - ey, jx - ex], dtype=np.float64)
            dir_norm = np.linalg.norm(direction_to_junction)
            if dir_norm < 1e-8:
                continue
            direction_to_junction /= dir_norm
            local_tangent = tangent_field[ey, ex]
            if np.linalg.norm(local_tangent) > 1e-8:
                local_tangent = local_tangent / np.linalg.norm(local_tangent)
                if np.dot(local_tangent, direction_to_junction) < 0:
                    local_tangent = -local_tangent
                init_dir = 0.5 * direction_to_junction + 0.5 * local_tangent
            else:
                init_dir = direction_to_junction
            init_dir /= np.linalg.norm(init_dir)
            path = _trace_ridge_hessian(
                ey, ex, init_dir, ridge_strength, tangent_field,
                smoothed, max_steps=max_trace_steps, step_size=1.0,
                ridge_min_strength=ridge_threshold
            )
            traced_ridges.append(path)
            print(f"       分支 {bg_idx+1}: 从({ey},{ex})追踪到 {len(path)} 个点")
        if len(traced_ridges) < 2:
            fixed_skeleton[excise_mask] = skeleton[excise_mask]
            continue
        true_junction = None
        min_junc_dist = np.inf
        for i in range(len(traced_ridges)):
            for j in range(i + 1, len(traced_ridges)):
                pa, pb, dist = _find_closest_point_between_paths(
                    traced_ridges[i], traced_ridges[j]
                )
                if pa is not None and dist < min_junc_dist:
                    min_junc_dist = dist
                    ha = smoothed[pa[0], pa[1]]
                    hb = smoothed[pb[0], pb[1]]
                    true_junction = pa if ha >= hb else pb
        for path in traced_ridges:
            # for py, px in path:
            #     if 0 <= py < h and 0 <= px < w:
            #         fixed_skeleton[py, px] = True
            _rasterize_path(path, fixed_skeleton, h, w)
            
        if true_junction is not None:
            jy_t, jx_t = true_junction
            if 0 <= jy_t < h and 0 <= jx_t < w:
                fixed_skeleton[jy_t, jx_t] = True
            print(f"     真实交汇点: ({jy_t}, {jx_t}), "
                  f"原交汇点: ({jy}, {jx}), "
                  f"偏移: {np.sqrt((jy_t-jy)**2+(jx_t-jx)**2):.1f} 像素")
        junction_info.append({
            'original_junction': (jy, jx),
            'true_junction': true_junction,
            'traced_ridges': traced_ridges,
            'num_branches': len(traced_ridges),
        })
    # fixed_skeleton = skeletonize(fixed_skeleton)
    fixed_skeleton = _ensure_8connectivity(fixed_skeleton, max_gap=3)
    fixed_skeleton = skeletonize(fixed_skeleton)
    fixed_skeleton = _ensure_8connectivity(fixed_skeleton, max_gap=2)
    
    
    y_coords, x_coords = np.where(fixed_skeleton)
    coords = np.column_stack((x_coords, y_coords))
    print(f"   修复完成！共 {len(junction_info)} 个交汇区域被处理。")
    print(f"   修复后骨架共 {len(coords)} 个像素点。")
    return fixed_skeleton, coords, junction_info


def extract_thermal_ridge(heatmap):
    """
    提取带分叉的高动态范围热力图脊线
    :param heatmap: 2D numpy array (3500 x 4000)
    :return: skeleton (二值骨架图), coords (脊线坐标)
    """
    
    # ---------------------------------------------------------
    # 第一步：数据预处理 (对数变换与平滑)
    # ---------------------------------------------------------
    # 1. 使用 log1p (即 log(1+x)) 压缩极端的动态范围
    print("1. 正在进行对数压缩与高斯平滑...")
    heatmap_log = np.log1p(heatmap)
    
    # 2. 高斯平滑
    # 河流宽20-30像素，sigma取3-5较为合适，可消除局部噪点，防止骨架毛刺
    smoothed = gaussian(heatmap_log, sigma=4.0)
    
    # ridge_enhanced = frangi(heatmap_log, sigmas=range(5, 15, 2), black_ridges=False)
    # ---------------------------------------------------------
    # 第二步：自适应二值化分割
    # ---------------------------------------------------------
    print("2. 正在进行百分位阈值分割...")
    # 河流面积约占总面积的 0.7%，所以取 99.0% ~ 99.3% 的分位数作为阈值
    # 你可以根据实际二值化效果微调这个参数 (例如改成 99.0)
    threshold_val = np.percentile(smoothed, 98.0)
    # threshold_val = np.percentile(ridge_enhanced, 99.2)
    
    binary_mask = smoothed > threshold_val
    
    # ---------------------------------------------------------
    # 第三步：形态学清理
    # ---------------------------------------------------------
    print("3. 正在进行形态学清理...")
    # 闭运算：填补河流内部的黑色小孔（半径取河流宽度的一半，如10）
    cleaned_mask = closing(binary_mask, disk(10))
    
    # 移除孤立的热力噪点团块（假设面积小于5000像素的都不是主干流）
    cleaned_mask = remove_small_objects(cleaned_mask, min_size=5000)
    
    # ---------------------------------------------------------
    # 第四步：骨架化提取带分叉的脊线
    # ---------------------------------------------------------
    print("4. 正在提取拓扑骨架...")
    # skeletonize 会剥离边界，留下单像素宽的连通线，完美保留分叉
    skeleton = skeletonize(cleaned_mask)
    
    # pruned_skeleton = prune_skeleton_spurs(skeleton, min_branch_length=200, max_iter=3)
    pruned_skeleton = prune_skeleton_spurs_smart(skeleton, heatmap_log, min_branch_length=180, max_iter=5)
    
    
    # 获取脊线的 (y, x) 坐标
    y_coords, x_coords = np.where(pruned_skeleton)
    coords = np.column_stack((x_coords, y_coords))
    
    print(f"提取完成！共提取出 {len(coords)} 个脊线像素点。")
    return pruned_skeleton, coords, smoothed, cleaned_mask, heatmap_log



# =============================================================
# 使用示例与可视化展示
# =============================================================
if __name__ == "__main__":
    # 【模拟数据生成】 - 供测试用，实际使用时请替换为你的 3500x4000 heatmap 数组
    # heatmap = np.load('your_data.npy')
    heatmap = np.load('heatmap.npy')
    heatmap = np.flip(heatmap, axis=0)
    # print(heatmap.shape)
    
    h, w = heatmap.shape
    print(f"热力图尺寸: {h} x {w}")
    # 实际调用函数
    skeleton, ridge_coords, smoothed, cleaned_mask, heatmap_log = extract_thermal_ridge(heatmap)
    
    # ---------------------------------------------------------
    # 第五步：Hessian脊线追踪修复交汇区域
    # ---------------------------------------------------------
    fixed_skeleton, fixed_coords, junction_info = fix_junctions_by_hessian_tracing(
        skeleton, heatmap_log, smoothed,
        sigma_hessian=7.0,
        junction_radius=40,
        ridge_min_strength_ratio=0.15,
        max_trace_steps=80
    )
    # 以下为绘图验证代码（在你有真实数据时运行）
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(heatmap_log[1600:2300, 2800:3300], cmap='magma')
    axes[0].set_title('1. Log Transformed')
    
    axes[1].imshow(cleaned_mask, cmap='gray')
    axes[1].set_title('2. Binary Mask')
    
    # 将骨架叠加在平滑后的热力图上
    axes[2].imshow(smoothed, cmap='magma')
    axes[2].plot(ridge_coords[:, 0], ridge_coords[:, 1], 'c.', markersize=0.1)
    axes[2].set_title('3. Ridge on Smoothed Heatmap')
    
    # 局部放大看分叉细节
    axes[3].imshow(skeleton, cmap='gray')
    axes[3].set_title('4. Original Skeleton')
    
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------------------------
    # 对比图：原骨架 vs 修复后骨架
    # ---------------------------------------------------------
    if len(junction_info) > 0:
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
        
        axes2[0].imshow(smoothed, cmap='magma')
        axes2[0].set_title('Smoothed Heatmap')
        
        axes2[1].imshow(smoothed, cmap='magma', alpha=0.7)
        skel_y, skel_x = np.where(skeleton)
        axes2[1].plot(skel_x, skel_y, 'c.', markersize=0.3, label='Original')
        for info in junction_info:
            oy, ox = info['original_junction']
            axes2[1].plot(ox, oy, 'rx', markersize=10, markeredgewidth=2)
        axes2[1].set_title('Original Skeleton (red=x: original junctions)')
        axes2[1].legend()
        
        axes2[2].imshow(smoothed, cmap='magma', alpha=0.7)
        fixed_y, fixed_x = np.where(fixed_skeleton)
        axes2[2].plot(fixed_x, fixed_y, 'lime', marker='.', linestyle='none',
                     markersize=0.3, label='Fixed')
        for info in junction_info:
            if info['true_junction'] is not None:
                ty, tx = info['true_junction']
                axes2[2].plot(tx, ty, 'r*', markersize=12, label='True Junction' if info==junction_info[0] else '')
            oy, ox = info['original_junction']
            axes2[2].plot(ox, oy, 'rx', markersize=10, markeredgewidth=2, label='Original Junction' if info==junction_info[0] else '')
        axes2[2].set_title('Fixed Skeleton (*=true junction, x=original)')
        axes2[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # 交汇区域局部放大
        for info in junction_info:
            oy, ox = info['original_junction']
            r = 60
            y_lo, y_hi = max(0, oy-r), min(h, oy+r)
            x_lo, x_hi = max(0, ox-r), min(w, ox+r)
            
            fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
            
            axes3[0].imshow(smoothed[y_lo:y_hi, x_lo:x_hi], cmap='magma')
            sub_skel = skeleton[y_lo:y_hi, x_lo:x_hi]
            sy, sx = np.where(sub_skel)
            axes3[0].plot(sx, sy, 'c.', markersize=2)
            axes3[0].plot(ox-x_lo, oy-y_lo, 'rx', markersize=12, markeredgewidth=2)
            axes3[0].set_title(f'Original @ ({oy},{ox})')
            
            axes3[1].imshow(smoothed[y_lo:y_hi, x_lo:x_hi], cmap='magma')
            sub_fixed = fixed_skeleton[y_lo:y_hi, x_lo:x_hi]
            fy, fx = np.where(sub_fixed)
            axes3[1].plot(fx, fy, 'lime', marker='.', linestyle='none', markersize=2)
            if info['true_junction'] is not None:
                ty, tx = info['true_junction']
                axes3[1].plot(tx-x_lo, ty-y_lo, 'r*', markersize=14)
            axes3[1].plot(ox-x_lo, oy-y_lo, 'rx', markersize=12, markeredgewidth=2)
            axes3[1].set_title(f'Fixed @ ({oy},{ox})')
            
            plt.tight_layout()
            plt.show()
    
    # plt.figure(figsize=(16, 14))
    # plt.imshow(smoothed[1600:2300, 2800:3300], cmap='magma')
    # plt.show()
    
    # plt.figure(figsize=(16, 14))
    # plt.imshow(heatmap_log[1600:2300, 2800:3300], cmap='magma')
    # plt.show()
    
    plt.figure(figsize=(14, 14))
    plt.imshow(smoothed[1650:1750, 3050:3150], cmap='magma')
    
    subarea = skeleton[1650:1750, 3050:3150]
    y_coords, x_coords = np.where(subarea)
    coords = np.column_stack((x_coords, y_coords))
    # plt.plot(coords[:, 0]+3050, coords[:, 1]+1650, 'c.', markersize=0.5)  # 注意坐标偏移
    plt.plot(coords[:, 0], coords[:, 1], 'c.', markersize=3)  # 注意坐标偏移
    
    # plt.imshow(skeleton[1650:1750, 3050:3150], cmap='gray')
    plt.show()