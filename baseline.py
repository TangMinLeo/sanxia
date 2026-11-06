import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from skimage.measure import find_contours
import shapely
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union


class AdaptiveBaselineProcessor:
    def __init__(self, trajectory_pts, grid_shape=(3500, 4000), resolution=10.0):
        """
        大范围航道基线方法自适应处理器
        :param trajectory_pts: 全局 AIS 轨迹点坐标数组 (N, 2)
        :param grid_shape: 栅格图像尺寸 (H, W)，默认 (3500, 4000)
        :param resolution: 栅格分辨率 (米/像素)，默认 10.0m
        """
        self.trajectory_pts = np.asarray(trajectory_pts)
        self.grid_shape = grid_shape
        self.res = resolution

    def _partition_spatial_blocks(self, block_size=500):
        """
        将 3500x4000 的大范围空间网格切分为若干个局部子方块 (Sub-blocks)
        """
        h, w = self.grid_shape
        blocks = []
        for y_start in range(0, h * int(self.res), block_size * int(self.res)):
            for x_start in range(0, w * int(self.res), block_size * int(self.res)):
                box = {
                    "xmin": x_start,
                    "xmax": x_start + block_size * int(self.res),
                    "ymin": y_start,
                    "ymax": y_start + block_size * int(self.res)
                }
                blocks.append(box)
        return blocks

    def adaptive_kdete_large_scale(self, block_size=500, quantile=0.80):
        """
        大范围分块自适应 KDE-T 算法：
        在每个子区块内根据局部轨迹点分布的标准差动态计算自适应高斯带宽 h
        """
        blocks = self._partition_spatial_blocks(block_size=block_size)
        all_polygons = []

        print(f"-> 正在执行大范围分块自适应 KDE-T (共划分 {len(blocks)} 个子区域)...")

        for idx, box in enumerate(blocks):
            # 1. 提取当前子区块内的轨迹点
            mask = (
                (self.trajectory_pts[:, 0] >= box["xmin"]) & 
                (self.trajectory_pts[:, 0] < box["xmax"]) &
                (self.trajectory_pts[:, 1] >= box["ymin"]) & 
                (self.trajectory_pts[:, 1] < box["ymax"])
            )
            local_pts = self.trajectory_pts[mask]
            
            # 若区块内点数过少，跳过
            if len(local_pts) < 15:
                continue

            # 2. 【自适应带宽估计】：基于局部点集空间标准差的 Silverman/Scott 变体法则
            local_std = np.mean(np.std(local_pts, axis=0))
            adaptive_h = max(20.0, min(120.0, 0.5 * local_std)) # 限制在合理物理范围内 (20m - 120m)

            # 3. 构造局部网格
            x_grid = np.arange(box["xmin"], box["xmax"], self.res)
            y_grid = np.arange(box["ymin"], box["ymax"], self.res)
            if len(x_grid) < 2 or len(y_grid) < 2:
                continue
            gx, gy = np.meshgrid(x_grid, y_grid)
            grid_coords = np.vstack([gx.ravel(), gy.ravel()]).T

            # 4. 局部 KDE 拟合与阈值分割
            try:
                kde = KernelDensity(bandwidth=adaptive_h, kernel='gaussian')
                kde.fit(local_pts)
                log_dens = kde.score_samples(grid_coords)
                dens = np.exp(log_dens).reshape(gx.shape)

                pos_dens = dens[dens > 1e-8]
                if len(pos_dens) == 0:
                    continue
                threshold = np.quantile(pos_dens, 1.0 - quantile)

                # 提取轮廓
                contours = find_contours(dens, threshold)
                for contour in contours:
                    x_pts = box["xmin"] + contour[:, 1] * self.res
                    y_pts = box["ymin"] + contour[:, 0] * self.res
                    pts = np.column_stack([x_pts, y_pts])
                    if len(pts) >= 4:
                        poly = Polygon(pts)
                        if poly.is_valid and not poly.is_empty:
                            all_polygons.append(poly)
            except Exception as e:
                continue

        # 5. 全局无缝拼接合并
        global_boundary = unary_union(all_polygons) if all_polygons else Polygon()
        return global_boundary

    def adaptive_gdbscan_large_scale(self, block_size=500, alpha_ratio=0.25):
        """
        大范围分块自适应 G-DBSCAN 算法：
        在每个子区块内根据局部点密度动态调整邻域半径 epsilon
        """
        blocks = self._partition_spatial_blocks(block_size=block_size)
        all_polygons = []

        print(f"-> 正在执行大范围分块自适应 G-DBSCAN (共划分 {len(blocks)} 个子区域)...")

        for idx, box in enumerate(blocks):
            mask = (
                (self.trajectory_pts[:, 0] >= box["xmin"]) & 
                (self.trajectory_pts[:, 0] < box["xmax"]) &
                (self.trajectory_pts[:, 1] >= box["ymin"]) & 
                (self.trajectory_pts[:, 1] < box["ymax"])
            )
            local_pts = self.trajectory_pts[mask]
            
            if len(local_pts) < 10:
                continue

            # 1. 【自适应邻域半径估计】：根据局部点间平均距离动态设定 epsilon
            # 避免宽阔江面与狭窄支流因固定 epsilon 产生聚类失效
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(5, len(local_pts))).fit(local_pts)
            distances, _ = nbrs.kneighbors(local_pts)
            avg_dist = np.mean(distances[:, -1])
            adaptive_eps = max(25.0, min(80.0, 1.5 * avg_dist))
            adaptive_min_samples = max(2, int(len(local_pts) * 0.02))

            # 2. 局部 DBSCAN 聚类
            try:
                db = DBSCAN(eps=adaptive_eps, min_samples=adaptive_min_samples).fit(local_pts)
                labels = db.labels_

                for label in set(labels):
                    if label == -1:
                        continue
                    c_pts = local_pts[labels == label]
                    if len(c_pts) < 4:
                        continue
                    
                    multi_pt = MultiPoint(c_pts)
                    poly = shapely.concave_hull(multi_pt, ratio=alpha_ratio)
                    
                    if poly.is_valid and not poly.is_empty:
                        all_polygons.append(poly)
            except Exception as e:
                continue

        # 3. 全局无缝拼接合并
        global_boundary = unary_union(all_polygons) if all_polygons else Polygon()
        return global_boundary


# =============================================================================
# 验证执行示例
# =============================================================================
if __name__ == "__main__":
    # 模拟 3500 x 4000 网格 (即 35km x 40km 范围) 的海量 AIS 轨迹点
    
    # x_min = 495000.0
    # x_max = 535000.0

    # y_min = 3385000.0
    # y_max = 3420000.0
    
    x_min = 501975.0
    x_max = 516975.0

    y_min = 3403000.0
    y_max = 3415000.0

    # row_num = (int(y_max) - int(y_min)) // 10
    # col_num = (int(x_max) - int(x_min)) // 10
    # print(row_num, col_num)

    # heatmap = np.zeros((row_num, col_num))
    track_points = []
    
    for i in range(1, 15):
        file_path = f'./track_2024/utm_{i}.npy'
        cordi = np.load(file_path)
        
        for j in range(len(cordi)):
            x = cordi[j, 0]
            y = cordi[j, 1]
            
            
            if x < x_min or x >= x_max or y < y_min or y >= y_max:
                continue
            
            track_points.append([x-x_min, y-y_min])
    
    np.save('./track_2024/track_points_small.npy', np.array(track_points))
    
    # # np.random.seed(42)
    # # mock_pts = np.random.uniform(500, 35000, size=(15000, 2))
    
    # # 初始化大范围自适应处理器
    # processor = AdaptiveBaselineProcessor(trajectory_pts=track_points, grid_shape=(3500, 4000), resolution=10.0)
    
    # # 运行自适应 KDE-T
    # kde_boundary = processor.adaptive_kdete_large_scale(block_size=500, quantile=0.80)
    # print(f"自适应 KDE-T 成功生成全局包络面积: {kde_boundary.area:.1f} 平方米")
    
    # # 运行自适应 G-DBSCAN
    # dbscan_boundary = processor.adaptive_gdbscan_large_scale(block_size=500, alpha_ratio=0.3)
    # print(f"自适应 G-DBSCAN 成功生成全局包络面积: {dbscan_boundary.area:.1f} 平方米")