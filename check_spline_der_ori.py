import numpy as np
from scipy import interpolate

# 创建参数化曲线（圆）
t = np.linspace(0, 2*np.pi, 100)
x = np.cos(t)
y = np.sin(t)

# 拟合
tck, u = interpolate.splprep([x, y], s=0)

# 比较不同参数点的导数方向
s_values = [0, 0.25, 0.5, 0.75]

for s in s_values:
    point = interpolate.splev(s, tck, der=0)
    derivative = interpolate.splev(s, tck, der=1)
    
    # 计算导数的模长和方向
    magnitude = np.sqrt(derivative[0]**2 + derivative[1]**2)
    angle = np.arctan2(derivative[1], derivative[0])
    
    print(f"s={s:.2f}: 位置=({point[0]:.2f}, {point[1]:.2f}), "
          f"导数方向角={np.degrees(angle):.1f}°, 速度={magnitude:.3f}")