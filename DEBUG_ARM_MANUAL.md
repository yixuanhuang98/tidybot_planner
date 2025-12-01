# 手动调试机械臂位置和姿态指南

## 方法1：使用代码中的手动调试模式

在 `policies.py` 的 `grasp_step == 0` 部分，我已经添加了手动调试代码（被注释掉了）。

### 步骤：

1. **找到手动调试代码**（大约在1094行附近）
2. **取消注释**手动调试部分（删除 `"""` 和 `"""`）
3. **注释掉**自动计算部分
4. **修改参数**：

```python
# 手动设置机械臂位置（相对于 base center）
MANUAL_ARM_POS = np.array([0.5, 0.0, 0.0])  # [x, y, z] 米

# 手动设置姿态（使用欧拉角，然后转换为四元数）
manual_roll = 0.0      # 绕 x 轴旋转（弧度）
manual_pitch = 0.0     # 绕 y 轴旋转（弧度）
manual_yaw = 0.0       # 绕 z 轴旋转（弧度）
```

### 常用姿态值：

- **手指朝下**：`pitch = -np.pi/2` 或 `[1.0, 0.0, 0.0, 0.0]`
- **手指水平向前**：`pitch = 0.0, yaw = 0.0`
- **手指水平向右**：`pitch = 0.0, yaw = np.pi/2`
- **手指水平向左**：`pitch = 0.0, yaw = -np.pi/2`

### 调试流程：

1. **先测试位置**：设置一个简单的位置，比如 `[0.3, 0.0, 0.0]`（base前方30cm）
2. **观察机械臂移动**：看是否到达目标位置
3. **再测试姿态**：固定位置，调整 `roll/pitch/yaw` 值
4. **逐步调整**：每次只改一个参数（比如只改 pitch），观察效果

---

## 方法2：在运行时打印当前值，然后复制使用

代码已经会打印调试信息，包括：
- `Current arm_pos (from base center)`
- `Current arm_quat`
- `Target arm_pos (from base center)`
- `Target arm_quat`

### 步骤：

1. **运行代码**，让它打印出当前值
2. **复制打印的 `Current arm_pos` 和 `Current arm_quat`**
3. **在手动调试模式中使用这些值**作为起点
4. **逐步调整**：比如把 z 值加 0.05（向上5cm），或者修改四元数

---

## 方法3：直接修改代码中的计算值

在自动计算代码中，直接修改这些变量：

```python
# 修改位置偏移
approach_offset = -0.05  # 改成 -0.10 或 0.0 试试

# 修改高度
object_3d_pos[2] + 0.10  # 改成 +0.15 或 +0.05 试试

# 修改姿态旋转角度
y_rot_to_horizontal = R.from_rotvec(np.array([0.0, 1.0, 0.0]) * (-math.pi / 2))
# 改成 -math.pi/4 或 -math.pi/3 试试不同的角度
```

---

## 理解坐标系

### arm_pos 坐标系：
- **相对于 base center**（不是 arm mount）
- x: 向前为正
- y: 向左为正（或向右，取决于定义）
- z: 向上为正

### arm_quat 四元数：
- 格式：`[x, y, z, w]`
- 表示末端执行器的旋转
- 常用值：
  - `[0.0, 0.0, 0.0, 1.0]` - 无旋转（默认）
  - `[1.0, 0.0, 0.0, 0.0]` - 绕 x 轴旋转180度（手指朝下）

### 欧拉角转四元数：
```python
from scipy.spatial.transform import Rotation as R
quat = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_quat()
```

---

## 调试技巧

1. **一次只改一个参数**：先改位置，再改姿态
2. **使用小步长**：每次改 0.05m 或 0.1 弧度
3. **记录有效值**：找到好的参数后，记录下来
4. **检查工作空间**：确保目标位置在机械臂可达范围内
5. **观察打印信息**：注意位置误差，如果 > 0.2m 可能超出范围

---

## 快速测试代码模板

```python
# 在 grasp_step == 0 中替换为：
MANUAL_ARM_POS = np.array([0.4, 0.0, 0.1])  # 前方40cm，高度10cm

# 测试不同的姿态
test_pitch = -np.pi / 4  # -45度（向下倾斜）
test_yaw = 0.0           # 不旋转
test_roll = 0.0          # 不旋转

MANUAL_ARM_QUAT = R.from_euler('xyz', [test_roll, test_pitch, test_yaw], degrees=False).as_quat()

action = {
    'base_pose': base_pose.copy(),
    'arm_pos': MANUAL_ARM_POS,
    'arm_quat': MANUAL_ARM_QUAT,
    'gripper_pos': np.array([0.0])
}
return action
```


