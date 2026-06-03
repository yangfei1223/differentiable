"""
Blender 数据导出脚本 — 可微烘焙管线 (Differentiable Baking Pipeline)

在 Blender 中运行此脚本，自动完成：
  1. 导出选中对象为 OBJ (低模)
  2. Fibonacci 半球均匀采样生成相机位置
  3. 逐相机 Cycles 渲染 GT 图像
  4. 导出 cameras.json

用法：
  blender --background your_scene.blend --python scripts/blender_export.py
  或在 Blender 脚本编辑器中直接运行。

依赖：Blender 3.x+，内置 mathutils / bpy
"""

import json
import math
import os
import sys

# Blender 内置模块（仅在 Blender 环境中可用）
import bpy
import bmesh
import mathutils

# ============================================================================
# 用户可配置参数
# ============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NUM_VIEWS = 100                 # 采样视角数量
CAMERA_RADIUS = 0               # 0 = 根据包围盒自动计算
RENDER_SAMPLES = 256            # Cycles 采样数
RESOLUTION = 1024               # 渲染分辨率 (正方形)
FOV_DEG = 45.0                  # 垂直视场角 (度)
TARGET_OBJECT = None            # 指定对象名，None = 当前选中对象

# 相机命名前缀（用于清理临时相机）
_CAMERA_PREFIX = "__bake_cam_"


# ============================================================================
# 工具函数
# ============================================================================

def get_target_object():
    """获取目标低模对象。

    优先使用 TARGET_OBJECT 指定名称查找，否则取当前选中的第一个网格对象。

    Returns:
        bpy.types.Object: 目标对象

    Raises:
        RuntimeError: 未找到有效对象
    """
    obj = None

    if TARGET_OBJECT is not None:
        obj = bpy.data.objects.get(TARGET_OBJECT)
        if obj is None:
            raise RuntimeError(f"未找到指定对象: '{TARGET_OBJECT}'")
    else:
        # 从选中对象中找第一个 Mesh
        for o in bpy.context.selected_objects:
            if o.type == 'MESH':
                obj = o
                break

    if obj is None:
        raise RuntimeError(
            "未找到目标对象。请先选中一个 Mesh 对象，或设置 TARGET_OBJECT 参数。"
        )

    print(f"[导出] 目标对象: '{obj.name}' (顶点数: {len(obj.data.vertices)})")
    return obj


def compute_bounding_info(obj):
    """计算对象的包围盒中心和半径。

    基于对象在世界空间中的包围盒顶点计算。

    Args:
        obj: bpy.types.Object — 目标对象

    Returns:
        tuple: (center, radius)
            - center: mathutils.Vector — 包围盒中心 (世界坐标)
            - radius: float — 包围球半径
    """
    # 获取世界空间中的包围盒角点
    bbox_world = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    # 计算中心
    xs = [v.x for v in bbox_world]
    ys = [v.y for v in bbox_world]
    zs = [v.z for v in bbox_world]
    center = mathutils.Vector((
        (min(xs) + max(xs)) / 2.0,
        (min(ys) + max(ys)) / 2.0,
        (min(zs) + max(zs)) / 2.0,
    ))

    # 计算半径 (中心到最远角点的距离)
    radius = max((v - center).length for v in bbox_world)

    print(f"[包围盒] 中心: ({center.x:.4f}, {center.y:.4f}, {center.z:.4f}), 半径: {radius:.4f}")
    return center, radius


def fibonacci_hemisphere(n, radius, center):
    """Fibonacci 半球均匀采样。

    使用 Fibonacci 螺旋算法在上半球面上生成均匀分布的采样点，
    并额外附加一个正上方 (Top) 视角。

    Args:
        n: int — 采样点数量 (不含 Top)
        radius: float — 采样球半径
        center: mathutils.Vector — 球心 (世界坐标)

    Returns:
        list[tuple]: 每个元素为 (position, look_at, up)
            - position: mathutils.Vector — 相机位置
            - look_at: mathutils.Vector — 注视目标
            - up: mathutils.Vector — 上方向
    """
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    cameras = []

    for i in range(n):
        # 极角 θ: 从 0 (顶) 到 π/2 (赤道) 均匀分布
        # 使用 (i + 0.5) / n 确保端点不重合
        t = (i + 0.5) / n
        theta = math.asin(math.sqrt(t))  # sqrt 映射使面积均匀

        # 方位角 φ: 黄金角递增
        phi = i * golden_angle

        # 球坐标转笛卡尔 (Blender Z-up)
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)

        pos = mathutils.Vector((x, y, z)) + center
        cameras.append((pos, center.copy(), mathutils.Vector((0, 0, 1))))

    # 附加正上方视角 (沿 -Z 方向俯视)
    top_pos = mathutils.Vector((center.x, center.y, center.z + radius))
    cameras.append((top_pos, center.copy(), mathutils.Vector((0, 1, 0))))

    print(f"[采样] 生成 {len(cameras)} 个相机位置 (含顶部视角)")
    return cameras


def export_obj(obj, filepath):
    """导出对象为 OBJ 文件。

    Args:
        obj: 目标对象
        filepath: 输出路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 先选中目标对象
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
        forward_axis='Y',
        up_axis='Z',
    )
    print(f"[导出] OBJ 已保存: {filepath}")


def setup_render_settings():
    """配置 Cycles 渲染参数。"""
    scene = bpy.context.scene

    # 切换到 Cycles
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    # 采样设置
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.preview_samples = RENDER_SAMPLES

    # 分辨率
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100

    # 色彩管理
    scene.view_settings.view_transform = 'Standard'

    # 背景透明（可选：如果场景有环境可以注释掉）
    scene.render.film_transparent = True

    # 输出格式
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'

    # 像素滤镜 (减少锯齿)
    scene.cycles.pixel_filter_type = 'GAUSSIAN'
    scene.cycles.filter_width = 1.5

    print(f"[渲染] 引擎: Cycles, 采样: {RENDER_SAMPLES}, 分辨率: {RESOLUTION}x{RESOLUTION}")


def create_camera(name, position, look_at, up, fov_deg):
    """创建临时相机并设置 Look-At。

    Args:
        name: 相机名称
        position: mathutils.Vector — 相机位置
        look_at: mathutils.Vector — 注视目标
        up: mathutils.Vector — 上方向
        fov_deg: float — 垂直视场角 (度)

    Returns:
        bpy.types.Object: 创建的相机对象
    """
    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # 设置位置
    cam_obj.location = position

    # 计算 Look-At 旋转
    direction = (look_at - position).normalized()
    # 使用 quaternion 从 direction 构建
    # Blender 默认相机朝向 -Z (局部坐标)
    # 我们需要旋转使相机 -Z 轴对齐 direction
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    # 设置 FOV
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians(fov_deg)

    # 设置为活动相机
    bpy.context.scene.camera = cam_obj

    return cam_obj


def render_view(cam_obj, output_path):
    """渲染当前视角并保存。

    Args:
        cam_obj: 相机对象
        output_path: 输出文件路径 (PNG)
    """
    bpy.context.scene.camera = cam_obj
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def cleanup_cameras():
    """清理所有临时相机对象和数据。"""
    removed = 0
    for obj in list(bpy.data.objects):
        if obj.name.startswith(_CAMERA_PREFIX):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    for cam in list(bpy.data.cameras):
        if cam.name.startswith(_CAMERA_PREFIX):
            bpy.data.cameras.remove(cam)
    print(f"[清理] 移除 {removed} 个临时相机")


def build_cameras_json(camera_data_list, output_dir):
    """构建 cameras.json 数据。

    Args:
        camera_data_list: list of dict — 每个相机记录
        output_dir: str — data 目录路径

    Returns:
        dict: cameras.json 完整内容
    """
    return {
        "blender_coordinate": True,
        "cameras": camera_data_list,
    }


# ============================================================================
# 主流程
# ============================================================================

def main():
    """执行完整的数据导出流程。"""
    print("=" * 60)
    print("可微烘焙数据导出 — Blender Export Script")
    print("=" * 60)

    # mathutils 已在模块顶部导入

    # 路径定义
    obj_path = os.path.join(OUTPUT_DIR, "scene", "lowpoly.obj")
    gt_dir = os.path.join(OUTPUT_DIR, "gt")
    cameras_path = os.path.join(OUTPUT_DIR, "cameras.json")

    # 清理可能存在的旧临时相机
    cleanup_cameras()

    # ----------------------------------------------------------
    # Step 1: 获取目标对象 & 导出 OBJ
    # ----------------------------------------------------------
    obj = get_target_object()
    export_obj(obj, obj_path)

    # ----------------------------------------------------------
    # Step 2: 计算包围信息 & 生成相机位置
    # ----------------------------------------------------------
    center, bbox_radius = compute_bounding_info(obj)

    # 自动计算相机半径：包围球半径的 2.0 倍，确保模型完全入画
    cam_radius = CAMERA_RADIUS if CAMERA_RADIUS > 0 else bbox_radius * 2.0
    print(f"[采样] 相机半径: {cam_radius:.4f} (包围球半径 x 2.0)")

    camera_positions = fibonacci_hemisphere(NUM_VIEWS, cam_radius, center)

    # ----------------------------------------------------------
    # Step 3: 配置渲染参数
    # ----------------------------------------------------------
    setup_render_settings()

    # 确保输出目录存在
    os.makedirs(gt_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Step 4: 逐相机渲染
    # ----------------------------------------------------------
    camera_records = []
    total = len(camera_positions)

    for idx, (pos, look_at, up) in enumerate(camera_positions):
        cam_name = f"{_CAMERA_PREFIX}{idx:04d}"
        img_name = f"view_{idx:04d}.png"
        img_path = os.path.join(gt_dir, img_name)

        print(f"[渲染] {idx + 1}/{total}: {img_name} — "
              f"位置 ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")

        # 创建相机并渲染
        cam_obj = create_camera(cam_name, pos, look_at, up, FOV_DEG)
        render_view(cam_obj, img_path)

        # 记录相机参数
        camera_records.append({
            "position": [round(pos.x, 6), round(pos.y, 6), round(pos.z, 6)],
            "look_at": [round(look_at.x, 6), round(look_at.y, 6), round(look_at.z, 6)],
            "up": [round(up.x, 6), round(up.y, 6), round(up.z, 6)],
            "fov_deg": FOV_DEG,
            "image_size": [RESOLUTION, RESOLUTION],
            "image_path": f"gt/{img_name}",
        })

    # ----------------------------------------------------------
    # Step 5: 导出 cameras.json
    # ----------------------------------------------------------
    cameras_data = build_cameras_json(camera_records, OUTPUT_DIR)
    with open(cameras_path, 'w', encoding='utf-8') as f:
        json.dump(cameras_data, f, indent=2, ensure_ascii=False)
    print(f"[导出] cameras.json 已保存: {cameras_path}")

    # ----------------------------------------------------------
    # Step 6: 清理临时相机
    # ----------------------------------------------------------
    cleanup_cameras()

    # ----------------------------------------------------------
    # 完成总结
    # ----------------------------------------------------------
    print("=" * 60)
    print(f"✓ 数据导出完成！")
    print(f"  低模:    {obj_path}")
    print(f"  GT 图像: {gt_dir}/ ({total} 张)")
    print(f"  相机参数: {cameras_path}")
    print(f"  总视角:  {total} (含顶部)")
    print("=" * 60)


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    main()
