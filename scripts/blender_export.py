"""
Blender 数据导出脚本 — 可微烘焙管线 (Differentiable Baking Pipeline)

完整工作流：
  1. 隐藏低模，显示高模集合
  2. 用高模包围盒生成相机位置（Fibonacci 半球采样）
  3. 逐相机 Cycles 渲染 GT 图像（高模）
  4. 导出 cameras.json
  5. 显示低模，隐藏高模
  6. 导出低模为 GLB

用法：
  blender --background your_scene.blend --python scripts/blender_export.py

依赖：Blender 3.x+，内置 mathutils / bpy
"""

import json
import math
import os
import random

import bpy
import mathutils

# ============================================================================
# 用户可配置参数
# ============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# --- 数据集命名 ---
SCENE_NAME = "piano"            # 场景名称，输出目录为 data/{SCENE_NAME}_{日期}
EXPORT_DATE = None               # None = 自动用今天日期，格式 yymmdd

# --- 场景对象 ---
HIGH_POLY_COLLECTION = "Original_Model"  # 高模集合名
LOW_POLY_OBJECT = "Merged_Model_Low"     # 低模对象名

# --- 相机采样 ---
NUM_VIEWS = 200               # 采样视角数量
CAMERA_RADIUS = 0              # 0 = 根据包围盒自动计算
FOV_MIN = 35.0                 # FOV 最小值（度）
FOV_MAX = 55.0                 # FOV 最大值（度）

# --- 渲染 ---
RENDER_SAMPLES = 128           # Cycles 采样数
RESOLUTION = 1024              # 渲染分辨率 (正方形)

# --- 导出 ---
EXPORT_FORMAT = "GLB"          # 低模导出格式: GLB 或 OBJ

# 相机命名前缀
_CAMERA_PREFIX = "__bake_cam_"


# ============================================================================
# 工具函数
# ============================================================================

def setup_visibility(render_high_poly=True):
    """设置场景可见性：渲染高模时隐藏低模，导出低模时隐藏高模。"""
    low_poly = bpy.data.objects.get(LOW_POLY_OBJECT)

    if render_high_poly:
        # 渲染 GT：隐藏低模，显示高模
        if low_poly:
            low_poly.hide_viewport = True
            low_poly.hide_render = True
            print(f"[可见性] 隐藏低模: {low_poly.name}")

        for coll in bpy.data.collections:
            if coll.name == HIGH_POLY_COLLECTION:
                coll.hide_viewport = False
                coll.hide_render = False
                for obj in coll.objects:
                    if obj and obj.type == 'MESH':
                        obj.hide_viewport = False
                        obj.hide_render = False
                print(f"[可见性] 显示高模集合: {coll.name}")
    else:
        # 导出低模：显示低模，隐藏高模
        if low_poly:
            low_poly.hide_viewport = False
            low_poly.hide_render = False

        for coll in bpy.data.collections:
            if coll.name == HIGH_POLY_COLLECTION:
                coll.hide_viewport = True
                coll.hide_render = True
                print(f"[可见性] 隐藏高模集合: {coll.name}")


def get_low_poly_object():
    """获取低模对象。"""
    obj = bpy.data.objects.get(LOW_POLY_OBJECT)
    if obj is None:
        raise RuntimeError(f"未找到低模对象: '{LOW_POLY_OBJECT}'")
    print(f"[低模] '{obj.name}' (顶点数: {len(obj.data.vertices)})")
    return obj


def compute_bounding_info(collection_name):
    """计算集合内所有 MESH 的合并包围盒中心和半径。"""
    coll = bpy.data.collections.get(collection_name)
    if coll is None:
        raise RuntimeError(f"未找到集合: '{collection_name}'")

    all_points = []
    for obj in coll.objects:
        if obj and obj.type == 'MESH':
            for corner in obj.bound_box:
                world_pt = obj.matrix_world @ mathutils.Vector(corner)
                all_points.append(world_pt)

    if not all_points:
        raise RuntimeError(f"集合 '{collection_name}' 中没有 MESH 对象")

    xs = [p.x for p in all_points]
    ys = [p.y for p in all_points]
    zs = [p.z for p in all_points]

    center = mathutils.Vector((
        (min(xs) + max(xs)) / 2.0,
        (min(ys) + max(ys)) / 2.0,
        (min(zs) + max(zs)) / 2.0,
    ))
    radius = max((p - center).length for p in all_points)

    print(f"[包围盒] 中心: ({center.x:.4f}, {center.y:.4f}, {center.z:.4f}), 半径: {radius:.4f}")
    return center, radius


def fibonacci_hemisphere(n, radius, center):
    """Fibonacci 半球均匀采样，附加一个正上方视角。"""
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    cameras = []

    for i in range(n):
        t = (i + 0.5) / n
        theta = math.asin(math.sqrt(t))
        phi = i * golden_angle

        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)

        pos = mathutils.Vector((x, y, z)) + center
        cameras.append((pos, center.copy(), mathutils.Vector((0, 0, 1))))

    top_pos = mathutils.Vector((center.x, center.y, center.z + radius))
    cameras.append((top_pos, center.copy(), mathutils.Vector((0, 1, 0))))

    print(f"[采样] 生成 {len(cameras)} 个相机位置")
    return cameras


def setup_render_settings():
    """配置 Cycles 渲染参数。"""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = RENDER_SAMPLES
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'Standard'
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.cycles.pixel_filter_type = 'GAUSSIAN'
    scene.cycles.filter_width = 1.5
    print(f"[渲染] Cycles {RENDER_SAMPLES} samples, {RESOLUTION}x{RESOLUTION}")


def create_camera(name, position, look_at, up, fov_deg):
    """创建临时相机。"""
    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = position
    direction = (look_at - position).normalized()
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians(fov_deg)
    return cam_obj


def cleanup_cameras():
    """清理所有临时相机。"""
    removed = 0
    for obj in list(bpy.data.objects):
        if obj.name.startswith(_CAMERA_PREFIX):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    for cam in list(bpy.data.cameras):
        if cam.name.startswith(_CAMERA_PREFIX):
            bpy.data.cameras.remove(cam)
    if removed:
        print(f"[清理] 移除 {removed} 个临时相机")


def export_low_poly(obj, output_dir):
    """导出低模。"""
    scene_dir = os.path.join(output_dir, "scene")
    os.makedirs(scene_dir, exist_ok=True)

    # 先清除所有选择，避免导出额外对象
    bpy.ops.object.select_all(action='DESELECT')

    # 确保低模可见且选中
    obj.hide_viewport = False
    obj.hide_render = False
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if EXPORT_FORMAT == "GLB":
        filepath = os.path.join(scene_dir, "lowpoly.glb")
        bpy.ops.export_scene.gltf(
            filepath=filepath,
            export_format='GLB',
            use_selection=True,
        )
    else:
        filepath = os.path.join(scene_dir, "lowpoly.obj")
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=True,
            export_materials=True,
            export_uv=True,
            export_normals=True,
            forward_axis='Y',
            up_axis='Z',
        )

    # 验证导出结果
    if not os.path.exists(filepath):
        raise RuntimeError(f"导出失败: 文件未生成 {filepath}")
    file_size = os.path.getsize(filepath)
    if file_size < 1024:  # < 1KB 视为异常
        raise RuntimeError(f"导出异常: 文件过小 ({file_size} bytes)，可能导出了空模型")
    print(f"[导出] 低模: {filepath} ({file_size / 1024 / 1024:.2f} MB, {len(obj.data.vertices)} vertices)")


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 60)
    print("可微烘焙数据导出")
    print("=" * 60)

    # 数据输出目录: data/{SCENE_NAME}_{yymmdd}
    from datetime import datetime
    date_str = EXPORT_DATE or datetime.now().strftime("%y%m%d")
    dataset_name = f"{SCENE_NAME}_{date_str}"
    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    gt_dir = os.path.join(dataset_dir, "gt")
    cameras_path = os.path.join(dataset_dir, "cameras.json")

    print(f"[输出] {dataset_dir}")

    cleanup_cameras()

    # ---- Step 1: 导出低模（先导出，渲染时不影响） ----
    low_poly = get_low_poly_object()
    export_low_poly(low_poly, dataset_dir)

    # ---- Step 2: 切换可见性 → 显示高模 ----
    setup_visibility(render_high_poly=True)

    # ---- Step 3: 用高模包围盒生成相机 ----
    center, bbox_radius = compute_bounding_info(HIGH_POLY_COLLECTION)
    cam_radius = CAMERA_RADIUS if CAMERA_RADIUS > 0 else bbox_radius * 2.0
    print(f"[采样] 相机半径: {cam_radius:.4f}")

    camera_positions = fibonacci_hemisphere(NUM_VIEWS, cam_radius, center)

    # ---- Step 4: 配置渲染 ----
    setup_render_settings()
    os.makedirs(gt_dir, exist_ok=True)

    # ---- Step 5: 逐相机渲染 GT ----
    camera_records = []
    total = len(camera_positions)

    for idx, (pos, look_at, up) in enumerate(camera_positions):
        fov = random.uniform(FOV_MIN, FOV_MAX)
        cam_name = f"{_CAMERA_PREFIX}{idx:04d}"
        img_name = f"view_{idx:04d}.png"
        img_path = os.path.join(gt_dir, img_name)

        cam_obj = create_camera(cam_name, pos, look_at, up, fov)
        bpy.context.scene.camera = cam_obj
        bpy.context.scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)

        camera_records.append({
            "position": [round(pos.x, 6), round(pos.y, 6), round(pos.z, 6)],
            "look_at": [round(look_at.x, 6), round(look_at.y, 6), round(look_at.z, 6)],
            "up": [round(up.x, 6), round(up.y, 6), round(up.z, 6)],
            "fov_deg": round(fov, 1),
            "image_size": [RESOLUTION, RESOLUTION],
            "image_path": f"gt/{img_name}",  # 相对于 dataset_dir
        })

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1}/{total}] {img_name}")

        # 清理临时相机
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        for cam in list(bpy.data.cameras):
            if cam.name == cam_name:
                bpy.data.cameras.remove(cam)

    # ---- Step 6: 导出 cameras.json ----
    with open(cameras_path, 'w', encoding='utf-8') as f:
        json.dump({"blender_coordinate": True, "cameras": camera_records}, f, indent=2)
    print(f"[导出] cameras.json ({total} 条)")

    cleanup_cameras()

    # ---- Step 7: 恢复可见性 ----
    setup_visibility(render_high_poly=False)

    print("=" * 60)
    print(f"导出完成: {total} 张 GT + 低模 + cameras.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
