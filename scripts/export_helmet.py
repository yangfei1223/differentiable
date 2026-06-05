"""导出头盔训练数据 — 从 Blender background 调用。"""
import json
import math
import os
import random

import bpy
import mathutils

# ============================================================================
# 配置
# ============================================================================
SCENE_NAME = "helmet"
HIGH_POLY_COLLECTION = "Original_Model_Helmet"
LOW_POLY_OBJECT = "Helmet_LowPoly"
NUM_VIEWS = 200
CAMERA_RADIUS = 0          # 0 = auto
FOV_MIN = 35.0
FOV_MAX = 55.0
RENDER_SAMPLES = 128
RESOLUTION = 1024
_CAMERA_PREFIX = "__bake_cam_"

# 输出目录
OUTPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
))
from datetime import datetime
date_str = datetime.now().strftime("%y%m%d")
dataset_dir = os.path.join(OUTPUT_DIR, f"{SCENE_NAME}_{date_str}")
gt_dir = os.path.join(dataset_dir, "gt")
cameras_path = os.path.join(dataset_dir, "cameras.json")
os.makedirs(gt_dir, exist_ok=True)

print(f"[Output] {dataset_dir}")

# ============================================================================
# Step 1: 查找高模 — 先按集合名找，找不到则按对象名
# ============================================================================
coll = bpy.data.collections.get(HIGH_POLY_COLLECTION)
high_meshes = []

if coll:
    for obj in coll.objects:
        if obj and obj.type == "MESH":
            high_meshes.append(obj)
    print(f"[HighPoly] Collection '{coll.name}': {len(high_meshes)} meshes")
else:
    # 集合不存在，尝试直接找对象
    print(f"[Warning] Collection '{HIGH_POLY_COLLECTION}' not found, searching objects...")
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name != LOW_POLY_OBJECT:
            # 排除已知的钢琴低模
            if not obj.name.startswith("defaultMaterial") and not obj.name.startswith("Object_"):
                high_meshes.append(obj)
    print(f"[HighPoly] Found {len(high_meshes)} candidate meshes: {[m.name for m in high_meshes]}")

if not high_meshes:
    raise RuntimeError("No high poly meshes found!")

# ============================================================================
# Step 2: 隐藏低模，显示高模
# ============================================================================
low_poly = bpy.data.objects.get(LOW_POLY_OBJECT)
if low_poly:
    low_poly.hide_viewport = True
    low_poly.hide_render = True
    print(f"[Visibility] Hidden low poly: {low_poly.name}")

# 确保高模可见
if coll:
    coll.hide_viewport = False
    coll.hide_render = False

for obj in high_meshes:
    obj.hide_viewport = False
    obj.hide_render = False
print(f"[Visibility] Showing {len(high_meshes)} high poly meshes")

# ============================================================================
# Step 3: 包围盒 + 相机采样
# ============================================================================
all_points = []
for obj in high_meshes:
    for corner in obj.bound_box:
        all_points.append(obj.matrix_world @ mathutils.Vector(corner))

xs = [p.x for p in all_points]
ys = [p.y for p in all_points]
zs = [p.z for p in all_points]
center = mathutils.Vector((
    (min(xs) + max(xs)) / 2,
    (min(ys) + max(ys)) / 2,
    (min(zs) + max(zs)) / 2,
))
radius = max((p - center).length for p in all_points)
print(f"[BBox] center=({center.x:.2f}, {center.y:.2f}, {center.z:.2f}), radius={radius:.2f}")

cam_radius = CAMERA_RADIUS if CAMERA_RADIUS > 0 else radius * 2.0

golden_angle = math.pi * (3.0 - math.sqrt(5.0))
cameras = []
for i in range(NUM_VIEWS):
    t = (i + 0.5) / NUM_VIEWS
    theta = math.asin(math.sqrt(t))
    phi = i * golden_angle
    x = cam_radius * math.sin(theta) * math.cos(phi)
    y = cam_radius * math.sin(theta) * math.sin(phi)
    z = cam_radius * math.cos(theta)
    cameras.append((mathutils.Vector((x, y, z)) + center, center.copy(), mathutils.Vector((0, 0, 1))))
# 正上方视角
cameras.append((mathutils.Vector((center.x, center.y, center.z + cam_radius)), center.copy(), mathutils.Vector((0, 1, 0))))
print(f"[Sampling] {len(cameras)} cameras, radius={cam_radius:.2f}")

# ============================================================================
# Step 4: 渲染设置
# ============================================================================
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = RENDER_SAMPLES
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = "Standard"
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"

# 清理旧临时相机
for obj in list(bpy.data.objects):
    if obj.name.startswith(_CAMERA_PREFIX):
        bpy.data.objects.remove(obj, do_unlink=True)
for cam in list(bpy.data.cameras):
    if cam.name.startswith(_CAMERA_PREFIX):
        bpy.data.cameras.remove(cam)

# ============================================================================
# Step 5: 逐相机渲染
# ============================================================================
camera_records = []
total = len(cameras)

for idx, (pos, look_at, up) in enumerate(cameras):
    fov = random.uniform(FOV_MIN, FOV_MAX)
    cam_name = f"{_CAMERA_PREFIX}{idx:04d}"
    img_name = f"view_{idx:04d}.png"
    img_path = os.path.join(gt_dir, img_name)

    cam_data = bpy.data.cameras.new(cam_name)
    cam_obj = bpy.data.objects.new(cam_name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = pos
    direction = (look_at - pos).normalized()
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(fov)

    bpy.context.scene.camera = cam_obj
    bpy.context.scene.render.filepath = img_path
    bpy.ops.render.render(write_still=True)

    camera_records.append({
        "position": [round(pos.x, 6), round(pos.y, 6), round(pos.z, 6)],
        "look_at": [round(look_at.x, 6), round(look_at.y, 6), round(look_at.z, 6)],
        "up": [round(up.x, 6), round(up.y, 6), round(up.z, 6)],
        "fov_deg": round(fov, 1),
        "image_size": [RESOLUTION, RESOLUTION],
        "image_path": f"gt/{img_name}",
    })

    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"  [{idx+1}/{total}] {img_name}")

    bpy.data.objects.remove(cam_obj, do_unlink=True)
    for cam in list(bpy.data.cameras):
        if cam.name == cam_name:
            bpy.data.cameras.remove(cam)

# ============================================================================
# Step 6: 导出 cameras.json
# ============================================================================
with open(cameras_path, "w", encoding="utf-8") as f:
    json.dump({"blender_coordinate": True, "cameras": camera_records}, f, indent=2)

print(f"[Export] cameras.json ({total} views)")
print("=" * 60)
print(f"Done: {total} GT images + cameras.json")
print(f"Output: {dataset_dir}")
print("=" * 60)
