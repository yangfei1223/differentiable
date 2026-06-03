"""相机模块 — Blender↔OpenGL 坐标转换与 MVP 矩阵。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 坐标转换常量
# ---------------------------------------------------------------------------
# Blender: Y-up 前方 → 但导出相机时通常为 Z-up 右手系
# OpenGL: Y-up, 相机朝 -Z
# 转换: Blender (X-right, Y-forward, Z-up) → OpenGL (X-right, Y-up, Z-back)
# 旋转轴: 绕 X 轴旋转 -90° (把 Z-up 映射到 Y-up)
_BL_GL_ROT = np.array(
    [
        [1,  0,  0, 0],
        [0,  0,  1, 0],
        [0, -1,  0, 0],
        [0,  0,  0, 1],
    ],
    dtype=np.float64,
)

_GL_BL_ROT = np.linalg.inv(_BL_GL_ROT)


def blender_to_opengl(matrix: np.ndarray) -> np.ndarray:
    """将 4×4 矩阵从 Blender 坐标系转换到 OpenGL 坐标系。"""
    return _BL_GL_ROT @ matrix


def opengl_to_blender(matrix: np.ndarray) -> np.ndarray:
    """将 4×4 矩阵从 OpenGL 坐标系转换到 Blender 坐标系。"""
    return _GL_BL_ROT @ matrix


# ---------------------------------------------------------------------------
# 四元数工具
# ---------------------------------------------------------------------------
def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数 q (w, x, y, z) 旋转向量 v。

    使用公式: v' = v + 2 * w * (u × v) + 2 * (u × (u × v))
    其中 u = (x, y, z), w = q[0]
    """
    w = q[0]
    u = q[1:4]
    cross_uv = np.cross(u, v)
    cross_uuv = np.cross(u, cross_uv)
    return v + 2.0 * w * cross_uv + 2.0 * cross_uuv


# ---------------------------------------------------------------------------
# Camera 数据类
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    """透视相机，使用 OpenGL 约定 (Y-up, 相机朝 -Z)。"""

    position: np.ndarray      # (3,) 世界坐标
    look_at: np.ndarray       # (3,) 注视目标
    up: np.ndarray            # (3,) 上方向
    fov_deg: float            # 垂直视场角 (度)
    image_width: int
    image_height: int
    near: float = 0.01
    far: float = 1000.0

    @classmethod
    def from_dict(cls, d: Dict) -> Camera:
        """从字典构建 Camera。

        支持两种格式:
        1. position + look_at + up (显式注视)
        2. position + rotation (四元数 wxyz) — 在 Blender 空间计算 look_at
        """
        position = np.array(d["position"], dtype=np.float64)
        fov_deg = float(d["fov_deg"])

        if "image_size" in d:
            image_width, image_height = int(d["image_size"][0]), int(d["image_size"][1])
        else:
            image_width = int(d.get("image_width", 800))
            image_height = int(d.get("image_height", 600))

        if "look_at" in d:
            look_at = np.array(d["look_at"], dtype=np.float64)
            up = np.array(d.get("up", [0.0, 1.0, 0.0]), dtype=np.float64)
        elif "rotation" in d:
            # rotation 为四元数 (w, x, y, z) — Blender 空间
            q = np.array(d["rotation"], dtype=np.float64)
            up = _quat_rotate(q, np.array([0.0, 0.0, 1.0]))  # Blender Z-up
            forward = _quat_rotate(q, np.array([0.0, -1.0, 0.0]))  # Blender 相机朝 -Y
            look_at = position + forward
        else:
            raise ValueError("字典必须包含 'look_at' 或 'rotation'")

        return cls(
            position=position,
            look_at=look_at,
            up=up,
            fov_deg=fov_deg,
            image_width=image_width,
            image_height=image_height,
            near=float(d.get("near", 0.01)),
            far=float(d.get("far", 1000.0)),
        )

    def view_matrix(self) -> np.ndarray:
        """OpenGL look-at 视点矩阵 (4×4)。"""
        f = self.look_at - self.position
        f = f / np.linalg.norm(f)

        up_norm = self.up / np.linalg.norm(self.up)
        s = np.cross(f, up_norm)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)

        M = np.eye(4, dtype=np.float64)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f

        T = np.eye(4, dtype=np.float64)
        T[0, 3] = -self.position[0]
        T[1, 3] = -self.position[1]
        T[2, 3] = -self.position[2]

        return M @ T

    def projection_matrix(self) -> np.ndarray:
        """OpenGL 透视投影矩阵 (4×4)。"""
        fov_rad = math.radians(self.fov_deg)
        aspect = self.image_width / self.image_height
        f = 1.0 / math.tan(fov_rad / 2.0)

        P = np.zeros((4, 4), dtype=np.float64)
        P[0, 0] = f / aspect
        P[1, 1] = f
        P[2, 2] = -(self.far + self.near) / (self.far - self.near)
        P[2, 3] = -(2.0 * self.far * self.near) / (self.far - self.near)
        P[3, 2] = -1.0
        return P

    def mvp(self) -> np.ndarray:
        """Model-View-Projection 矩阵 (numpy 4×4)。"""
        return self.projection_matrix() @ self.view_matrix()

    def mvp_torch(self):
        """Model-View-Projection 矩阵 (torch tensor [1, 4, 4])。"""
        import torch
        mvp_np = self.mvp()
        return torch.from_numpy(mvp_np).float().unsqueeze(0)


# ---------------------------------------------------------------------------
# JSON 加载
# ---------------------------------------------------------------------------
def load_cameras(json_path: str | Path) -> List[Camera]:
    """从 JSON 文件加载相机列表。

    JSON 格式::
        {
            "blender_coordinate": true,   // 可选，默认 false
            "cameras": [ ... ]
        }
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    blender_coordinate = data.get("blender_coordinate", False)
    cameras_raw = data["cameras"]

    result: List[Camera] = []
    for cam_dict in cameras_raw:
        cam = Camera.from_dict(cam_dict)

        if blender_coordinate:
            # 将 Blender 坐标系的 position, look_at, up 转换到 OpenGL
            pos4 = np.append(cam.position, 1.0)
            look4 = np.append(cam.look_at, 1.0)
            up4 = np.append(cam.up, 0.0)  # 方向向量 w=0

            cam.position = (_BL_GL_ROT @ pos4)[:3]
            cam.look_at = (_BL_GL_ROT @ look4)[:3]
            cam.up = (_BL_GL_ROT @ up4)[:3]

        result.append(cam)

    return result
