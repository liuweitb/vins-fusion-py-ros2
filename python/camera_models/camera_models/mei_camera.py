import numpy as np
import cv2
import yaml
from .camera_base import CameraBase


class MeiCamera(CameraBase):
    """Mei (unified projection) omnidirectional camera model."""

    def __init__(self, width: int, height: int,
                 xi: float,
                 fx: float, fy: float, cx: float, cy: float,
                 k1: float = 0.0, k2: float = 0.0,
                 p1: float = 0.0, p2: float = 0.0):
        super().__init__(width, height)
        self.xi = xi
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self._dist = np.array([k1, k2, p1, p2], dtype=np.float64)

    @classmethod
    def from_yaml(cls, path: str) -> "MeiCamera":
        with open(path) as f:
            data = yaml.safe_load(f)
        ip = data["intrinsic_parameters"]
        dp = data.get("distortion_parameters", {})
        return cls(
            width=data["image_width"],
            height=data["image_height"],
            xi=ip["xi"],
            fx=ip["gamma1"], fy=ip["gamma2"],
            cx=ip["u0"], cy=ip["v0"],
            k1=dp.get("k1", 0.0), k2=dp.get("k2", 0.0),
            p1=dp.get("p1", 0.0), p2=dp.get("p2", 0.0),
        )

    def lift_projective(self, p: np.ndarray) -> np.ndarray:
        """Unproject 2D pixel to unit 3D bearing vector."""
        mx = (p[0] - self.cx) / self.fx
        my = (p[1] - self.cy) / self.fy
        # Inverse distortion (iterative Newton)
        for _ in range(20):
            r2 = mx**2 + my**2
            r4 = r2**2
            delta = 1 + self.k1*r2 + self.k2*r4
            dx = 2*self.p1*mx*my + self.p2*(r2 + 2*mx**2)
            dy = self.p1*(r2 + 2*my**2) + 2*self.p2*mx*my
        # Lift to unit sphere
        xi = self.xi
        r2 = mx**2 + my**2
        z = (1 - xi * r2 + np.sqrt(1 + (1 - xi**2)*r2)) / (1 + r2)
        Xw = z * mx
        Yw = z * my
        Zw = z - xi
        n = np.sqrt(Xw**2 + Yw**2 + Zw**2)
        return np.array([Xw/n, Yw/n, Zw/n])

    def space_to_plane(self, P: np.ndarray) -> np.ndarray:
        """Project 3D camera-frame point to pixel."""
        norm = np.sqrt(P[0]**2 + P[1]**2 + P[2]**2)
        mx = P[0] / (P[2] + self.xi * norm)
        my = P[1] / (P[2] + self.xi * norm)
        r2 = mx**2 + my**2
        r4 = r2**2
        radial = 1 + self.k1*r2 + self.k2*r4
        dx = 2*self.p1*mx*my + self.p2*(r2 + 2*mx**2)
        dy = self.p1*(r2 + 2*my**2) + 2*self.p2*mx*my
        mxd = mx*radial + dx
        myd = my*radial + dy
        return np.array([self.fx*mxd + self.cx, self.fy*myd + self.cy])

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        K = self._K
        dist = self._dist
        return cv2.undistort(img, K, dist)
