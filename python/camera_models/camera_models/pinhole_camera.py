import numpy as np
import cv2
import yaml
from .camera_base import CameraBase


class PinholeCamera(CameraBase):
    """Standard pinhole camera with radial/tangential distortion (plumb-bob model)."""

    def __init__(self, width: int, height: int,
                 fx: float, fy: float, cx: float, cy: float,
                 k1: float = 0.0, k2: float = 0.0,
                 p1: float = 0.0, p2: float = 0.0,
                 k3: float = 0.0):
        super().__init__(width, height)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3

        self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self._dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    @classmethod
    def from_yaml(cls, path: str) -> "PinholeCamera":
        with open(path) as f:
            data = yaml.safe_load(f)
        ip = data["intrinsic_parameters"]
        dp = data.get("distortion_parameters", {})
        return cls(
            width=data["image_width"],
            height=data["image_height"],
            fx=ip["fx"], fy=ip["fy"], cx=ip["cx"], cy=ip["cy"],
            k1=dp.get("k1", 0.0), k2=dp.get("k2", 0.0),
            p1=dp.get("p1", 0.0), p2=dp.get("p2", 0.0),
            k3=dp.get("k3", 0.0),
        )

    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def dist_coeffs(self) -> np.ndarray:
        return self._dist

    def lift_projective(self, p: np.ndarray) -> np.ndarray:
        """Unproject 2D pixel to normalised 3D bearing [x, y, 1]."""
        pts = np.array([[p]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, self._K, self._dist)
        x, y = undist[0, 0]
        return np.array([x, y, 1.0])

    def space_to_plane(self, P: np.ndarray) -> np.ndarray:
        """Project 3D point (in camera frame) to pixel coordinates."""
        x = P[0] / P[2]
        y = P[1] / P[2]
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2**3
        radial = 1 + self.k1*r2 + self.k2*r4 + self.k3*r6
        xd = x*radial + 2*self.p1*x*y + self.p2*(r2 + 2*x**2)
        yd = y*radial + self.p1*(r2 + 2*y**2) + 2*self.p2*x*y
        return np.array([self.fx*xd + self.cx, self.fy*yd + self.cy])

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.undistort(img, self._K, self._dist)

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """points: (N,2) array of pixel coordinates -> (N,2) undistorted."""
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.undistortPoints(pts, self._K, self._dist, P=self._K)
        return undist.reshape(-1, 2)
