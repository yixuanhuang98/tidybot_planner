import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

import cv2 as cv
import numpy as np


def rot_x(theta: float) -> np.ndarray:
	c, s = np.cos(theta), np.sin(theta)
	return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_y(theta: float) -> np.ndarray:
	c, s = np.cos(theta), np.sin(theta)
	return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rot_z(theta: float) -> np.ndarray:
	c, s = np.cos(theta), np.sin(theta)
	return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def euler_xyz_to_R(ex: float, ey: float, ez: float) -> np.ndarray:
	"""Compute rotation matrix from intrinsic XYZ Euler angles (roll->pitch->yaw).

	MuJoCo applies the rotations sequentially: first about X (roll), then about Y (pitch), then about Z (yaw).
	That corresponds to R = Rz(yaw) * Ry(pitch) * Rx(roll) when the rotation matrices
	are post-multiplied to transform coordinates from the local frame to the world frame.

	Because we want a world-to-camera rotation (camera frame), we first build the
	camera-to-world rotation as Rx * Ry * Rz and later transpose it.
	"""
	return rot_x(ex) @ rot_y(ey) @ rot_z(ez)


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> Tuple[np.ndarray, float, float, float, float]:
	fovy = np.deg2rad(fovy_deg)
	fy = (height / 2.0) / np.tan(fovy / 2.0)
	fx = fy  # assume square pixels
	cx, cy = width / 2.0, height / 2.0
	K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
	return K, fx, fy, cx, cy


def world_to_camera(point_world: np.ndarray, cam_pos_xyz: Tuple[float, float, float], cam_euler_xyz: Tuple[float, float, float]) -> np.ndarray:
	R_c2w = euler_xyz_to_R(*cam_euler_xyz)
	t_c2w = np.asarray(cam_pos_xyz, dtype=float).reshape(3)
	p_w = np.asarray(point_world, dtype=float).reshape(3)
	# world->camera: p_cam = R^T (p_w - t)
	return R_c2w.T @ (p_w - t_c2w)


def project_to_pixel(p_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float]:
	# Camera looks along -Z. Use depth = -Z (must be > 0 to be in front).
	depth = -float(p_cam[2])
	if depth <= 0:
		return float("nan"), float("nan"), depth
	u = fx * (p_cam[0] / depth) + cx
	v = fy * (-p_cam[1] / depth) + cy  # flip because camera +Y is up, pixels increase downward
	return float(u), float(v), depth


def load_overview_camera_from_xml(xml_path: Path, camera_name: str = "overview") -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, int, int]:
	tree = ET.parse(str(xml_path))
	root = tree.getroot()
	cam_el = None
	for cam in root.iter("camera"):
		if cam.attrib.get("name") == camera_name:
			cam_el = cam
			break
	if cam_el is None:
		raise RuntimeError(f"Camera '{camera_name}' not found in {xml_path}")

	pos_str = cam_el.attrib.get("pos", None)
	euler_str = cam_el.attrib.get("euler", None)
	fovy_str = cam_el.attrib.get("fovy", None)
	res_str = cam_el.attrib.get("resolution", None)
	if not all([pos_str, euler_str, fovy_str, res_str]):
		raise RuntimeError("Camera element missing required attributes 'pos', 'euler', 'fovy', or 'resolution'")

	pos = tuple(float(x) for x in pos_str.strip().split())
	euler = tuple(float(x) for x in euler_str.strip().split())
	fovy = float(fovy_str)
	w_str, h_str = res_str.strip().split()
	width, height = int(w_str), int(h_str)
	return (pos, euler, fovy, width, height)


def annotate_image(image_path: Path, u: float, v: float, output_path: Path, radius: int = 5) -> None:
	img = cv.imread(str(image_path), cv.IMREAD_COLOR)
	if img is None:
		raise RuntimeError(f"Failed to read image: {image_path}")

	h, w = img.shape[:2]
	if not (np.isfinite(u) and np.isfinite(v)):
		print("Projected point behind camera or invalid; annotating with text only.")
		cv.putText(img, "point invalid (behind camera)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	else:
		u_int, v_int = int(round(u)), int(round(v))
		# Draw if inside image bounds
		if 0 <= u_int < w and 0 <= v_int < h:
			cv.circle(img, (u_int, v_int), radius, (0, 0, 255), thickness=-1)
		else:
			cv.putText(img, f"point outside image: ({u_int}, {v_int})", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

	cv.imwrite(str(output_path), img)
	print(f"Annotated image written to: {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Project a world 3D point onto the overview camera and annotate the image.")
	parser.add_argument("--point", nargs=3, type=float, default=(1.0, -0.4, 0.33), help="World point XYZ to project (default: 1.0 -0.6 0.2)")
	parser.add_argument("--image", type=str, default=str(Path("overview_images") / "overview_000000.png"), help="Path to overview image to annotate")
	parser.add_argument("--model", type=str, default=str(Path("models") / "stanford_tidybot" / "cupboard_scene.xml"), help="Path to the cupboard_scene.xml model")
	parser.add_argument("--camera", type=str, default="overview", help="Camera name in the XML (default: overview)")
	parser.add_argument("--output", type=str, default=str(Path("overview_images") / "overview_000000_annotated.png"), help="Output annotated image path")
	args = parser.parse_args()

	# Resolve paths relative to this file to be robust when run from anywhere
	script_dir = Path(__file__).resolve().parent
	image_path = (script_dir / args.image).resolve()
	model_path = (script_dir / args.model).resolve()
	output_path = (script_dir / args.output).resolve()

	if not image_path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")
	if not model_path.exists():
		raise FileNotFoundError(f"Model XML not found: {model_path}")

	# Load camera parameters
	(cam_pos, cam_euler, fovy_deg, width, height) = load_overview_camera_from_xml(model_path, camera_name=args.camera)
	K, fx, fy, cx, cy = intrinsics_from_fovy(fovy_deg, width, height)
	print(f"Camera pos: {cam_pos}, euler(xyz rad): {cam_euler}, fovy: {fovy_deg}, res: {width}x{height}")
	print(f"K =\n{K}")

	# Project world point
	point_world = np.array(args.point, dtype=float)
	p_cam = world_to_camera(point_world, cam_pos, cam_euler)
	u, v, depth = project_to_pixel(p_cam, fx, fy, cx, cy)
	print(f"World point: {point_world}")
	print(f"Camera coords: {p_cam}, depth (forward) = {depth}")
	print(f"Pixel: (u, v) = ({u}, {v})")

	# Annotate image
	annotate_image(image_path, u, v, output_path)


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1) 