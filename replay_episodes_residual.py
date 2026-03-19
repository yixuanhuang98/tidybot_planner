import argparse
import time
from pathlib import Path

import cv2 as cv

from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeReader
from mujoco_env import MujocoEnv


def replay_episode(
    env,
    episode_dir,
    show_images=False,
    execute_obs=False,
    save_video=False,
    video_dir="data/replay_videos",
    video_camera="base_image",
):
    # Reset env
    env.reset()

    # Load episode data
    reader = EpisodeReader(episode_dir)
    print(f"Loaded episode from {episode_dir}")

    # Optional video writer
    video_writer = None
    out_path = None
    if save_video:
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        first_obs = reader.observations[0]
        if video_camera not in first_obs:
            raise KeyError(
                f"Camera key '{video_camera}' not found in observation. "
                f"Available keys: {list(first_obs.keys())}"
            )
        frame = first_obs[video_camera]  # RGB, HxWx3
        h, w = frame.shape[:2]
        fps = int(round(1.0 / POLICY_CONTROL_PERIOD))
        out_path = Path(video_dir) / f"{Path(episode_dir).name}_{video_camera}.mp4"
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        video_writer = cv.VideoWriter(str(out_path), fourcc, fps, (w, h))

    start_time = time.time()
    for step_idx, (obs, action) in enumerate(zip(reader.observations, reader.actions)):
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        # Show image observations
        if show_images:
            window_idx = 0
            for k, v in obs.items():
                if getattr(v, "ndim", 0) == 3:
                    cv.imshow(k, cv.cvtColor(v, cv.COLOR_RGB2BGR))
                    cv.moveWindow(k, 640 * window_idx, 0)
                    window_idx += 1
            cv.waitKey(1)

        # Save video frame (from episode obs image)
        if video_writer is not None:
            frame_rgb = obs[video_camera]
            frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        # Execute action in env
        if execute_obs:
            env.step(obs)
        else:
            env.step(action)

    if video_writer is not None:
        video_writer.release()
        print(f"Saved replay video to {out_path}")


def main(args):
    # Create env
    if args.sim:
        env = MujocoEnv(render_images=False)
    else:
        from real_env import RealEnv

        env = RealEnv()

    try:
        episode_dirs = sorted([child for child in Path(args.input_dir).iterdir() if child.is_dir()])
        for episode_dir in episode_dirs:
            replay_episode(
                env,
                episode_dir,
                show_images=args.show_images,
                execute_obs=args.execute_obs,
                save_video=args.save_video,
                video_dir=args.video_dir,
                video_camera=args.video_camera,
            )
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/demos")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--show-images", action="store_true")
    parser.add_argument("--execute-obs", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", default="data/replay_videos")
    parser.add_argument("--video-camera", default="base_image")
    main(parser.parse_args())

