import cv2
import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20, flip_vertical=False):
        if root_dir is not None:
            self.save_dir = root_dir / "eval_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.flip_vertical = bool(flip_vertical)
        self.frames = []

    def init(self, env=None, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        if env is not None:
            self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, "physics"):
                frame = env.physics.render(
                    height=self.render_size, width=self.render_size, camera_id=0
                )
            else:
                frame = env.render()
            self.record_frame(frame)

    def record_frame(self, frame, apply_flip=True, apply_resize=True):
        if self.enabled:
            frame = np.asarray(frame)
            if apply_flip and self.flip_vertical:
                frame = np.flipud(frame).copy()
            if apply_resize and (
                frame.shape[0] != self.render_size or frame.shape[1] != self.render_size
            ):
                frame = cv2.resize(
                    frame,
                    dsize=(self.render_size, self.render_size),
                    interpolation=cv2.INTER_CUBIC,
                )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / "train_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(
                obs[-3:].transpose(1, 2, 0),
                dsize=(self.render_size, self.render_size),
                interpolation=cv2.INTER_CUBIC,
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
