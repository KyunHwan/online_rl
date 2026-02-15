from __future__ import annotations

import sys
import queue as py_queue
from typing import Optional

import torch
from tensordict import TensorDict

import ray

@ray.remote
class ManualRewardLabelerActor:
    """
    Actor: runs a PySide6 app that pulls TensorDicts from a Ray queue,
    lets the user label per-frame binary rewards, and pushes back the labeled TensorDict.
    """

    def __init__(self, episode_queue_handle, replay_buffer_actor, img_frame_key: str="head", reward_key: str="reward", window_title: str = "Reward Labeler", ):
        self.episode_queue_handle = episode_queue_handle
        self.replay_buffer_actor = replay_buffer_actor
        self.img_frame_key = img_frame_key
        self.reward_key = reward_key
        self.window_title = window_title

    def start(self) -> int:
        """
        Blocking call that starts Qt event loop. Returns Qt exit code when window closes.
        """
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
        
        def torch_frame_to_qimage(frame_t: torch.Tensor) -> QImage:
            """
            frame_t:
            - uint8 or float tensor
            - either [H, W, 3] (HWC, RGB) or [3, H, W] (CHW, RGB)
            Returns a detached/copy QImage safe for Qt usage.
            """
            
            if frame_t.device.type != "cpu":
                frame_t = frame_t.cpu()

            if len(frame_t.shape) == 4: frame_t = frame_t.squeeze()

            if frame_t.ndim != 3:
                raise ValueError(f"Expected frame with 3 dims, got {tuple(frame_t.shape)}")

            # Accept CHW or HWC, convert to HWC for QImage
            if frame_t.shape[-1] == 3:
                # HWC
                frame_t = frame_t
            elif frame_t.shape[0] == 3:
                # CHW -> HWC
                frame_t = frame_t.permute(1, 2, 0)
            else:
                raise ValueError(
                    f"Expected frame shape [H,W,3] or [3,H,W], got {tuple(frame_t.shape)}"
                )

            # Ensure uint8 in [0,255]
            if frame_t.dtype != torch.uint8:
                if frame_t.is_floating_point():
                    # Common cases: [0,1] float or [0,255] float
                    # Heuristic: if max <= 1.0-ish, assume [0,1]
                    maxv = float(frame_t.max().item()) if frame_t.numel() > 0 else 1.0
                    if maxv <= 1.0 + 1e-3:
                        frame_t = frame_t * 255.0
                    frame_t = frame_t.round().clamp(0, 255).to(torch.uint8)
                else:
                    frame_t = frame_t.clamp(0, 255).to(torch.uint8)

            # IMPORTANT: QImage expects tight packed rows for our bytes_per_line calculation
            frame_t = frame_t.contiguous()

            np_img = frame_t.numpy()
            h, w, c = np_img.shape  # c should be 3
            bytes_per_line = c * w
            qimg = QImage(np_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            return qimg

        class VideoLabelerWindow(QMainWindow):
            def __init__(self, episode_queue_handle, replay_buffer_actor, title: str, img_frame_key: str="head", reward_key: str="reward",):
                super().__init__()
                self.episode_queue_handle = episode_queue_handle
                self.replay_buffer_actor = replay_buffer_actor
                self.img_frame_key = img_frame_key
                self.reward_key = reward_key

                self.setWindowTitle(title)

                # Current work item
                self.current_td: Optional[TensorDict] = None
                self.frames: Optional[torch.Tensor] = None  # [T,H,W,3]
                self.reward: Optional[torch.Tensor] = None  # [T]
                self._current_pixmap: Optional[QPixmap] = None

                # Widgets
                self.image_label = QLabel("Waiting for video...")
                self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.image_label.setMinimumSize(640, 360)

                self.slider = QSlider(Qt.Orientation.Horizontal)
                self.slider.setEnabled(False)

                self.frame_info = QLabel("Frame: - / -")
                self.reward_info = QLabel("Reward: -")

                self.btn_reward0 = QPushButton("Set Reward = 0")
                self.btn_reward1 = QPushButton("Set Reward = 1")
                self.btn_complete = QPushButton("Complete")

                self.btn_reward0.setEnabled(False)
                self.btn_reward1.setEnabled(False)
                self.btn_complete.setEnabled(False)

                # Layout
                root = QWidget()
                self.setCentralWidget(root)

                v = QVBoxLayout(root)
                v.addWidget(self.image_label)
                v.addWidget(self.slider)

                info_row = QHBoxLayout()
                info_row.addWidget(self.frame_info)
                info_row.addStretch(1)
                info_row.addWidget(self.reward_info)
                v.addLayout(info_row)

                btn_row = QHBoxLayout()
                btn_row.addWidget(self.btn_reward0)
                btn_row.addWidget(self.btn_reward1)
                btn_row.addStretch(1)
                btn_row.addWidget(self.btn_complete)
                v.addLayout(btn_row)

                # Signals
                self.slider.valueChanged.connect(self._on_slider_changed)
                self.btn_reward0.clicked.connect(lambda: self._set_reward(0))
                self.btn_reward1.clicked.connect(lambda: self._set_reward(1))
                self.btn_complete.clicked.connect(self._on_complete)

                # Poll queue for new TensorDicts when idle
                self.poll_timer = QTimer(self)
                self.poll_timer.setInterval(100)  # ms
                self.poll_timer.timeout.connect(self._poll_for_work)
                self.poll_timer.start()

            def resizeEvent(self, event):
                super().resizeEvent(event)
                self._rescale_pixmap()

            def _rescale_pixmap(self):
                if self._current_pixmap is None:
                    return
                scaled = self._current_pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.image_label.setPixmap(scaled)

            def _set_ui_busy(self, busy: bool):
                self.slider.setEnabled(busy)
                self.btn_reward0.setEnabled(busy)
                self.btn_reward1.setEnabled(busy)
                self.btn_complete.setEnabled(busy)

            def _poll_for_work(self):
                # Only pull a new item if we're not currently labeling something.
                if self.current_td is not None:
                    return

                try:
                    item = self.episode_queue_handle.get_nowait()
                except py_queue.Empty:
                    return

                # Robustly support either:
                # - item is TensorDict (if you put TD directly)
                # - item is ObjectRef[TensorDict] (your current writer behavior)
                try:
                    td = ray.get(item) if isinstance(item, ray.ObjectRef) else item

                    frames = td[self.img_frame_key]   # [T,H,W,3]
                    reward = td[self.reward_key]      # [T]
                    T = int(frames.shape[0])

                    # Only commit state AFTER we know td is valid.
                    self.current_td = td
                    self.frames = frames
                    self.reward = reward

                    self.slider.setRange(0, max(0, T - 1))
                    self.slider.setValue(0)
                    self._set_ui_busy(True)
                    self._render_frame(0)

                except Exception as e:
                    # Never get stuck half-initialized.
                    print(f"[GUI] Failed to load work item: {e}")
                    self.current_td = None
                    self.frames = None
                    self.reward = None
                    self._current_pixmap = None
                    self.image_label.setText("Waiting for video...")
                    self._set_ui_busy(False)
                    self.slider.setEnabled(False)
                    return

            def _render_frame(self, idx: int):
                if self.frames is None or self.reward is None:
                    return
                T = int(self.frames.shape[0])
                idx = max(0, min(idx, T - 1))

                frame_t = self.frames[idx]
                qimg = torch_frame_to_qimage(frame_t)
                pix = QPixmap.fromImage(qimg)
                self._current_pixmap = pix
                self._rescale_pixmap()

                r = int(self.reward[idx].item())
                self.frame_info.setText(f"Frame: {idx + 1} / {T}")
                self.reward_info.setText(f"Reward: {r}")

            def _on_slider_changed(self, value: int):
                if self.current_td is None:
                    return
                self._render_frame(int(value))

            def _set_reward(self, value: int):
                if self.reward is None:
                    return
                idx = int(self.slider.value())
                self.reward[idx] = int(value)
                self.reward_info.setText(f"Reward: {int(self.reward[idx].item())}")

            def _on_complete(self):
                """
                Push labeled TensorDict back to reader.
                """
                if self.current_td is None:
                    return

                # reward tensor is mutated in-place; current_td already reflects changes.
                self.replay_buffer_actor.add.remote(self.current_td)

                # Reset UI state
                self.current_td = None
                self.frames = None
                self.reward = None
                self._current_pixmap = None

                self.image_label.setText("Waiting for video...")
                self.image_label.setPixmap(QPixmap())
                self.frame_info.setText("Frame: - / -")
                self.reward_info.setText("Reward: -")
                self._set_ui_busy(False)
                self.slider.setEnabled(False)

            def closeEvent(self, event):
                """
                Try to unblock the reader cleanly on GUI close.

                Strategy:
                  - Drain one item from to_gui queue if present (frees bounded slot).
                  - Put a sentinel None into from_gui queue so reader can break.
                """
                try:
                    # Drain to free space if the reader is blocked on put().
                    try:
                        _ = self.episode_queue_handle.get_nowait()
                    except py_queue.Empty:
                        pass
                except Exception:
                    pass

                super().closeEvent(event)

        app = QApplication.instance() or QApplication(sys.argv)
        win = VideoLabelerWindow(self.episode_queue_handle, self.replay_buffer_actor, self.window_title, self.img_frame_key, self.reward_key)
        win.show()
        return app.exec()