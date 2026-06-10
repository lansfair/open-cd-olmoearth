# Copyright (c) Open-CD. All rights reserved.
import os
import os.path as osp
import warnings
from collections.abc import Sequence as SequenceABC
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmengine.runner import Runner

from mmseg.engine import SegVisualizationHook
from mmseg.structures import SegDataSample
from opencd.registry import HOOKS
from opencd.visualization import CDLocalVisualizer


@HOOKS.register_module()
class CDVisualizationHook_Jie(SegVisualizationHook):
    """Change Detection Visualization Hook for Jie.

    This file is intended to replace only:

        opencd/engine/hooks/visualization_hook_Jie.py

    It does not require changing ``tools/test.py`` and does not introduce any
    new command-line arguments.

    Main fix:
        In change detection, ``output.img_path`` may be stored as one of:

            "A/xxx.png"
            ["A/xxx.png", "B/xxx.png"]
            [["A/xxx.png", "B/xxx.png"]]

        The original logic usually does something like:

            img_path = output.img_path[0]
            window_name = osp.basename(img_path).split(".")[0]

        When ``output.img_path[0]`` is still a list, ``osp.basename`` or
        ``fileio.get`` will raise:

            TypeError: expected str, bytes or os.PathLike object, not list

    This implementation first flattens ``output.img_path`` into a plain list of
    string paths, and then uses the first path for naming / image-shape lookup.
    """

    def __init__(self,
                 img_shape: tuple = None,
                 draw_on_from_to_img: bool = False,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self.img_shape = img_shape
        self.draw_on_from_to_img = draw_on_from_to_img
        if self.draw_on_from_to_img:
            warnings.warn('`draw_on_from_to_img` works only in '
                          'semantic change detection.')

        self._visualizer: CDLocalVisualizer = \
            CDLocalVisualizer.get_current_instance()

        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw

        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    @staticmethod
    def _is_path_like(obj) -> bool:
        """Return whether ``obj`` can be used as a filesystem path."""
        return isinstance(obj, (str, bytes, os.PathLike))

    @classmethod
    def _flatten_img_paths(cls, img_paths):
        """Flatten image path containers into ``list[str]``.

        Supported examples:
            "A/001.png"
            ["A/001.png", "B/001.png"]
            [["A/001.png", "B/001.png"]]
            np.array(["A/001.png", "B/001.png"], dtype=object)
        """
        if img_paths is None:
            return []

        if cls._is_path_like(img_paths):
            return [os.fspath(img_paths)]

        if isinstance(img_paths, np.ndarray):
            img_paths = img_paths.tolist()

        if isinstance(img_paths, SequenceABC):
            flattened = []
            for item in img_paths:
                flattened.extend(cls._flatten_img_paths(item))
            return flattened

        raise TypeError(
            '`img_path` must be a path-like object or a sequence of '
            f'path-like objects, but got {type(img_paths)}: {repr(img_paths)}'
        )

    @staticmethod
    def _safe_window_name(img_path: str) -> str:
        """Build a stable visualization window/file name from a path."""
        img_path = os.fspath(img_path)
        stem = osp.splitext(osp.basename(img_path))[0]
        return stem if stem else 'vis_result'

    def _read_rgb(self, img_path: str):
        """Read an RGB image from a path-like object."""
        img_path = os.fspath(img_path)
        img_bytes = fileio.get(img_path, backend_args=self.backend_args)
        return mmcv.imfrombytes(img_bytes, channel_order='rgb')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation/testing iteration."""
        if self.draw is False or mode == 'train':
            return

        if not self.every_n_inner_iters(batch_idx, self.interval):
            return

        for output in outputs:
            # Key fix: output.img_path can be nested list in CD tasks.
            img_paths = self._flatten_img_paths(output.img_path)

            if len(img_paths) == 0:
                warnings.warn('Skip visualization because `output.img_path` '
                              'is empty.')
                continue

            # Use the first temporal image path for naming / shape lookup.
            # In CD this is usually the A / pre-event image.
            img_path = img_paths[0]
            window_name = self._safe_window_name(img_path)

            img_from_to = []

            if self.img_shape is not None:
                assert len(self.img_shape) == 3, \
                    '`img_shape` should be (H, W, C)'
            else:
                img = self._read_rgb(img_path)
                self.img_shape = img.shape

            if self.draw_on_from_to_img:
                # Draw predictions on A/B images. Use the first two flattened
                # paths. If only one path exists, fall back to one image and
                # let the visualizer handle the normal binary result.
                if len(img_paths) < 2:
                    warnings.warn(
                        '`draw_on_from_to_img=True` requires two image paths '
                        f'but only got {len(img_paths)}: {img_paths}. '
                        'Will draw binary result only.'
                    )
                else:
                    for _img_path in img_paths[:2]:
                        img_from_to.append(self._read_rgb(_img_path))

            # Must be uint8 because CDLocalVisBackend asserts image.dtype.
            img = np.zeros(self.img_shape, dtype=np.uint8)

            # Do not pass out_file here.
            # tools/test.py already handles --show-dir by setting:
            #   cfg.visualizer['save_dir'] = args.show_dir
            # Therefore the existing CDLocalVisualizer/CDLocalVisBackend will
            # save images under the visualizer save_dir.
            self._visualizer.add_datasample(
                window_name,
                img,
                img_from_to,
                data_sample=output,
                show=self.show,
                wait_time=self.wait_time,
                step=runner.iter,
                draw_gt=False)
