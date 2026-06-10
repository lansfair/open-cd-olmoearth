# Copyright (c) Open-CD. All rights reserved.
import os
import os.path as osp
import warnings
from collections.abc import Sequence as SequenceABC
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmengine import mkdir_or_exist
from mmengine.runner import Runner

from mmseg.engine import SegVisualizationHook
from mmseg.structures import SegDataSample
from opencd.registry import HOOKS
from opencd.visualization import CDLocalVisualizer


@HOOKS.register_module()
class CDVisualizationHook_Jie(SegVisualizationHook):
    """Change Detection Visualization Hook.

    This patched version fixes two common issues in change-detection testing:

    1. ``output.img_path`` may be a nested list, for example:
       ``[['A/xxx.png', 'B/xxx.png']]`` or ``['A/xxx.png', 'B/xxx.png']``.
       The original implementation used ``output.img_path[0]`` directly.
       When that element is still a list, ``osp.basename`` raises:
       ``TypeError: expected str, bytes or os.PathLike object, not list``.

    2. ``tools/test.py --show-dir xxx`` normally injects ``test_out_dir`` into
       the visualization hook config. This hook now accepts ``test_out_dir`` and
       passes ``out_file`` to the visualizer, so results are saved to show-dir.

    Args:
        img_shape (tuple): If ``img_shape`` is given and
            ``draw_on_from_to_img`` is False, original images will not be read.
        draw_on_from_to_img (bool): Whether to draw semantic prediction results
            on the original two temporal images. Defaults to False.
        draw (bool): Whether to draw visualization. Defaults to False.
        interval (int): Visualization interval. Defaults to 50.
        show (bool): Whether to show visualization windows. Defaults to False.
        wait_time (float): Wait time for showing windows. Defaults to 0.
        backend_args (dict, optional): Backend args for file I/O.
        test_out_dir (str, optional): Output directory set by ``--show-dir``.
    """

    def __init__(self,
                 img_shape: tuple = None,
                 draw_on_from_to_img: bool = False,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None,
                 test_out_dir: Optional[str] = None):
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
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        self.test_out_dir = test_out_dir

        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    @staticmethod
    def _is_path_like(obj) -> bool:
        return isinstance(obj, (str, bytes, os.PathLike))

    @classmethod
    def _flatten_img_paths(cls, img_paths):
        """Flatten path-like / list / tuple / nested list image paths.

        Examples:
            'A/1.png' -> ['A/1.png']
            ['A/1.png', 'B/1.png'] -> ['A/1.png', 'B/1.png']
            [['A/1.png', 'B/1.png']] -> ['A/1.png', 'B/1.png']
        """
        if img_paths is None:
            return []

        if cls._is_path_like(img_paths):
            return [os.fspath(img_paths)]

        # Some pipelines may store paths in numpy arrays.
        if isinstance(img_paths, np.ndarray):
            img_paths = img_paths.tolist()

        if isinstance(img_paths, SequenceABC):
            flattened = []
            for item in img_paths:
                flattened.extend(cls._flatten_img_paths(item))
            return flattened

        raise TypeError(
            '`img_path` must be a path-like object or a sequence of '
            f'path-like objects, but got {type(img_paths)}: {img_paths}')

    def _get_test_out_file(self, runner: Runner, img_path: str,
                           window_name: str, mode: str) -> Optional[str]:
        """Build output file path when running test with --show-dir."""
        if mode != 'test' or self.test_out_dir is None:
            return None

        test_out_dir = self.test_out_dir
        if not osp.isabs(test_out_dir):
            test_out_dir = osp.join(runner.work_dir, test_out_dir)

        mkdir_or_exist(test_out_dir)

        # Keep png output because visualization result is an image canvas.
        return osp.join(test_out_dir, f'{window_name}.png')

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
            img_paths = self._flatten_img_paths(output.img_path)
            if len(img_paths) == 0:
                warnings.warn('Skip visualization because `output.img_path` '
                              'is empty.')
                continue

            # Use the first temporal image path only for naming / shape lookup.
            # For change detection this is usually the A/pre-event image.
            img_path = img_paths[0]
            window_name = osp.splitext(osp.basename(img_path))[0]

            img_from_to = []
            if self.img_shape is not None:
                assert len(self.img_shape) == 3, \
                    '`img_shape` should be (H, W, C)'
            else:
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                self.img_shape = img.shape

            if self.draw_on_from_to_img:
                # For semantic change detection, draw on the original
                # from/to images. Only the first two flattened paths are used.
                assert len(img_paths) >= 2, (
                    '`draw_on_from_to_img=True` requires at least two '
                    f'image paths, but got {img_paths}')
                for _img_path in img_paths[:2]:
                    _img_bytes = fileio.get(
                        _img_path, backend_args=self.backend_args)
                    _img = mmcv.imfrombytes(
                        _img_bytes, channel_order='rgb')
                    img_from_to.append(_img)

            img = np.zeros(self.img_shape, dtype=np.uint8)
            out_file = self._get_test_out_file(
                runner, img_path, window_name, mode)

            self._visualizer.add_datasample(
                window_name,
                img,
                img_from_to,
                data_sample=output,
                show=self.show,
                wait_time=self.wait_time,
                step=runner.iter,
                draw_gt=False,
                out_file=out_file)
