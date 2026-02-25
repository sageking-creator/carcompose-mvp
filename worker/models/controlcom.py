from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

from PIL import Image

from exceptions import ControlComSetupError, ModelInferenceError


class ControlComHarmonizer:
    def __init__(
        self,
        *,
        repo_dir: Path,
        ckpt_path: Path,
        clip_dir: Path,
        timeout_s: int = 600,
    ):
        self.repo_dir = Path(repo_dir)
        self.ckpt_path = Path(ckpt_path)
        self.clip_dir = Path(clip_dir)
        self.timeout_s = int(timeout_s)

        self.script_path = self.repo_dir / "scripts" / "inference.py"

        if not self.script_path.exists():
            raise ControlComSetupError(f"ControlCom script not found: {self.script_path}")

        if not self.ckpt_path.exists() or self.ckpt_path.stat().st_size < 1_000_000:
            raise ControlComSetupError(
                f"ControlCom checkpoint missing or too small: {self.ckpt_path}\n"
                "Expected: ControlCom_blend_harm.pth (downloaded via HuggingFace; gdown fallback)."
            )

        if not self.clip_dir.exists() or not any(self.clip_dir.iterdir()):
            raise ControlComSetupError(
                f"CLIP model dir missing or empty: {self.clip_dir}\n"
                "Expected: openai/clip-vit-large-patch14 (downloaded from HuggingFace)."
            )

    def harmonize(
        self,
        *,
        background_image: Image.Image,
        fg_crop: Image.Image,
        fg_mask_crop: Image.Image,
        placement_bbox: Tuple[int, int, int, int],
        sample_steps: int = 50,
    ) -> Image.Image:
        """
        Runs the upstream ControlCom inference script as-is.

        The vendor script expects a test directory layout:
          testdir/background/<name>.(png/jpg)
          testdir/foreground/<name>.(png/jpg)
          testdir/bbox/<name>.txt              (x1 y1 x2 y2)
          testdir/foreground_mask/<name>.(png) (optional)

        And a checkpoint directory containing:
          ControlCom_blend_harm.pth
          openai-clip-vit-large-patch14/
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            test_dir = tmp / "test"
            bg_dir = test_dir / "background"
            fg_dir = test_dir / "foreground"
            bbox_dir = test_dir / "bbox"
            mk_dir = test_dir / "foreground_mask"
            out_dir = tmp / "output"
            for d in (bg_dir, fg_dir, bbox_dir, mk_dir, out_dir):
                d.mkdir(parents=True, exist_ok=True)

            sample_name = "sample.png"
            bg_path = bg_dir / sample_name
            fg_path = fg_dir / sample_name
            mk_path = mk_dir / sample_name
            bbox_path = bbox_dir / "sample.txt"

            background_image.convert("RGB").save(bg_path)
            fg_crop.convert("RGB").save(fg_path)
            fg_mask_crop.convert("L").save(mk_path)

            x1, y1, x2, y2 = placement_bbox
            bbox_path.write_text(f"{int(x1)} {int(y1)} {int(x2)} {int(y2)}\n", encoding="utf-8")

            ckpt_dir = self.ckpt_path.parent
            expected_clip_dir = ckpt_dir / "openai-clip-vit-large-patch14"
            if expected_clip_dir.resolve() != self.clip_dir.resolve():
                raise ControlComSetupError(
                    "ControlCom expects CLIP weights under ckpt_dir/openai-clip-vit-large-patch14.\n"
                    f"ckpt_dir={ckpt_dir}\n"
                    f"clip_dir={self.clip_dir}\n"
                    "Set CLIP_MODEL_DIR accordingly or store the model in the expected path."
                )

            env = os.environ.copy()
            existing_py_path = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{self.repo_dir}:{existing_py_path}" if existing_py_path else str(self.repo_dir)

            cmd = [
                sys.executable,
                str(self.script_path),
                "--testdir",
                str(test_dir),
                "--outdir",
                str(out_dir),
                "--skip_grid",
                "--task",
                "harmonization",
                "--ckpt_dir",
                str(ckpt_dir),
                "--num_samples",
                "1",
                "--sample_steps",
                str(int(sample_steps)),
                "--gpu",
                "0",
            ]

            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=str(self.repo_dir),
                timeout=self.timeout_s,
                check=False,
            )
            if result.returncode != 0:
                raise ModelInferenceError(
                    "ControlCom",
                    RuntimeError(
                        f"Exit code {result.returncode}. "
                        f"STDOUT(last 2k)={result.stdout[-2000:]} "
                        f"STDERR(last 2k)={result.stderr[-2000:]}"
                    ),
                )

            outputs = list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.jpeg"))
            if not outputs:
                raise ModelInferenceError("ControlCom", RuntimeError("No output files produced."))

            return Image.open(outputs[0]).convert("RGB")
