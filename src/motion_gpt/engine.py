"""Inference-only use of pinned upstream MotionGPT modules and feature utilities.

The upstream source is untyped. Dynamic types are confined to this boundary;
public operations accept and return concrete NumPy arrays and Python scalars.
"""

import re
from pathlib import Path
from typing import Any, cast

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from src.assets import resolve_file
from src.motion_gpt.config import MotionGPTConfig
from src.motion_gpt.geometry import Positions
from src.motion_gpt.quaternion import quaternion_between
from src.motion_gpt.source import import_source_module


class MotionGPTEngine:
    """One loaded FLAN-T5 motion model and VQ tokenizer/decoder; no training model."""

    def __init__(self, config: MotionGPTConfig) -> None:
        """Download required assets, import the pinned code and load strict weights.

        :param config: Cache locations and CPU/MPS selection.
        :raises ValueError: For incompatible source, statistics or checkpoints.
        """
        import torch

        source = config.source_path or Path(
            snapshot_download(
                config.source_repo,
                repo_type="space",
                revision=config.source_revision,
                allow_patterns=[
                    "mGPT/archs/**",
                    "mGPT/data/humanml/common/**",
                    "mGPT/data/humanml/scripts/**",
                    "mGPT/data/humanml/utils/paramUtil.py",
                ],
                cache_dir=config.cache_dir,
                local_files_only=config.local_files_only,
            )
        )
        source = source.resolve()
        if not (source / "mGPT" / "archs" / "mgpt_lm.py").is_file():
            raise ValueError("MotionGPT source must contain mGPT/archs/mgpt_lm.py")
        lm_module = import_source_module(source, "archs.mgpt_lm")
        vq_module = import_source_module(source, "archs.mgpt_vq")
        self._motion = import_source_module(source, "geometry.scripts.motion_process")
        self._params = import_source_module(source, "geometry.utils.paramUtil")
        # Upstream normalizes a zero quaternion for exactly opposing bone vectors.
        # Override only the isolated inference modules, never the downloaded files.
        vars(self._motion)["qbetween_np"] = quaternion_between
        vars(import_source_module(source, "geometry.common.skeleton"))["qbetween_np"] = (
            quaternion_between
        )
        text_path = config.text_model_path or Path(
            snapshot_download(
                config.text_model_repo,
                revision=config.text_model_revision,
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "model.safetensors",
                    "tokenizer*",
                    "spiece.model",
                    "special_tokens_map.json",
                ],
                cache_dir=config.cache_dir,
                local_files_only=config.local_files_only,
            )
        )
        self._mean = np.load(
            hf_hub_download(
                config.source_repo,
                "assets/meta/mean.npy",
                repo_type="space",
                revision=config.source_revision,
                cache_dir=config.cache_dir,
                local_files_only=config.local_files_only,
            ),
            allow_pickle=False,
        ).astype(np.float32)
        self._std = np.load(
            hf_hub_download(
                config.source_repo,
                "assets/meta/std.npy",
                repo_type="space",
                revision=config.source_revision,
                cache_dir=config.cache_dir,
                local_files_only=config.local_files_only,
            ),
            allow_pickle=False,
        ).astype(np.float32)
        if (
            self._mean.shape != (263,)
            or self._std.shape != (263,)
            or not np.isfinite(self._mean).all()
            or not np.isfinite(self._std).all()
            or np.any(self._std <= 0)
        ):
            raise ValueError(
                "Expected finite 263-feature MotionGPT statistics with positive standard deviations"
            )
        self.device = (
            "mps"
            if config.device == "auto" and torch.backends.mps.is_available()
            else config.device
        )
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        # These two constructors match the published base checkpoint architecture.
        self._lm: Any = lm_module.MLM(
            model_path=str(text_path),
            model_type="t5",
            stage="lm_instruct",
            motion_codebook_size=512,
        )
        self._vae: Any = vq_module.VQVae(
            nfeats=263,
            quantizer="ema_reset",
            code_num=512,
            code_dim=512,
            output_emb_width=512,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            norm=None,
            activation="relu",
        )
        checkpoint = config.checkpoint.model_copy(
            update={
                "cache_dir": config.cache_dir or config.checkpoint.cache_dir,
                "local_files_only": config.local_files_only or config.checkpoint.local_files_only,
            }
        )
        state = torch.load(resolve_file(checkpoint), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not isinstance(state.get("state_dict"), dict):
            raise TypeError("Expected the official MotionGPT state_dict checkpoint")
        weights = state["state_dict"]
        for prefix, module in (("lm.", self._lm), ("vae.", self._vae)):
            module.load_state_dict(
                {
                    key[len(prefix) :]: value
                    for key, value in weights.items()
                    if key.startswith(prefix)
                },
                strict=True,
            )
            module.eval().requires_grad_(False).to(self.device)
        del state, weights

    def infer(self, prompt: str, max_tokens: int, sample: bool) -> str:
        """Run the shared text backbone without silently truncating its input.

        :param prompt: MotionGPT instruction with optional motion tokens.
        :param max_tokens: Maximum generated token count.
        :param sample: Whether to sample motion instead of greedy captioning.
        :returns: Raw decoded text, retaining motion markers for validation.
        """
        import torch

        with torch.inference_mode():
            encoded = self._lm.tokenizer(prompt, return_tensors="pt", truncation=False)
            if encoded["input_ids"].shape[1] > 256:
                raise ValueError("MotionGPT instruction exceeds its 256-token input budget")
            output = self._lm.language_model.generate(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
                max_new_tokens=max_tokens,
                do_sample=sample,
                num_beams=1,
            )
            return cast(str, self._lm.tokenizer.batch_decode(output, skip_special_tokens=True)[0])

    def caption(self, positions: Positions) -> str:
        """Convert canonical positions into normalized HumanML3D features and caption.

        Input must already have HumanML3D-compatible body proportions. Feature
        extraction centers the root, floors the body and aligns initial heading;
        it does not perform camera skeleton mapping or infer missing joints.

        :param positions: Complete canonical 20 Hz poses of one person.
        :returns: A grounded motion caption from the instruction-tuned checkpoint.
        """
        import torch

        model_positions = positions.copy() * np.array([-1, 1, -1], dtype=np.float32)
        model_positions[:, :, 1] -= model_positions[:, :, 1].min()
        model_positions[:, :, (0, 2)] -= model_positions[0, 0, (0, 2)]
        across = (
            model_positions[0, 2]
            - model_positions[0, 1]
            + model_positions[0, 17]
            - model_positions[0, 16]
        )
        forward = np.cross(np.array([0, 1, 0]), across)
        if np.linalg.norm(forward[[0, 2]]) < 1e-6:
            raise ValueError("Cannot estimate heading from degenerate hips and shoulders")
        angle = np.arctan2(forward[0], forward[2])
        rotation = np.array(
            [[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]],
            dtype=np.float32,
        )
        model_positions = model_positions @ rotation.T
        with np.errstate(invalid="ignore", divide="ignore"):
            features = np.asarray(
                self._motion.extract_features(
                    model_positions,
                    0.002,
                    torch.from_numpy(self._params.t2m_raw_offsets),
                    self._params.t2m_kinematic_chain,
                    [2, 1, 17, 16],
                    [8, 11],
                    [7, 10],
                ),
                dtype=np.float32,
            )
        if features.shape != (len(positions) - 1, 263) or not np.isfinite(features).all():
            raise ValueError("Joint geometry produced invalid HumanML3D features")
        # VQ encoder downsamples by four; discard at most three trailing feature rows.
        features = features[: len(features) // 4 * 4]
        if len(features) < 4:
            raise ValueError("At least four feature frames are needed for the motion tokenizer")
        with torch.inference_mode():
            tokens, _ = self._vae.encode(
                torch.from_numpy((features - self._mean) / self._std).unsqueeze(0).to(self.device)
            )
            values = tokens[0].detach().cpu().tolist()
        motion = (
            "<motion_id_512>"
            + "".join(f"<motion_id_{int(value)}>" for value in values)
            + "<motion_id_513>"
        )
        caption = self.infer("Generate text: " + motion, max_tokens=64, sample=False).strip()
        if not caption or "<motion_id_" in caption:
            raise ValueError("MotionGPT returned no usable motion caption")
        return caption

    def generate(self, prompt: str, frame_count: int, seed: int | None) -> Positions:
        """Generate a full native-rate clip and decode it into canonical positions.

        :param prompt: Complete action description.
        :param frame_count: Desired duration in native 20 Hz frames (4–196).
        :param seed: Optional request seed; backend operations are serialized.
        :returns: Decoded canonical positions; final duration is adapted by stage 6.
        """
        import torch

        if not 4 <= frame_count <= 196:
            raise ValueError("MotionGPT supports requested durations of 0.2 to 9.8 seconds")
        if seed is not None:
            torch.manual_seed(seed)
        raw = self.infer(
            f"Generate motion with {frame_count} frames: {prompt}", max_tokens=128, sample=True
        )
        match = re.search(r"<motion_id_512>(.*?)<motion_id_513>", raw, re.DOTALL)
        if match is None:
            raise ValueError("MotionGPT returned no complete motion token sequence")
        token_text = match.group(1)
        values = [int(value) for value in re.findall(r"<motion_id_(\d+)>", token_text)]
        if (
            not values
            or len(values) > 49
            or any(value >= 512 for value in values)
            or re.sub(r"<motion_id_\d+>", "", token_text).strip()
        ):
            raise ValueError("MotionGPT returned invalid or excessive motion tokens")
        with torch.inference_mode():
            tokens = torch.tensor(values, dtype=torch.long, device=self.device)
            features = self._vae.decode(tokens).detach().cpu()
            features = features * torch.from_numpy(self._std) + torch.from_numpy(self._mean)
            joints = self._motion.recover_from_ric(features, 22)[0].numpy().astype(np.float32)
        if joints.ndim != 3 or joints.shape[1:] != (22, 3) or not np.isfinite(joints).all():
            raise ValueError("MotionGPT decoded invalid joint positions")
        joints *= np.array([-1, 1, -1], dtype=np.float32)
        # Canonical origin is the initial pelvis, rather than the dataset's floor.
        joints -= joints[0, 0].copy()
        return cast(Positions, joints)
