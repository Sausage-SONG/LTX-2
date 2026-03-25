from dataclasses import dataclass, replace

import torch

from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.primitives import LoraStateDictWithStrength
from ltx_core.loader.registry import DummyRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import to_denoised


@dataclass(frozen=True)
class TransformerBlockShard:
    start: int
    end: int
    device: torch.device


class BlockShardedX0Model(torch.nn.Module):
    def __init__(
        self,
        velocity_model: LTXModel,
        block_shards: tuple[TransformerBlockShard, ...],
        io_device: torch.device,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.block_shards = block_shards
        self.io_device = io_device
        self.device = io_device

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: object | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        input_video = video
        input_audio = audio

        if video is not None and video.latent.device != self.io_device:
            video = _move_modality(video, self.io_device)
        if audio is not None and audio.latent.device != self.io_device:
            audio = _move_modality(audio, self.io_device)

        video_args = self.velocity_model.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.velocity_model.audio_args_preprocessor.prepare(audio, video) if audio is not None else None

        for shard in self.block_shards:
            if video_args is not None:
                video_args = _move_transformer_args(video_args, shard.device)
            if audio_args is not None:
                audio_args = _move_transformer_args(audio_args, shard.device)

            for block_idx in range(shard.start, shard.end):
                video_args, audio_args = self.velocity_model.transformer_blocks[block_idx](
                    video=video_args,
                    audio=audio_args,
                    perturbations=perturbations,
                )

        vx = self._project_output(
            transformer_args=video_args,
            scale_shift_table=getattr(self.velocity_model, "scale_shift_table", None),
            norm_out=getattr(self.velocity_model, "norm_out", None),
            proj_out=getattr(self.velocity_model, "proj_out", None),
        )
        ax = self._project_output(
            transformer_args=audio_args,
            scale_shift_table=getattr(self.velocity_model, "audio_scale_shift_table", None),
            norm_out=getattr(self.velocity_model, "audio_norm_out", None),
            proj_out=getattr(self.velocity_model, "audio_proj_out", None),
        )

        denoised_video = None
        denoised_audio = None

        if input_video is not None and vx is not None:
            if vx.device != input_video.latent.device:
                vx = vx.to(input_video.latent.device)
            denoised_video = to_denoised(input_video.latent, vx, input_video.timesteps)

        if input_audio is not None and ax is not None:
            if ax.device != input_audio.latent.device:
                ax = ax.to(input_audio.latent.device)
            denoised_audio = to_denoised(input_audio.latent, ax, input_audio.timesteps)

        return denoised_video, denoised_audio

    def _project_output(
        self,
        transformer_args: TransformerArgs | None,
        scale_shift_table: torch.Tensor | None,
        norm_out: torch.nn.LayerNorm | None,
        proj_out: torch.nn.Linear | None,
    ) -> torch.Tensor | None:
        if transformer_args is None or scale_shift_table is None or norm_out is None or proj_out is None:
            return None

        x = transformer_args.x
        embedded_timestep = transformer_args.embedded_timestep
        if x.device != self.io_device:
            x = x.to(self.io_device)
        if embedded_timestep.device != self.io_device:
            embedded_timestep = embedded_timestep.to(self.io_device)

        return self.velocity_model._process_output(
            scale_shift_table,
            norm_out,
            proj_out,
            x,
            embedded_timestep,
        )


@dataclass(frozen=True)
class BlockShardedTransformerBuilder:
    base_builder: SingleGPUModelBuilder[LTXModel]
    devices: tuple[torch.device, ...]
    block_boundaries: tuple[int, ...] | None = None
    io_device: torch.device | None = None

    def build(self, dtype: torch.dtype | None = None) -> BlockShardedX0Model:
        config = self.base_builder.model_config()
        model = self.base_builder.meta_model(config, self.base_builder.module_ops)
        if not isinstance(model, LTXModel):
            raise TypeError(
                f"BlockShardedTransformerBuilder only supports LTXModel, got {type(model)!r}"
            )
        if dtype is not None:
            model = model.to(dtype=dtype)

        shard_devices = tuple(torch.device(device) for device in self.devices)
        io_device = torch.device(self.io_device) if self.io_device is not None else shard_devices[0]
        block_shards = _resolve_block_shards(
            num_blocks=len(model.transformer_blocks),
            devices=shard_devices,
            block_boundaries=self.block_boundaries,
        )
        _allocate_sharded_ltx_model(model, block_shards=block_shards, io_device=io_device)

        model_paths = (
            list(self.base_builder.model_path)
            if isinstance(self.base_builder.model_path, tuple)
            else [self.base_builder.model_path]
        )
        model_state_dict = self.base_builder.load_sd(
            model_paths,
            sd_ops=self.base_builder.model_sd_ops,
            registry=DummyRegistry(),
            device=torch.device("cpu"),
        )

        lora_strengths = [lora.strength for lora in self.base_builder.loras]
        if lora_strengths and not (min(lora_strengths) == 0 and max(lora_strengths) == 0):
            lora_state_dicts = [
                self.base_builder.load_sd(
                    [lora.path],
                    sd_ops=lora.sd_ops,
                    registry=DummyRegistry(),
                    device=self.base_builder.lora_load_device,
                )
                for lora in self.base_builder.loras
            ]
            lora_sd_and_strengths = [
                LoraStateDictWithStrength(sd, strength)
                for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
            ]
            model_state_dict = apply_loras(
                model_sd=model_state_dict,
                lora_sd_and_strengths=lora_sd_and_strengths,
                dtype=dtype,
                destination_sd=model_state_dict,
            )

        model.load_state_dict(model_state_dict.sd, strict=False, assign=False)
        _assert_no_meta_tensors(model)
        return BlockShardedX0Model(
            velocity_model=model,
            block_shards=block_shards,
            io_device=io_device,
        )


def _resolve_block_shards(
    num_blocks: int,
    devices: tuple[torch.device, ...],
    block_boundaries: tuple[int, ...] | None,
) -> tuple[TransformerBlockShard, ...]:
    if len(devices) < 2:
        raise ValueError("Block sharding requires at least two devices.")

    if block_boundaries is None:
        block_boundaries = tuple((num_blocks * idx) // len(devices) for idx in range(1, len(devices)))

    if len(block_boundaries) != len(devices) - 1:
        raise ValueError(
            f"Expected {len(devices) - 1} block boundary values for {len(devices)} devices, "
            f"got {len(block_boundaries)}"
        )

    boundaries = (0, *block_boundaries, num_blocks)
    block_shards: list[TransformerBlockShard] = []
    for start, end, device in zip(boundaries[:-1], boundaries[1:], devices, strict=True):
        if not (0 <= start < end <= num_blocks):
            raise ValueError(
                f"Invalid sharding boundary range ({start}, {end}) for {num_blocks} blocks."
            )
        block_shards.append(TransformerBlockShard(start=start, end=end, device=device))

    return tuple(block_shards)


def _allocate_sharded_ltx_model(
    model: LTXModel,
    *,
    block_shards: tuple[TransformerBlockShard, ...],
    io_device: torch.device,
) -> None:
    for _, child in list(model.named_children()):
        if child is model.transformer_blocks:
            continue
        child.to_empty(device=io_device)

    for name, parameter in list(model.named_parameters(recurse=False)):
        setattr(
            model,
            name,
            torch.nn.Parameter(
                torch.empty_like(parameter, device=io_device),
                requires_grad=parameter.requires_grad,
            ),
        )

    for name, buffer in list(model.named_buffers(recurse=False)):
        model.register_buffer(
            name,
            torch.empty_like(buffer, device=io_device),
            persistent=name not in model._non_persistent_buffers_set,
        )

    for shard in block_shards:
        for block_idx in range(shard.start, shard.end):
            model.transformer_blocks[block_idx].to_empty(device=shard.device)


def _move_modality(modality: Modality, device: torch.device) -> Modality:
    return replace(
        modality,
        latent=modality.latent.to(device),
        sigma=modality.sigma.to(device),
        timesteps=modality.timesteps.to(device),
        positions=modality.positions.to(device),
        context=modality.context.to(device),
        context_mask=_move_optional_tensor(modality.context_mask, device),
        attention_mask=_move_optional_tensor(modality.attention_mask, device),
    )


def _move_transformer_args(args: TransformerArgs, device: torch.device) -> TransformerArgs:
    return replace(
        args,
        x=args.x.to(device),
        context=args.context.to(device),
        context_mask=_move_optional_tensor(args.context_mask, device),
        timesteps=args.timesteps.to(device),
        embedded_timestep=args.embedded_timestep.to(device),
        positional_embeddings=_move_tensor_structure(args.positional_embeddings, device),
        cross_positional_embeddings=_move_tensor_structure(args.cross_positional_embeddings, device),
        cross_scale_shift_timestep=_move_optional_tensor(args.cross_scale_shift_timestep, device),
        cross_gate_timestep=_move_optional_tensor(args.cross_gate_timestep, device),
        prompt_timestep=_move_optional_tensor(args.prompt_timestep, device),
        self_attention_mask=_move_optional_tensor(args.self_attention_mask, device),
    )


def _move_optional_tensor(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.device == device:
        return tensor
    return tensor.to(device)


def _move_tensor_structure(value: object, device: torch.device) -> object:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.device == device:
            return value
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tensor_structure(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensor_structure(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tensor_structure(item, device) for key, item in value.items()}
    return value


def _assert_no_meta_tensors(model: torch.nn.Module) -> None:
    uninitialized_params = [name for name, param in model.named_parameters() if str(param.device) == "meta"]
    uninitialized_buffers = [name for name, buffer in model.named_buffers() if str(buffer.device) == "meta"]
    if uninitialized_params or uninitialized_buffers:
        raise RuntimeError(
            "Uninitialized parameters or buffers remain on meta device: "
            f"{uninitialized_params + uninitialized_buffers}"
        )
