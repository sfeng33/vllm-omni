from vllm_omni.entrypoints.openai.protocol.video import OpenAICreateVideoRequest, VideoJob
from vllm.logger import init_logger
from vllm.entrypoints.openai.engine.serving import OpenAIServing
import uuid
import time
import base64
from io import BytesIO
from PIL import Image
from typing import Any
from fastapi import Request, HTTPException
from http import HTTPStatus
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage, ChatCompletionResponse, ChatCompletionResponseChoice
from vllm.entrypoints.openai.engine.protocol import UsageInfo

logger = init_logger(__name__)


class OmniOpenAIServingVideo(OpenAIServing):
    @classmethod
    def create(cls) -> "OmniOpenAIServingVideo":
        """Create a video serving instance.

        Returns:
            OmniOpenAIServingVideo instance that retrieves dependencies from app state at runtime.
        """
        return cls.__new__(cls)  

    async def create_video(
        self,
        request: OpenAICreateVideoRequest,
        raw_request: Request,
    ):
        """Generate videos from text prompts using diffusion models.

        OpenAI-compatible endpoint for text-to-video generation.
        Supports multi-stage omni mode with video diffusion stages.

        Args:
            request: Video generation request with prompt and parameters
            raw_request: Raw FastAPI request for accessing app state

        Returns:
            JSONResponse with generated video data

        Raises:
            HTTPException: For validation errors, missing engine, or generation failures
        """
        engine_client = getattr(raw_request.app.state, "engine_client", None)
        if engine_client is None:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                detail="Engine not initialized. Start server with a video generation model.",
            )

        # Wan models (T2V, I2V, TI2V) are single-stage diffusion
        stage_types: list[str] = ["diffusion"]

        # Get server's loaded model name
        # serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
        # if serving_models and hasattr(serving_models, "base_model_paths") and serving_models.base_model_paths:
        #     model_name = serving_models.base_model_paths[0].name
        # else:
        #     model_name = "unknown"

        # # Validate model field (warn if mismatch, don't error)
        # if request.model is not None and request.model != model_name:
        #     logger.warning(
        #         f"Model mismatch: request specifies '{request.model}' but "
        #         f"server is running '{model_name}'. Using server model."
        #     )

        try:
            # Build params - pass through user values directly
            gen_params = {
                "prompt": request.prompt,
            }

            # height width

            # num

            # Add optional parameters ONLY if provided
            if request.num_frames is not None:
                gen_params["num_frames"] = request.num_frames
            if request.num_inference_steps is not None:
                gen_params["num_inference_steps"] = request.num_inference_steps
            if request.guidance_scale is not None:
                gen_params["guidance_scale"] = request.guidance_scale
            if request.guidance_scale_high is not None:
                gen_params["guidance_scale_high"] = request.guidance_scale_high
            if request.negative_prompt is not None:
                gen_params["negative_prompt"] = request.negative_prompt
            # if request.fps is not None:
            #     gen_params["fps"] = request.fps
            # if request.height is not None:
            #     gen_params["height"] = request.height
            # if request.width is not None:
            #     gen_params["width"] = request.width
            # if request.seconds is not None:
            #     gen_params["seconds"] = request.seconds
            
            gen_params["request_id"] = f"video_gen_{int(time.time())}"

            logger.info(f"Generating video with prompt: {request.prompt[:50]}...")

            # Generate video using AsyncOmni (multi-stage mode)
            result = None
            stage_list = getattr(engine_client, "stage_list", None)
            print("FLORA. stage_list", len(stage_list))
            if isinstance(stage_list, list):
                default_params_list = getattr(engine_client, "default_sampling_params_list", None)
                if not isinstance(default_params_list, list):
                    default_params_list = [{} for _ in stage_types]
                else:
                    default_params_list = list(default_params_list)
                if len(default_params_list) != len(stage_types):
                    default_params_list = (default_params_list + [{} for _ in stage_types])[: len(stage_types)]

                sampling_params_list: list[dict[str, Any]] = []
                for idx, stage_type in enumerate(stage_types):
                    if stage_type == "diffusion":
                        sampling_params_list.append(gen_params)
                    else:
                        base_params = default_params_list[idx]
                        sampling_params_list.append(dict(base_params) if isinstance(base_params, dict) else base_params)

                async for output in engine_client.generate(
                    prompt=gen_params["prompt"],
                    request_id=gen_params["request_id"],
                    sampling_params_list=sampling_params_list,
                ):
                    result = output
            else:
                result = await engine_client.generate(**gen_params)

            if result is None:
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    detail="No output generated from multi-stage pipeline.",
                )

            logger.info(f"Successfully generated video")
            images = getattr(result.request_output, "images", [])

            # Handle video (multiple frames)
            import numpy as np
            import torch
            from diffusers.utils import export_to_video
            import tempfile

            # Get frames - could be tensor or in images list
            frames = images[0] if images else None
            logger.info(f"[Flora DEBUG] frames type: {type(frames)}")

            # Determine output path early so it's available for response
            output_path = request.output_path if request.output_path else "./video.mp4"

            if frames is not None:
            # Process tensor like text_to_video.py
                if isinstance(frames, torch.Tensor):
                    video_tensor = frames.detach().cpu()
                    if video_tensor.dim() == 5:
                        # [B, C, F, H, W] or [B, F, H, W, C]
                        if video_tensor.shape[1] in (3, 4):
                            video_tensor = video_tensor[0].permute(1, 2, 3, 0)
                        else:
                            video_tensor = video_tensor[0]
                    elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
                        video_tensor = video_tensor.permute(1, 2, 3, 0)
                    # If float, assume [-1,1] and normalize to [0,1]
                    if video_tensor.is_floating_point():
                        video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
                    video_array = video_tensor.float().numpy()
                else:
                    video_array = frames
                    if hasattr(video_array, "shape") and video_array.ndim == 5:
                        video_array = video_array[0]

                # Convert 4D array to list for export_to_video
                if isinstance(video_array, np.ndarray) and video_array.ndim == 4:
                    video_array = list(video_array)

                # Export video to file
                fps = request.fps if request.fps is not None else 24
                export_to_video(video_array, str(output_path), fps=fps)

            # Determine size string
            if request.size:
                size_str = request.size
            elif request.width and request.height:
                size_str = f"{request.width}x{request.height}"
            else:
                size_str = "unknown"

            video_job = VideoJob(
                id=f"video-{uuid.uuid4().hex[:12]}",
                object="video",
                model=model_name,
                status="queued",
                progress=0,
                prompt=request.prompt,
                size=size_str,
                seconds=str(request.seconds) if request.seconds else "4",
                file_path=output_path if frames is not None else None,
            )

            return JSONResponse(content=video_job.model_dump())

        except HTTPException:
            # Re-raise HTTPExceptions as-is
            raise
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))
        except Exception as e:
            logger.exception(f"Video generation failed: {e}")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Video generation failed: {str(e)}"
            )