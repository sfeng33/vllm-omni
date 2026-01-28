import time
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator


class OpenAICreateVideoRequest(BaseModel):
    """
    OpenAI compatible video generation request.

    Follows the OpenAI Video API specification with vllm-omni extensions
    for advanced diffusion parameters.
    """
    
    prompt: str
    """Text prompt that describes the video to generate."""
    
    character_ids: Optional[list[str]] = None
    """Character IDs to include in the generation."""
    
    input_reference: Optional[bytes] = None
    """Optional image reference that guides generation."""
    
    model: Optional[str] = None
    """The video generation model to use."""
    
    seconds: Optional[int] = None
    """Clip duration in seconds."""
    
    size: Optional[str] = None
    """Output resolution formatted as width x height."""
    
    # vLLM-Omni extensions for diffusion control
    
    num_frames: Optional[int] = None
    """Number of frames to generate in the output video."""
    
    num_inference_steps: Optional[int] = None
    """Number of denoising steps for the diffusion process. Higher values improve quality but increase generation time."""
    
    guidance_scale: Optional[float] = None
    """Classifier-free guidance scale. Higher values increase prompt adherence at the cost of diversity."""
    
    guidance_scale_high: Optional[float] = None
    """Upper bound for guidance scale when using dynamic guidance."""
    
    negative_prompt: Optional[str] = None
    """Text prompt describing elements to avoid in the generated video."""
    
    fps: Optional[int] = None
    """Frames per second for the output video."""
    
    output_path: Optional[str] = None
    """File path where the generated video will be saved."""
    
    height: Optional[int] = None
    """Output video height in pixels. Overrides the height from 'size' if set."""
    
    width: Optional[int] = None
    """Output video width in pixels. Overrides the width from 'size' if set."""    


class VideoJobError(BaseModel):
    """Error details when video generation fails."""
    
    code: Optional[str] = None
    """Error code identifying the type of failure."""
    
    message: Optional[str] = None
    """Human-readable description of the error."""


class VideoJob(BaseModel):
    """Structured information describing a generated video job."""
    
    id: str
    """Unique identifier for the video job."""
    
    object: Literal["video"] = "video"
    """The object type, which is always 'video'."""
    
    model: str
    """The video generation model that produced the job."""
    
    status: Literal["queued", "in_progress", "completed", "failed"]
    """Current lifecycle status of the video job."""
    
    progress: int
    """Approximate completion percentage for the generation task (0-100)."""
    
    created_at: int = Field(default_factory=lambda: int(time.time()))
    """Unix timestamp (seconds) for when the job was created."""
    
    prompt: str
    """The prompt that was used to generate the video."""
    
    size: str
    """The resolution of the generated video (e.g., '1280x720')."""
    
    seconds: int
    """Duration of the generated clip in seconds."""

    file_path: Optional[str] = None
    """Local file path where the generated video is stored, if available."""
    
    completed_at: Optional[int] = None
    """Unix timestamp (seconds) for when the job completed, if finished."""
    
    error: Optional[VideoJobError] = None
    """Error payload that explains why generation failed, if applicable."""
    
    expires_at: Optional[int] = None
    """Unix timestamp (seconds) for when the downloadable assets expire, if set."""
    
    remixed_from_video_id: Optional[str] = None
    """Identifier of the source video if this video is a remix."""