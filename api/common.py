from pydantic import BaseModel, Field
from typing import Optional, Any


class PredictionRequest(BaseModel):
    """
    This object specifies the request sent to the /completion endpoint.
    It includes the prompt, maximum response length, and sampling parameters for the LM.

    Attributes:
        prompt (str or list[str]): Input text or list of texts (batch) to the language model.
        max_tokens (int, optional): Maximum number of tokens to generate.
        parameters (dict, optional): Additional parameters for vLLM.
    """
    prompt: str | list[str]
    max_tokens: int = Field(
        default=400,
        ge=10,
        le=2048,
        description="Maximum number of tokens in model output."
    )
    parameters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional sampling parameters to override SamplingParams."
    )


class PredictionResponse(BaseModel):
    """
    This object specifies the response sent by the /completion endpoint.

    Attributes:
        response (str or list[str]): The generated text.
    """
    completion: str | list[str] = Field(..., description="Generated response(s) from LLM.")