import os, logging

from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from fastapi import FastAPI

# interesting that we implement API schema as an object
from .common import PredictionResponse


# >>>===<<< Create an LLM Model >>>===<<<
load_dotenv()
# NOTE: better as a YAML file
model = os.environ.get("MODEL_NAME")
model_name = model.split("/")[-1]
model = LLM(
    model=model_name,
    tensor_parallel_size=1,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
)

# >>>===<<< Create a FastAPI App >>>===<<<
logger = logging.getLogger(__name__)
app = FastAPI()


# >>>ENDPOINT<<< / >>>ENDPOINT<<<
@app.get("/")
async def health():
    return {"Status": "Healthy", "Model": model_name}


# >>>ENDPOINT<<< /completion >>>ENDPOINT<<<
@app.post(f"/completion/", response_model=PredictionResponse)
async def inference(
    request: PredictionRequest = Depends(parse_prediction_request),
    ) -> PredictionResponse:
        
    if isinstance(request.prompt, str):
        # request can either be one prompt
        queries = [request.prompt]
    elif isinstance(request.prompt, list):
        # can also be a list of prompts
        queries = request.prompt
        if not queries:
            raise HTTPException(
                status_code=400,
                detail="List of prompts is empty"
            )
    else:        
        raise HTTPException(
            status_code=400,
            detail=f"""Invalid prompt format. 
            Should be str or list; current format: {type(request.prompt)}"""
        )

    default_params = {
        "temperature": 0.8,
        "repetition_penalty": 1.0,
        "top_p": 1.0,
    }
    if request.parameters:
        # if parameters field is populated
        default_params.update(request.parameters)
    
    if "max_tokens" not in default_params:
        default_params["max_tokens"] = request.max_tokens
    sampling_params = SamplingParams(**default_params)