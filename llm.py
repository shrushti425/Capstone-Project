import os
from config import LLM_MODEL

def get_llm(model_name: str = None):
    if os.getenv("USE_LLAMACPP"):
        from langchain_community.llms import LlamaCpp
        return LlamaCpp(
            model_path=os.getenv("MODEL_PATH", "./models/model.gguf"),
            n_gpu_layers=int(os.getenv("N_GPU_LAYERS", 20)),
            n_ctx=2048,
            temperature=0.1,
            verbose=False
        )
    else:
        from langchain_ollama import OllamaLLM
        model = model_name or LLM_MODEL
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return OllamaLLM(
            model=model,
            base_url=base_url,
            temperature=0.1,
            num_ctx=4096,
        )

def generate_response(prompt: str, model_name: str = None) -> str:
    llm = get_llm(model_name)
    response = llm.invoke(prompt)
    return response