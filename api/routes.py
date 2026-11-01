from fastapi import APIRouter
from api.schemas import RagPlaygroundRequest, LlmPlaygroundRequest
from api.comparison_runner import run_rag_playground, run_llm_playground

router = APIRouter()


@router.post("/rag-playground")
def rag_playground(req: RagPlaygroundRequest):
    results = run_rag_playground(req.question, req.model, req.pipelines)
    return {"results": results}


@router.post("/llm-playground")
def llm_playground(req: LlmPlaygroundRequest):
    results = run_llm_playground(req.question, req.pipeline, req.models)
    return {"results": results}