# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Any, Dict
from pathlib import Path
import os, sys, re, json

# =========================
# ENV & UTIL
# =========================
load_dotenv()

def env(name: str, required: bool = False, default: str = "") -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing env: {name}")
    return val

TEST_MODE        = os.getenv("CHATBEATS_TEST_MODE", "1") == "1"
MISTRAL_API_KEY  = env("MISTRAL_API_KEY",  required=False)
HF_TOKEN         = env("HF_TOKEN",         required=False)
TAVILY_API_KEY   = env("TAVILY_API_KEY",   required=False)
WEAVIATE_URL     = env("WEAVIATE_URL",     required=False)
WEAVIATE_API_KEY = env("WEAVIATE_API_KEY", required=False)

# =========================
# PATH VERS TES AGENTS (RAG)
# =========================
root = Path(__file__).resolve().parent
candidates = [
    root.parent / "Get-Hired-Project",
    root / "Get-Hired-Project",
    root.parent.parent / "Get-Hired-Project",
]
for p in candidates:
    if (p / "agentic_scorer.py").exists():
        sys.path.insert(0, str(p))
        print("✅ sys.path +", p)
        break
else:
    print("⚠️ Get-Hired-Project introuvable près de", root)

# =========================
# IMPORTS AGENTS (ROBUSTES)
# =========================
AGENTS_OK = True
try:
    from agentic_scorer import create_dual_llm_scorer, ScoringDecision
    from agentic_research import create_research_agent, ResearchReport
except Exception as e:
    print("❌ Import agents failed:", e)
    AGENTS_OK = False
    create_dual_llm_scorer = None
    ScoringDecision = None
    create_research_agent = None
    ResearchReport = None

# =========================
# FASTAPI
# =========================
app = FastAPI(title="ChatBeats API (RAG)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200","http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# STATE GLOBAL
# =========================
scorer = None
research_agent = None
vectorstore = None
weaviate_client = None
mistral_llm = None

# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    sessionId: str
    message: str

class ChatResponse(BaseModel):
    role: str = "assistant"
    type: str  # candidate_screening | market_research | text | error
    content: Any

class _Resp(BaseModel):
    role: str = "assistant"
    type: str
    content: Any

# =========================
# STARTUP / SHUTDOWN
# =========================
@app.on_event("startup")
async def on_start():
    """
    Initialise tous les composants. En TEST_MODE, on passe en mode léger.
    """
    global scorer, research_agent, vectorstore, weaviate_client, mistral_llm

    if TEST_MODE:
        print("🧪 TEST_MODE=1 → skip heavy init")
        return

    if not AGENTS_OK:
        print("⚠️ Agents not importable → TEST fallback only")
        return

    # 1) Router LLM (Mistral Large) — optionnel
    try:
        from langchain_mistralai import ChatMistralAI
        mistral_llm = ChatMistralAI(
            api_key=MISTRAL_API_KEY,
            model="mistral-large-latest",
            temperature=0.1,
            max_tokens=1024,
        )
        print("✅ Mistral router ready")
    except Exception as e:
        print("⚠️ Router init failed:", e)

    # 2) Weaviate + Embeddings + VectorStore
    try:
        import weaviate
        from weaviate.classes.init import Auth
        # Astuce: privilégier connect_to_weaviate_cloud pour éviter le warning
        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
        )

        if not weaviate_client.is_ready():
            raise RuntimeError("Weaviate not ready")

        # Embeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # VectorStore
        from langchain_weaviate.vectorstores import WeaviateVectorStore
        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name="Resume",               # ⚠️ adapte si ta classe a un autre nom
            text_key="preprocessed_text",      # ⚠️ adapte si ton champ texte a un autre nom
            embedding=embeddings,
        )
        print("✅ Vector store ready")
    except Exception as e:
        print("⚠️ Vector/Weaviate init failed:", e)

    # 3) Scorer (Llama-70B via HF)
    try:
        scorer = create_dual_llm_scorer(
            huggingface_api_key=HF_TOKEN,
            temperature=0.2
        )
        print("✅ Scorer ready")
    except Exception as e:
        print("⚠️ Scorer init failed:", e)

    # 4) Research Agent (Mistral-7B + Tavily) - CORRIGÉ ICI
    try:
        research_agent = create_research_agent(
            mistral_api_key=MISTRAL_API_KEY,  # ✅ Changé de huggingface_api_key
            tavily_api_key=TAVILY_API_KEY,
            model="open-mistral-7b"  # ✅ Changé le modèle pour l'API Mistral
        )
        print("✅ Research agent ready")
    except Exception as e:
        print("⚠️ Research init failed:", e)

@app.on_event("shutdown")
async def on_stop():
    global weaviate_client
    try:
        if weaviate_client:
            weaviate_client.close()
            print("✅ Weaviate closed")
    except:
        pass

# =========================
# ROUTAGE
# =========================
def simple_route(q: str) -> str:
    """Routage local par mots-clés (fallback)."""
    ql = q.lower()
    if any(k in ql for k in [
        "salary","salaire","compensation","market","tendance","coût","budget","wage","hiring","trend","trends"
    ]):
        return "market_research"
    return "candidate_screening"

def mistral_route(user_msg: str) -> Dict[str, Any]:
    """Routage via Mistral-Large (retourne JSON) avec fallback JSON local."""
    if not mistral_llm:
        return {"tool": simple_route(user_msg), "args": {"query": user_msg}}

    prompt = f"""
Analyze the user's query and decide the tool:
- candidate_screening: find/search/evaluate candidates or resumes
- market_research: salary/compensation/market trends/hiring costs

User: {user_msg}

Return ONLY JSON:
{{{{"tool":"candidate_screening"|"market_research","args":{{"query":"..."}}}}}}
"""
    try:
        res = mistral_llm.invoke(prompt)
        content = getattr(res, "content", str(res))
        m = re.search(r'\{[^{}]*"tool"[^{}]*\}', content, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return json.loads(content)  # si le modèle a renvoyé du JSON pur
    except Exception:
        return {"tool": simple_route(user_msg), "args": {"query": user_msg}}

# =========================
# TOOLS
# =========================
def run_candidate_screening(args: Dict[str, Any]) -> Dict[str, Any]:
    """Recherche + scoring candidats avec gestion d'erreurs intégrée."""
    if not scorer or not vectorstore:
        raise HTTPException(status_code=503, detail="Scorer/Vectorstore not initialized")

    q = args.get("query", "")
    try:
        # 1) Exigences
        reqs = scorer.extract_requirements(q)

        # 2) Recherche vectorielle
        docs_scores = vectorstore.similarity_search_with_score(q, k=10)
        if not docs_scores:
            return {
                "type": "candidate_screening",
                "content": {"message": "Aucun candidat trouvé.", "candidates": [], "total_found": 0}
            }

        # 3) Évaluation LLM
        decisions = []
        for doc, score in docs_scores:
            dec = scorer.evaluate_candidate(
                candidate_doc=doc,
                semantic_score=score,
                requirements=reqs
            )
            decisions.append(dec)

        decisions.sort(key=lambda x: x.final_score, reverse=True)

        top = [{
            "name": d.candidate_name,
            "score": round(d.final_score, 2),
            "confidence": d.confidence,
            "strengths": d.strengths[:3],
            "concerns": d.concerns[:3],
            "recommendation": (d.recommendation or "")[:200]
        } for d in decisions[:5]]

        return {
            "type": "candidate_screening",
            "content": {
                "message": f"Top {len(top)} résultat(s)",
                "candidates": top,
                "total_found": len(decisions)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("❌ candidate_screening error:", e)
        print(traceback.format_exc())
        return {
            "type": "error",
            "content": {"message": f"Erreur screening: {str(e)[:300]}"}
        }

def run_market_research(args: Dict[str, Any]) -> Dict[str, Any]:
    """Recherche de marché avec gestion d'erreurs intégrée."""
    if not research_agent:
        raise HTTPException(status_code=503, detail="Research agent not initialized")

    q = args.get("query", "")
    try:
        report: ResearchReport = research_agent.conduct_research(q)
        return {
            "type": "market_research",
            "content": {
                "salary_overview": (report.salary_overview or "")[:1000],
                "market_insights": (report.market_insights or "")[:1000],
                "hiring_recommendations": (report.hiring_recommendations or "")[:1000],
                "sources": report.sources[:5] if report.sources else []
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("❌ market_research error:", e)
        print(traceback.format_exc())
        return {
            "type": "error",
            "content": {"message": f"Erreur market: {str(e)[:300]}"}
        }

# =========================
# API
# =========================
@app.get("/")
def root():
    return {"ok": True, "health": "/healthz", "chat": "/chat"}

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "test_mode": TEST_MODE,
        "agents_loaded": bool(scorer) and bool(research_agent)
    }

@app.get("/debug")
def debug():
    return {
        "mistral_llm": bool(mistral_llm),
        "weaviate": bool(weaviate_client),
        "vectorstore": bool(vectorstore),
        "scorer": bool(scorer),
        "research_agent": bool(research_agent),
        "test_mode": TEST_MODE,
        "agents_ok": AGENTS_OK,
    }

@app.post("/test/market")
def test_market(payload: Dict[str, Any]):
    try:
        q = payload.get("q", "")
        return run_market_research({"query": q})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return {"type": "error", "content": {"message": f"market error: {str(e)[:300]}"}}

@app.post("/test/candidate")
def test_candidate(payload: Dict[str, Any]):
    try:
        q = payload.get("q", "")
        return run_candidate_screening({"query": q})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return {"type": "error", "content": {"message": f"candidate error: {str(e)[:300]}"}}

@app.post("/chat", response_model=_Resp)
def chat(req: ChatRequest):
    """
    Route principal — ne renvoie JAMAIS un 500 brut :
    renvoie toujours un JSON {type, content} et loggue l'exception côté serveur.
    """
    print(f"\n📥 Received: {req.sessionId} | {req.message[:120]}")
    try:
        # Mode test = court-circuit
        if TEST_MODE or not AGENTS_OK:
            return _Resp(type="text", content={"message": f"Réponse test pour: {req.message}"})

        # Choix du routeur (Mistral si prêt, sinon fallback local)
        use_mistral_router = bool(MISTRAL_API_KEY) and bool(mistral_llm)
        tool_args = (
            mistral_route(req.message)
            if use_mistral_router else
            {"tool": simple_route(req.message), "args": {"query": req.message}}
        )

        tool = tool_args.get("tool", "candidate_screening")
        args = tool_args.get("args", {"query": req.message})
        print("🔀 Router:", {"use_mistral": use_mistral_router, "tool": tool, "args": args})

        # Appel outil
        if tool == "market_research":
            res = run_market_research(args)
        else:
            res = run_candidate_screening(args)

        return _Resp(type=res.get("type", "text"), content=res.get("content", {}))

    except HTTPException as he:
        print(f"⚠️ HTTPException: {he.status_code} {he.detail}")
        return _Resp(type="error", content={"message": f"HTTP {he.status_code}: {he.detail}"})
    except Exception as e:
        import traceback
        print("❌ /chat error:", e)
        print(traceback.format_exc())
        return _Resp(type="error", content={"message": f"Erreur serveur: {str(e)[:300]}"})

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )