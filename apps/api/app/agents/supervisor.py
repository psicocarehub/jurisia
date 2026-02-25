"""
Supervisor agent: routes to research, drafting, analysis, memory, or chat.
Uses trained intent classifier when available, LLM fallback otherwise.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_INTENT_DIR = Path(__file__).resolve().parents[4] / "training" / "models" / "intent"
_intent_model = None
_intent_loaded = False
_intent_labels: list[str] = []


def _load_intent_model():
    global _intent_model, _intent_loaded, _intent_labels
    if _intent_loaded:
        return _intent_model
    _intent_loaded = True
    model_path = _INTENT_DIR / "intent_classifier.joblib"
    config_path = _INTENT_DIR / "intent_classifier_config.json"
    if model_path.exists():
        try:
            import joblib
            _intent_model = joblib.load(model_path)
            if config_path.exists():
                _intent_labels = json.loads(config_path.read_text())["labels"]
            logger.info("Loaded trained intent classifier")
        except Exception as e:
            logger.warning("Failed to load intent classifier: %s", e)
    return _intent_model


async def supervisor_node(state: dict) -> dict:
    """Route to the appropriate agent based on user intent."""
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""

    model = _load_intent_model()
    if model is not None and _intent_labels:
        try:
            pred = model.predict([last_message])[0]
            intent = _intent_labels[pred]
            logger.debug("ML intent classification: %s (label_idx=%d)", intent, pred)
            valid_intents = {"research", "drafting", "analysis", "memory", "chat", "updates"}
            if intent in valid_intents:
                return {"current_agent": intent}
        except Exception as e:
            logger.debug("ML intent classification failed: %s", e)

    from app.services.llm.router import LLMRouter

    router = LLMRouter()
    classification_prompt = f"""Classifique a intenção do usuário em uma das categorias:
- research: pesquisa jurídica, busca de jurisprudência, legislação
- drafting: redação de petição, documento jurídico
- analysis: análise de caso, risk assessment, timeline
- memory: consulta sobre casos/clientes anteriores
- updates: novidades, atualizações recentes, conteúdo novo publicado
- chat: conversa geral, dúvidas simples

Mensagem: {last_message}

Responda APENAS com a categoria (research, drafting, analysis, memory ou chat)."""

    intent = await router.quick_classify(classification_prompt)
    intent = intent.strip().lower().split()[0] if intent else "chat"

    if "draft" in intent or "redac" in intent:
        intent = "drafting"
    elif "analy" in intent or "análise" in intent:
        intent = "analysis"
    elif "memory" in intent or "memór" in intent:
        intent = "memory"
    elif "research" in intent or "pesquis" in intent:
        intent = "research"
    elif "update" in intent or "novidade" in intent or "atualiz" in intent:
        intent = "updates"
    else:
        intent = "chat"

    if intent == "updates":
        intent = "research"

    return {"current_agent": intent}
