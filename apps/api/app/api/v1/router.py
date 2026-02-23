from fastapi import APIRouter

from app.api.v1 import (
    admin,
    alerts,
    cases,
    chat,
    compliance,
    documents,
    feedback,
    jurimetrics,
    memory,
    petitions,
    search,
)

api_router = APIRouter()

api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(cases.router)
api_router.include_router(petitions.router)
api_router.include_router(jurimetrics.router)
api_router.include_router(memory.router)
api_router.include_router(admin.router)
api_router.include_router(feedback.router)
api_router.include_router(alerts.router)
api_router.include_router(compliance.router)
