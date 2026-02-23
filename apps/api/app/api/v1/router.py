from fastapi import APIRouter

from app.api.v1 import chat, documents, search, cases, petitions, jurimetrics, memory, admin

api_router = APIRouter()

api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(cases.router)
api_router.include_router(petitions.router)
api_router.include_router(jurimetrics.router)
api_router.include_router(memory.router)
api_router.include_router(admin.router)
