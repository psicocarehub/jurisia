from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

AI_DISCLAIMER = (
    "⚠️ Conteúdo gerado com auxílio de inteligência artificial.\n"
    "Conforme CNJ Resolução 615/2025 e recomendações da OAB "
    "(Proposição 49.0000.2024.007325-9/COP), este conteúdo deve ser "
    "revisado por advogado habilitado antes de qualquer uso em processo judicial."
)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
