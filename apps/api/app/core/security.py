import bcrypt

AI_DISCLAIMER = (
    "Conteudo gerado com auxilio de inteligencia artificial.\n"
    "Conforme CNJ Resolucao 615/2025 e recomendacoes da OAB "
    "(Proposicao 49.0000.2024.007325-9/COP), este conteudo deve ser "
    "revisado por advogado habilitado antes de qualquer uso em processo judicial."
)


def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    pwd_bytes = plain_password.encode("utf-8")[:72]
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(pwd_bytes, hashed_bytes)
