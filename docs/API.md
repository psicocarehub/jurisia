# API Juris.AI — Documentação

Documentação da API REST do Juris.AI — copiloto de IA para advogados brasileiros.

---

## Autenticação

### Fluxo de Autenticação

1. **Registro** — `POST /api/v1/admin/register`
2. **Login** — `POST /api/v1/admin/login`
3. Incluir o token no header: `Authorization: Bearer <token>`

### Obter Token

**Registro:**
```http
POST /api/v1/admin/register
Content-Type: application/json

{
  "email": "advogado@escritorio.com",
  "name": "João Silva",
  "password": "senha-segura",
  "tenant_slug": "escritorio-abc",
  "role": "lawyer",
  "oab_number": "SP 123456"
}
```

**Login:**
```http
POST /api/v1/admin/login
Content-Type: application/json

{
  "email": "advogado@escritorio.com",
  "password": "senha-segura",
  "tenant_slug": "escritorio-abc"
}
```

**Resposta:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "tenant_id": "uuid-do-tenant",
  "user_id": "uuid-do-usuario"
}
```

---

## Endpoints

Base URL: `/api/v1`

### Health

| Método | Endpoint    | Descrição        | Auth  |
|--------|-------------|------------------|-------|
| GET    | /health     | Health check     | Não   |

**Exemplo:**
```http
GET /health
```

```json
{
  "status": "ok",
  "service": "Juris.AI"
}
```

---

### Chat

| Método | Endpoint           | Descrição             | Auth |
|--------|--------------------|----------------------|------|
| POST   | /chat/completions  | Completions de chat  | Sim  |

**Exemplo:**
```http
POST /api/v1/chat/completions
Authorization: Bearer <token>
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Qual a súmula aplicável a rescisão de contrato?"}
  ],
  "case_id": "uuid-opcional",
  "session_id": "uuid-opcional",
  "stream": true,
  "use_rag": true,
  "use_memory": true
}
```

**Resposta (stream):** SSE com chunks `data: {"type": "token", "content": "..."}`

**Resposta (não-stream):**
```json
{
  "message": {"role": "assistant", "content": "..."},
  "sources": [],
  "thinking": null,
  "model_used": ""
}
```

---

### Documentos

| Método | Endpoint          | Descrição           | Auth |
|--------|-------------------|---------------------|------|
| GET    | /documents        | Lista documentos    | Sim  |
| POST   | /documents/upload | Upload de documento | Sim  |

**Listar:**
```http
GET /api/v1/documents?case_id=uuid&skip=0&limit=50
Authorization: Bearer <token>
```

**Upload:**
```http
POST /api/v1/documents/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <arquivo PDF>
case_id: uuid-opcional
```

---

### Busca Semântica

| Método | Endpoint | Descrição          | Auth |
|--------|----------|--------------------|------|
| POST   | /search  | Busca semântica    | Sim  |

```http
POST /api/v1/search
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "indenização por danos morais",
  "area": "cível",
  "court": "TJSP",
  "date_from": "2020-01-01",
  "date_to": "2025-01-01",
  "top_k": 10
}
```

---

### Casos

| Método | Endpoint    | Descrição       | Auth |
|--------|-------------|-----------------|------|
| GET    | /cases      | Lista casos     | Sim  |
| POST   | /cases      | Cria caso       | Sim  |
| GET    | /cases/{id} | Detalhe do caso | Sim  |
| PATCH  | /cases/{id} | Atualiza caso   | Sim  |
| DELETE | /cases/{id} | Remove caso     | Sim  |

**Criar caso:**
```http
POST /api/v1/cases
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "Ação de indenização",
  "cnj_number": "1234567-89.2024.8.26.0100",
  "description": "...",
  "area": "cível",
  "client_name": "Cliente XYZ",
  "court": "TJSP",
  "estimated_value": 50000.00
}
```

---

### Petições

| Método | Endpoint | Descrição          | Auth |
|--------|----------|--------------------|------|
| GET    | /petitions | Lista petições  | Sim  |
| POST   | /petitions | Cria petição    | Sim  |

```http
POST /api/v1/petitions
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "Petição Inicial",
  "case_id": "uuid",
  "petition_type": "peticao_inicial",
  "content": null
}
```

---

### Jurimetria

| Método | Endpoint               | Descrição        | Auth |
|--------|------------------------|------------------|------|
| GET    | /jurimetrics/judges/{name} | Perfil do juiz | Sim  |

```http
GET /api/v1/jurimetrics/judges/Maria%20Silva?court=TJSP
Authorization: Bearer <token>
```

---

### Memória

| Método | Endpoint     | Descrição       | Auth |
|--------|--------------|-----------------|------|
| POST   | /memory/search | Busca memória | Sim  |

```http
POST /api/v1/memory/search
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "preferências do cliente X",
  "case_id": "uuid-opcional"
}
```

---

### Admin

| Método | Endpoint    | Descrição      | Auth |
|--------|-------------|----------------|------|
| POST   | /admin/tenants  | Cria tenant | Não  |
| POST   | /admin/register  | Registra usuário | Não |
| POST   | /admin/login  | Login | Não |

---

## Códigos de Status

| Código | Significado        |
|--------|--------------------|
| 200    | Sucesso            |
| 201    | Criado             |
| 204    | Sem conteúdo       |
| 400    | Requisição inválida|
| 401    | Não autorizado     |
| 404    | Não encontrado     |
| 409    | Conflito           |
| 500    | Erro interno       |
