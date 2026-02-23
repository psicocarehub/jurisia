# Checklist de Compliance — LGPD, OAB, CNJ, PL 2338/2023

Checklist de conformidade para a plataforma Juris.AI.

---

## LGPD (Lei 13.709/2018)

- [ ] **Base legal**: Art. 7, VI (exercício regular de direitos em processo judicial)
- [ ] **Privacy by Design**: Art. 46, §2 — proteção desde a concepção
- [ ] **DPO nomeado**: Art. 41 — responsável pela proteção de dados
- [ ] **DPIA**: Art. 38 — avaliação de impacto para processamento de alto risco
- [ ] **Direito à revisão**: Art. 20 — decisões automatizadas devem permitir revisão humana
- [ ] **Criptografia**: AES-256 em repouso + TLS 1.3 em trânsito
- [ ] **Data residency**: Dados em território brasileiro (AWS sa-east-1)
- [ ] **Log de interações**: Todas as interações com IA registradas
- [ ] **Zero treinamento**: Política de não treinar modelos com dados de clientes

---

## OAB (Proposição 49.0000.2024.007325-9/COP, Nov 2024)

- [ ] **Item 4.4.1**: Formalizar por escrito intenção de usar IA ao cliente
- [ ] **Item 4.4.3**: Obter consentimento explícito ASSINADO
- [ ] **Item 2.3**: IA NÃO treinada com dados de clientes
- [ ] **Item 3.7**: Advogado REVISA todo output antes de protocolar
- [ ] **Funcionalidade**: Gerar, rastrear e arquivar termos de consentimento

---

## CNJ Resolução 615/2025

- [ ] **Classificação de risco**: Baixo (extração de dados) vs alto (análise comportamental)
- [ ] **Tags de transparência**: Em TODO conteúdo gerado por IA
- [ ] **12 meses**: Prazo para compliance (tribunais)
- [ ] **Art. 23**: Modelos preditivos em matéria criminal DESENCORAJADOS
- [ ] **Supervisão humana**: Obrigatória em decisões assistidas por IA

---

## PL 2338/2023 (Marco Legal da IA)

- [ ] **IA jurídica**: Provavelmente ALTO RISCO
- [ ] **Impact assessments**: Avaliações de impacto necessárias
- [ ] **Transparência**: Sobre capacidades e limitações
- [ ] **Human review**: Mecanismos de revisão humana
- [ ] **Audit trails**: Trilhas de auditoria completas

---

## Implementações Técnicas

### AI Label (CNJ Res. 615/2025)
Todo conteúdo gerado por IA deve incluir:
```
⚠️ Conteúdo gerado com auxílio de inteligência artificial.
Conforme CNJ Resolução 615/2025 e recomendações da OAB,
este conteúdo deve ser revisado por advogado habilitado
antes de qualquer uso em processo judicial.
```

### Predição Criminal (CNJ Art. 23)
O módulo de predição (`OutcomePredictor`) retorna aviso e não prediz em matéria criminal.

### Consentimento
Modelo de dados: `ai_consent_given` em usuários para rastrear consentimento OAB Item 4.4.3.
