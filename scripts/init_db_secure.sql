-- ============================================
-- Juris.AI — Production RLS Policies
-- ============================================
-- Replaces the permissive "allow_all_*" dev policies with
-- tenant-scoped isolation.
--
-- Prerequisites:
--   1. Run init_db.sql first (creates tables + enables RLS)
--   2. The API middleware must call:
--        SET LOCAL app.tenant_id = '<uuid>';
--      at the start of every request (within a transaction).
-- ============================================

-- Helper: drop all dev policies before creating production ones
DO $$ BEGIN
  DROP POLICY IF EXISTS "allow_all_tenants" ON tenants;
  DROP POLICY IF EXISTS "allow_all_users" ON users;
  DROP POLICY IF EXISTS "allow_all_cases" ON cases;
  DROP POLICY IF EXISTS "allow_all_documents" ON documents;
  DROP POLICY IF EXISTS "allow_all_chunks" ON document_chunks;
  DROP POLICY IF EXISTS "allow_all_conversations" ON conversations;
  DROP POLICY IF EXISTS "allow_all_messages" ON messages;
  DROP POLICY IF EXISTS "allow_all_petitions" ON petitions;
  DROP POLICY IF EXISTS "allow_all_audit" ON audit_logs;
  DROP POLICY IF EXISTS "allow_all_alerts" ON alerts;
  DROP POLICY IF EXISTS "allow_all_alert_subs" ON alert_subscriptions;
  DROP POLICY IF EXISTS "allow_all_feedback" ON feedback_log;
  DROP POLICY IF EXISTS "allow_all_kg_nodes" ON kg_nodes;
  DROP POLICY IF EXISTS "allow_all_kg_edges" ON kg_edges;
END $$;

-- ============================================
-- TENANTS: only see own tenant
-- ============================================
CREATE POLICY tenant_isolation_tenants ON tenants
  FOR ALL
  USING (id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (id = current_setting('app.tenant_id', true)::uuid);

-- ============================================
-- USERS: only see users in same tenant
-- ============================================
CREATE POLICY tenant_isolation_users ON users
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ============================================
-- CASES
-- ============================================
CREATE POLICY tenant_isolation_cases ON cases
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ============================================
-- DOCUMENTS
-- ============================================
CREATE POLICY tenant_isolation_documents ON documents
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ============================================
-- DOCUMENT CHUNKS
-- ============================================
CREATE POLICY tenant_isolation_chunks ON document_chunks
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ============================================
-- CONVERSATIONS
-- ============================================
CREATE POLICY tenant_isolation_conversations ON conversations
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ============================================
-- MESSAGES (inherit via conversation's tenant_id)
-- ============================================
CREATE POLICY tenant_isolation_messages ON messages
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ============================================
-- PETITIONS
-- ============================================
CREATE POLICY tenant_isolation_petitions ON petitions
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ============================================
-- AUDIT LOGS (read own tenant, insert only)
-- ============================================
CREATE POLICY tenant_isolation_audit_read ON audit_logs
  FOR SELECT
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

CREATE POLICY tenant_isolation_audit_insert ON audit_logs
  FOR INSERT
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ============================================
-- ALERTS
-- ============================================
CREATE POLICY tenant_isolation_alerts ON alerts
  FOR ALL
  USING (
    tenant_id IS NULL
    OR tenant_id = current_setting('app.tenant_id', true)::uuid
  )
  WITH CHECK (
    tenant_id IS NULL
    OR tenant_id = current_setting('app.tenant_id', true)::uuid
  );

-- ============================================
-- ALERT SUBSCRIPTIONS
-- ============================================
CREATE POLICY tenant_isolation_alert_subs ON alert_subscriptions
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ============================================
-- FEEDBACK LOG
-- ============================================
CREATE POLICY tenant_isolation_feedback ON feedback_log
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ============================================
-- KNOWLEDGE GRAPH (shared across tenants for legal knowledge)
-- ============================================
CREATE POLICY allow_read_kg_nodes ON kg_nodes
  FOR SELECT USING (true);
CREATE POLICY allow_insert_kg_nodes ON kg_nodes
  FOR INSERT WITH CHECK (true);

CREATE POLICY allow_read_kg_edges ON kg_edges
  FOR SELECT USING (true);
CREATE POLICY allow_insert_kg_edges ON kg_edges
  FOR INSERT WITH CHECK (true);

-- ============================================
-- SERVICE ROLE bypass (for background tasks, ingestion)
-- ============================================
-- Supabase service_role already bypasses RLS by default.
-- No additional policies needed for service-level operations.
