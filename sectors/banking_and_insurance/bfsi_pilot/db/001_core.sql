-- BFSI pilot — core schema (digital twin of a small insurer).
-- Runs automatically on first `docker compose up` (docker-entrypoint-initdb.d).
-- Flat SQL, no ORM. One deliberate addition vs the original spec:
-- policies.vehicle_plate — the honest link path invoice → claim (the
-- extractor reads a plate off the scan; plates live on KASKO policies).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE clients (
  id text PRIMARY KEY,            -- 'CLT-0001'
  name text NOT NULL,
  type text CHECK (type IN ('person','company')),
  risk_level text DEFAULT 'normal'
);

CREATE TABLE policies (
  id text PRIMARY KEY,            -- 'POL-0001'
  client_id text REFERENCES clients(id),
  product text CHECK (product IN ('KASKO','PROPERTY','HEALTH')),
  valid_from date, valid_to date,
  deductible_uah numeric,
  limit_uah numeric,
  vehicle_plate text,             -- KASKO only; link path for invoice scans
  file_ref text                   -- s3://bfsi/policies/POL-0001.pdf
);

CREATE TABLE claims (
  id text PRIMARY KEY,            -- 'CLM-0001'
  policy_id text REFERENCES policies(id),
  status text CHECK (status IN
    ('fnol','docs_pending','review','approved','paid','denied')),
  fnol_date date NOT NULL,
  loss_description text,
  amount_claimed_uah numeric,
  assignee text
);

CREATE TABLE documents (
  id text PRIMARY KEY,            -- 'DOC-0001'
  entity_type text CHECK (entity_type IN ('claim','policy','client')),
  entity_id text NOT NULL,        -- '' = not linked yet (worker fills after extract)
  doc_type text CHECK (doc_type IN
    ('repair_invoice','damage_photo_act','police_report','policy_pdf')),
  file_ref text NOT NULL,
  status text DEFAULT 'pending' CHECK (status IN
    ('pending','extracted','failed','verified')),
  extracted jsonb,
  confidence numeric,
  error text,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE doc_chunks (
  id bigserial PRIMARY KEY,
  document_id text REFERENCES documents(id),
  seq int,
  chunk_text text NOT NULL,
  section_ref text,               -- 'п. 5.4' style, from the numbering regex
  embedding vector(1024)          -- Qwen3-Embedding-0.6B
);
CREATE INDEX ON doc_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE audit_log (
  id bigserial PRIMARY KEY,
  actor text, action text, detail jsonb, ts timestamptz DEFAULT now()
);

-- ── Roles ────────────────────────────────────────────────────────────────────
-- ingest_rw: the ingest worker. agent_ro: the agent backend — SELECT
-- everywhere, INSERT only into audit_log. The agent NEVER runs as
-- ingest_rw; `python agent_server.py --selftest` proves the wall holds.

CREATE ROLE ingest_rw LOGIN PASSWORD 'ingest_rw';
CREATE ROLE agent_ro  LOGIN PASSWORD 'agent_ro';

GRANT USAGE ON SCHEMA public TO ingest_rw, agent_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ingest_rw, agent_ro;
GRANT INSERT, UPDATE ON documents, doc_chunks TO ingest_rw;
GRANT INSERT ON audit_log TO ingest_rw, agent_ro;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO ingest_rw, agent_ro;
