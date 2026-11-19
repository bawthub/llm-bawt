-- llm-bawt baseline schema reference (TASK-631, revision 0001_baseline, master 9dfbd3c9).
-- Documentation only; not a restore recipe. Regenerate with: make migrate-schema-snapshot
--
-- PostgreSQL database dump
--



SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: pg_trgm; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: entitytype; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.entitytype AS ENUM (
    'USER',
    'BOT'
);


--
-- Name: jobstatus; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.jobstatus AS ENUM (
    'PENDING',
    'RUNNING',
    'SUCCESS',
    'FAILED',
    'SKIPPED'
);


--
-- Name: jobtype; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.jobtype AS ENUM (
    'PROFILE_MAINTENANCE',
    'MEMORY_CONSOLIDATION',
    'MEMORY_DECAY',
    'HISTORY_SUMMARIZATION',
    'MEMORY_EXTRACTION',
    'MEDIA_GC',
    'TOOL_RESULT_GC',
    'SEND_PROMPT',
    'IDLE_CONTEXT_RESET',
    'HARNESS_VERSION_CHECK',
    'PLAYGROUND_EXPIRY',
    'THREAD_AUTO_ARCHIVE',
    'GUIDANCE_RUN'
);


--
-- Name: bot_profiles_check_model_endpoint(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.bot_profiles_check_model_endpoint() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    endpoint_protocol TEXT;
    endpoint_vendor TEXT;
    endpoint_model_key TEXT;
BEGIN
    IF NEW.endpoint_id IS NULL THEN
        RAISE EXCEPTION 'bot %: harness % requires endpoint_id', NEW.slug, NEW.harness
        USING ERRCODE = 'not_null_violation';
    END IF;

    SELECT a.protocol, a.vendor, m.key
      INTO endpoint_protocol, endpoint_vendor, endpoint_model_key
      FROM model_endpoints e
      JOIN access_paths a ON a.id = e.access_path_id
      JOIN models m ON m.id = e.model_id
     WHERE e.id = NEW.endpoint_id;

    IF endpoint_protocol IS NULL THEN
        RAISE EXCEPTION 'bot %: endpoint_id % does not exist', NEW.slug, NEW.endpoint_id
        USING ERRCODE = 'foreign_key_violation';
    END IF;

    IF NEW.harness = 'chat' AND endpoint_protocol <> 'chat-completions' THEN
        RAISE EXCEPTION 'bot %: chat harness requires chat-completions endpoint, got %',
            NEW.slug, endpoint_protocol USING ERRCODE = 'check_violation';
    ELSIF NEW.harness = 'codex' AND endpoint_protocol <> 'responses' THEN
        RAISE EXCEPTION 'bot %: codex harness requires responses endpoint, got %',
            NEW.slug, endpoint_protocol USING ERRCODE = 'check_violation';
    ELSIF NEW.harness = 'claude-code'
          AND (endpoint_protocol <> 'anthropic-messages' OR endpoint_vendor <> 'anthropic') THEN
        RAISE EXCEPTION 'bot %: claude-code requires an Anthropic Messages endpoint',
            NEW.slug USING ERRCODE = 'check_violation';
    ELSIF NEW.harness = 'claude-proxy'
          AND (endpoint_vendor = 'anthropic'
               OR endpoint_protocol NOT IN ('anthropic-messages', 'responses', 'chat-completions')) THEN
        RAISE EXCEPTION 'bot %: claude-proxy requires a non-Anthropic proxy endpoint',
            NEW.slug USING ERRCODE = 'check_violation';
    ELSIF NEW.harness NOT IN ('chat', 'codex', 'claude-code', 'claude-proxy') THEN
        RAISE EXCEPTION 'bot %: unknown harness %', NEW.slug, NEW.harness
        USING ERRCODE = 'check_violation';
    END IF;

    -- The normalized pair is canonical. These two columns are derived mirrors
    -- retained for existing bot/runtime interfaces, never independent inputs.
    NEW.default_model := endpoint_model_key;
    NEW.agent_backend := CASE NEW.harness
        WHEN 'chat' THEN NULL
        WHEN 'claude-proxy' THEN 'claude-code'
        ELSE NEW.harness
    END;
    RETURN NEW;
END;
$$;


--
-- Name: bot_slug_aliases_guard(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.bot_slug_aliases_guard() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  PERFORM pg_advisory_xact_lock(hashtext('bot-storage'), hashtext(NEW.old_slug));
  IF EXISTS (SELECT 1 FROM bot_profiles
             WHERE COALESCE(NULLIF(regexp_replace(lower(slug), '[^a-z0-9_]', '', 'g'), ''), 'default') = NEW.old_slug) THEN
    RAISE EXCEPTION 'bot slug % is a live bot; an alias may only reserve a retired slug', NEW.old_slug
      USING ERRCODE = 'unique_violation';
  END IF;
  RETURN NEW;
END $$;


--
-- Name: inter_bot_deliveries_bump_revision(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.inter_bot_deliveries_bump_revision() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
        BEGIN
            NEW.revision := OLD.revision + 1;
            RETURN NEW;
        END;
        $$;


--
-- Name: inter_bot_deliveries_settle_batch(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.inter_bot_deliveries_settle_batch() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
        BEGIN
            IF NEW.status = OLD.status
               OR NOT (COALESCE(OLD.metadata_json, '{}'::jsonb) ? 'batch_member_ids') THEN
                RETURN NEW;
            END IF;
            IF NEW.status = 'DELIVERED' THEN
                UPDATE inter_bot_deliveries SET status='DELIVERED',
                    delivered_at=COALESCE(delivered_at, now()), updated_at=now()
                WHERE batch_head_id=NEW.id AND status='DISPATCHING' AND delivery_mode='batched';
            ELSIF NEW.status = 'FAILED'
                  AND (COALESCE(NEW.metadata_json->>'restart_retry_pending', '') = 'true'
                       OR current_setting('llm_bawt.batch_hold', true) = NEW.id) THEN
                -- A restart-retry decision is pending: settle_retry() decides
                -- the followers with it, keeping the batch under one owner.
                RETURN NEW;
            ELSIF NEW.status = 'QUEUED' THEN
                UPDATE inter_bot_deliveries SET status='QUEUED', delivery_mode=NULL,
                    target_turn_id=NULL, available_at=now(), next_retry_at=NULL,
                    last_error='released with its inbox batch head', updated_at=now()
                WHERE batch_head_id=NEW.id AND status='DISPATCHING' AND delivery_mode='batched';
            ELSIF NEW.status IN ('FAILED', 'CANCELLED') THEN
                UPDATE inter_bot_deliveries SET status=NEW.status,
                    last_error=COALESCE(NEW.last_error, 'inbox batch head ended'), updated_at=now()
                WHERE batch_head_id=NEW.id AND status='DISPATCHING' AND delivery_mode='batched';
            ELSE
                RETURN NEW;
            END IF;
            IF NEW.status <> 'DELIVERED' THEN
                NEW.metadata_json := NEW.metadata_json - 'batch_member_ids' - 'batch_prompt';
            END IF;
            RETURN NEW;
        END;
        $$;


--
-- Name: messages_content_tsv(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.messages_content_tsv(content text) RETURNS tsvector
    LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE
    AS $$
BEGIN
    RETURN to_tsvector('english'::regconfig, content);
EXCEPTION WHEN program_limit_exceeded THEN
    RETURN to_tsvector('english'::regconfig, left(content, 100000));
END
$$;


--
-- Name: messages_content_tsv_fill(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.messages_content_tsv_fill() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.content_tsv := messages_content_tsv(NEW.content);
    RETURN NEW;
END
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: access_paths; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.access_paths (
    id bigint NOT NULL,
    key character varying(128) NOT NULL,
    vendor character varying(64) NOT NULL,
    protocol character varying(32) NOT NULL,
    base_url text,
    auth_mechanism character varying(64) NOT NULL,
    engine_kind character varying(64),
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    system_prompt_overrides jsonb DEFAULT '{}'::jsonb NOT NULL,
    CONSTRAINT access_paths_protocol_check CHECK (((protocol)::text = ANY ((ARRAY['chat-completions'::character varying, 'responses'::character varying, 'anthropic-messages'::character varying])::text[])))
);


--
-- Name: access_paths_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.access_paths_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: access_paths_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.access_paths_id_seq OWNED BY public.access_paths.id;


--
-- Name: agent_content_lineage; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.agent_content_lineage (
    target_id character varying(64) NOT NULL,
    source_id character varying(64) NOT NULL,
    purpose character varying(16) DEFAULT 'legacy'::character varying NOT NULL
);


--
-- Name: agent_content_turns; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.agent_content_turns (
    id character varying(64) NOT NULL,
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    turn_id character varying(128) NOT NULL,
    trigger_message_id character varying(128) NOT NULL,
    untrusted_seen boolean NOT NULL
);


--
-- Name: agent_content_write_decisions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.agent_content_write_decisions (
    id character varying(36) NOT NULL,
    source_turn character varying(128) NOT NULL,
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    trigger_message_id character varying(128) NOT NULL,
    tool_name character varying(128) NOT NULL,
    write_kind character varying(32) NOT NULL,
    action character varying(32) NOT NULL,
    invocation_hash character varying(64) NOT NULL,
    untrusted_seen boolean NOT NULL,
    decision character varying(16) NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: api_balance_ledger; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.api_balance_ledger (
    turn_id character varying(128) NOT NULL,
    provider character varying(64) NOT NULL,
    access_path_key character varying(64) NOT NULL,
    turn_created_at timestamp with time zone NOT NULL,
    cost_usd numeric(18,8),
    updated_at timestamp with time zone NOT NULL,
    source character varying(16) DEFAULT 'chat'::character varying NOT NULL,
    usage_json text
);


--
-- Name: api_balances; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.api_balances (
    provider character varying(64) NOT NULL,
    anchor_usd numeric(18,8) NOT NULL,
    anchor_at timestamp with time zone NOT NULL,
    spent_usd numeric(18,8) DEFAULT 0 NOT NULL,
    unpriced_turns integer DEFAULT 0 NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    CONSTRAINT api_balances_anchor_usd_nonnegative CHECK ((anchor_usd >= (0)::numeric))
);


--
-- Name: approval_approver_fanout; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.approval_approver_fanout (
    request_id text NOT NULL,
    approver_bot_id text NOT NULL,
    user_id text NOT NULL,
    state text NOT NULL,
    request_delivery_id text,
    resolved_delivery_id text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: bawthub_auth_secrets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bawthub_auth_secrets (
    name character varying(80) NOT NULL,
    value_enc text NOT NULL,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL
);


--
-- Name: bot_profile_changes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bot_profile_changes (
    id bigint NOT NULL,
    bot_slug character varying(128) NOT NULL,
    changed_by character varying(128) NOT NULL,
    change_note text,
    changes jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT bot_profile_changes_changes_check CHECK ((jsonb_typeof(changes) = 'object'::text))
);


--
-- Name: bot_profile_changes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.bot_profile_changes_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: bot_profile_changes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.bot_profile_changes_id_seq OWNED BY public.bot_profile_changes.id;


--
-- Name: bot_profile_tags; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bot_profile_tags (
    bot_slug character varying(128) NOT NULL,
    tag_slug character varying(64) NOT NULL,
    assigned_at timestamp with time zone DEFAULT now() NOT NULL,
    assigned_by character varying(128),
    CONSTRAINT bot_profile_tags_bot_slug_check CHECK ((((bot_slug)::text = lower(btrim((bot_slug)::text))) AND ((bot_slug)::text <> ''::text)))
);


--
-- Name: bot_profiles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bot_profiles (
    id integer NOT NULL,
    slug character varying(128) NOT NULL,
    name character varying(255) NOT NULL,
    description text NOT NULL,
    system_prompt text NOT NULL,
    requires_memory boolean NOT NULL,
    voice_optimized boolean NOT NULL,
    uses_tools boolean NOT NULL,
    uses_search boolean NOT NULL,
    uses_home_assistant boolean NOT NULL,
    default_model character varying(255),
    nextcloud_config jsonb,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    color character varying(64),
    agent_backend character varying(128),
    agent_backend_config jsonb,
    tts_mode boolean DEFAULT false NOT NULL,
    include_summaries boolean DEFAULT true NOT NULL,
    default_voice character varying(128),
    avatar character varying(512),
    bot_type character varying(32),
    include_in_global_search boolean DEFAULT true NOT NULL,
    avatar_render text,
    prompt_override_id integer,
    harness character varying(32) DEFAULT 'chat'::character varying NOT NULL,
    endpoint_id bigint,
    last_session_id character varying(36),
    house character varying(32),
    house_pinned boolean DEFAULT false NOT NULL,
    cli_color character varying(64),
    CONSTRAINT bot_profiles_harness_check CHECK (((harness)::text = ANY ((ARRAY['chat'::character varying, 'claude-code'::character varying, 'codex'::character varying, 'claude-proxy'::character varying])::text[])))
);


--
-- Name: bot_profiles_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.bot_profiles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: bot_profiles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.bot_profiles_id_seq OWNED BY public.bot_profiles.id;


--
-- Name: bot_slug_aliases; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bot_slug_aliases (
    old_slug character varying(128) NOT NULL,
    bot_slug character varying(128) NOT NULL,
    reason text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT bot_slug_aliases_old_slug_check CHECK (((old_slug)::text = COALESCE(NULLIF(regexp_replace(lower((old_slug)::text), '[^a-z0-9_]'::text, ''::text, 'g'::text), ''::text), 'default'::text)))
);


--
-- Name: bot_tag_catalog; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bot_tag_catalog (
    slug character varying(64) NOT NULL,
    name character varying(128) NOT NULL,
    description text DEFAULT ''::text NOT NULL,
    kind character varying(16) NOT NULL,
    prompt_key character varying(160),
    capabilities jsonb DEFAULT '[]'::jsonb NOT NULL,
    system boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    color character varying(32),
    CONSTRAINT bot_tag_catalog_capabilities_check CHECK ((jsonb_typeof(capabilities) = 'array'::text)),
    CONSTRAINT bot_tag_catalog_check CHECK (((prompt_key IS NULL) OR (((kind)::text = 'role'::text) AND ((prompt_key)::text ~~ 'role.%'::text)))),
    CONSTRAINT bot_tag_catalog_check1 CHECK (((NOT system) OR ((kind)::text = 'behavior'::text))),
    CONSTRAINT bot_tag_catalog_kind_check CHECK (((kind)::text = ANY ((ARRAY['behavior'::character varying, 'role'::character varying])::text[]))),
    CONSTRAINT bot_tag_catalog_slug_check CHECK (((slug)::text ~ '^[a-z][a-z0-9_]{0,63}$'::text))
);


--
-- Name: bridge_run_accepts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bridge_run_accepts (
    backend character varying(64) NOT NULL,
    request_id character varying(128) NOT NULL,
    boot_id character varying(128) NOT NULL,
    pid1 boolean,
    accepted_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: chat_pending_questions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chat_pending_questions (
    tool_use_id character varying(128) NOT NULL,
    created_at timestamp with time zone NOT NULL,
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    turn_id character varying(128) NOT NULL,
    trigger_message_id character varying(128),
    session_key character varying(128),
    tool_name character varying(128) NOT NULL,
    arguments_json text NOT NULL,
    status character varying(32) NOT NULL,
    answer text,
    answered_at timestamp with time zone,
    origin_harness character varying(32) DEFAULT 'claude'::character varying NOT NULL,
    answer_json text,
    answered_turn_id character varying(128),
    live_deadline timestamp with time zone
);


--
-- Name: commit_authoring_turns; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.commit_authoring_turns (
    repo_key character varying(128) NOT NULL,
    commit_sha character varying(64) NOT NULL,
    turn_id character varying(128) NOT NULL,
    bot_id character varying(128) NOT NULL,
    created_at timestamp with time zone NOT NULL
);


--
-- Name: entity_profile_attributes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_profile_attributes (
    id integer NOT NULL,
    entity_type public.entitytype NOT NULL,
    entity_id character varying(100) NOT NULL,
    category character varying(50) NOT NULL,
    key character varying(100) NOT NULL,
    value json,
    confidence double precision NOT NULL,
    source character varying(50),
    source_message_id character varying(100),
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    status character varying(16) DEFAULT 'live'::character varying NOT NULL,
    retire_reason text
);


--
-- Name: entity_profile_attributes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entity_profile_attributes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entity_profile_attributes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entity_profile_attributes_id_seq OWNED BY public.entity_profile_attributes.id;


--
-- Name: entity_profiles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_profiles (
    id integer NOT NULL,
    entity_type public.entitytype NOT NULL,
    entity_id character varying(100) NOT NULL,
    display_name character varying(100),
    description character varying(500),
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    summary text,
    email character varying(255),
    summary_updated_at timestamp without time zone,
    color character varying(64),
    avatar character varying(512),
    avatar_render text,
    is_service boolean DEFAULT false NOT NULL
);


--
-- Name: entity_profiles_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entity_profiles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entity_profiles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entity_profiles_id_seq OWNED BY public.entity_profiles.id;


--
-- Name: forgotten_messages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.forgotten_messages (
    bot_id character varying(64) NOT NULL,
    id character varying(36) NOT NULL,
    role character varying(20) NOT NULL,
    content text NOT NULL,
    "timestamp" double precision NOT NULL,
    session_id character varying(36),
    processed boolean DEFAULT false,
    created_at timestamp without time zone,
    forgotten_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    author_entity_type character varying(16),
    author_entity_id character varying(100),
    interrupt_source_message_id character varying(36),
    interrupt_content_offset integer,
    summarized boolean DEFAULT false,
    summary_metadata jsonb,
    recalled_history boolean DEFAULT false,
    attachments jsonb DEFAULT '[]'::jsonb NOT NULL,
    reasoning text
)
PARTITION BY LIST (bot_id);


--
-- Name: gate_escalations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.gate_escalations (
    id bigint NOT NULL,
    gate_id bigint NOT NULL,
    bot_id character varying(128) NOT NULL,
    kind character varying(32) NOT NULL,
    subject character varying(128) NOT NULL,
    actor character varying(160) NOT NULL,
    turn_id character varying(128),
    outcome text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: gate_escalations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.gate_escalations ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.gate_escalations_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: gate_members; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.gate_members (
    id bigint NOT NULL,
    gate_id bigint NOT NULL,
    bot_id character varying(128) NOT NULL,
    turn_id character varying(128) NOT NULL,
    turn_status character varying(32) NOT NULL,
    disposition character varying(24) NOT NULL,
    backend character varying(64),
    agent_session_key character varying(128),
    agent_request_id character varying(128),
    producer_kind character varying(16) DEFAULT 'ordinary'::character varying NOT NULL,
    producer_id character varying(128),
    producer_claim character varying(128),
    control_state character varying(16) DEFAULT 'pending'::character varying NOT NULL,
    control_error text,
    ack_status character varying(32),
    ack_note text,
    acked_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    control_changed_at timestamp with time zone DEFAULT now(),
    execution_exit jsonb,
    control_delivery_id character varying(96),
    agent_steer_caps character varying(64),
    CONSTRAINT gate_members_control_state_check CHECK (((control_state)::text = ANY ((ARRAY['pending'::character varying, 'steering'::character varying, 'steered'::character varying, 'unsupported'::character varying, 'not_needed'::character varying, 'self'::character varying, 'unconfirmed'::character varying, 'cancelled'::character varying])::text[]))),
    CONSTRAINT gate_members_disposition_check CHECK (((disposition)::text = ANY ((ARRAY['revoked_unstarted'::character varying, 'accepted'::character varying, 'running'::character varying])::text[])))
);


--
-- Name: gate_members_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.gate_members ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.gate_members_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: google_home_probe_captures; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.google_home_probe_captures (
    request_key character varying(64) NOT NULL,
    created_at double precision NOT NULL,
    payload_enc text NOT NULL
);


--
-- Name: google_home_probe_links; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.google_home_probe_links (
    link_key character varying(64) NOT NULL,
    generation character varying(64) NOT NULL
);


--
-- Name: google_home_probe_tokens; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.google_home_probe_tokens (
    digest character varying(64) NOT NULL,
    link_key character varying(64) NOT NULL,
    generation character varying(64) NOT NULL,
    kind character varying(16) NOT NULL,
    expires_at double precision NOT NULL,
    data text NOT NULL
);


--
-- Name: gpu_handoff_plans; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.gpu_handoff_plans (
    generation integer NOT NULL,
    plan_json text NOT NULL,
    voice_lease character varying(32)
);


--
-- Name: gpu_handoff_plans_generation_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.gpu_handoff_plans_generation_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: gpu_handoff_plans_generation_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.gpu_handoff_plans_generation_seq OWNED BY public.gpu_handoff_plans.generation;


--
-- Name: gpu_handoff_resolved_video_jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.gpu_handoff_resolved_video_jobs (
    id character varying(32) NOT NULL,
    claim_generation integer NOT NULL,
    owner_generation integer NOT NULL,
    reviewed_by character varying(128) NOT NULL,
    outcome character varying(32) NOT NULL,
    evidence text NOT NULL,
    observations_json text NOT NULL,
    resolved_at double precision NOT NULL
);


--
-- Name: gpu_handoff_state; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.gpu_handoff_state (
    id integer NOT NULL,
    owner character varying(32) NOT NULL,
    phase character varying(32) NOT NULL,
    generation integer NOT NULL,
    offer_hash character varying(64),
    offer_user character varying(128),
    offer_target character varying(32),
    offer_expires double precision,
    actions text,
    next_action integer NOT NULL,
    pending_action character varying(128),
    last_job_id character varying(128),
    last_error text,
    tts_restart_key character varying(160),
    reconcile_key character varying(160)
);


--
-- Name: gpu_handoff_state_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.gpu_handoff_state_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: gpu_handoff_state_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.gpu_handoff_state_id_seq OWNED BY public.gpu_handoff_state.id;


--
-- Name: gpu_handoff_video_jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.gpu_handoff_video_jobs (
    id character varying(32) NOT NULL,
    generation integer NOT NULL
);


--
-- Name: gpu_video_calibrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.gpu_video_calibrations (
    generation integer NOT NULL,
    job_id character varying(32) NOT NULL,
    measurement_json text
);


--
-- Name: gpu_video_calibrations_generation_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.gpu_video_calibrations_generation_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: gpu_video_calibrations_generation_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.gpu_video_calibrations_generation_seq OWNED BY public.gpu_video_calibrations.generation;


--
-- Name: guidance_proposals; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.guidance_proposals (
    id bigint NOT NULL,
    user_id character varying(128) NOT NULL,
    guidance_key character varying(255) NOT NULL,
    base_version integer,
    base_version_row_id bigint,
    proposed_body text NOT NULL,
    diff text NOT NULL,
    reason text NOT NULL,
    evidence_json jsonb NOT NULL,
    status character varying(16) DEFAULT 'pending'::character varying NOT NULL,
    proposed_by character varying(128) NOT NULL,
    decided_by character varying(128),
    decided_at timestamp with time zone,
    result_version_row_id bigint,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT guidance_proposals_status_check CHECK (((status)::text = ANY ((ARRAY['pending'::character varying, 'approved'::character varying, 'rejected'::character varying, 'stale'::character varying])::text[])))
);


--
-- Name: guidance_proposals_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.guidance_proposals_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: guidance_proposals_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.guidance_proposals_id_seq OWNED BY public.guidance_proposals.id;


--
-- Name: guidance_run_receipts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.guidance_run_receipts (
    id bigint NOT NULL,
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    guidance_key character varying(255) NOT NULL,
    guidance_version integer,
    guidance_version_row_id bigint,
    body_sha256 character(64),
    cursor_before_ts double precision,
    cursor_before_id text,
    cursor_after_ts double precision,
    cursor_after_id text,
    consumed_count integer,
    status character varying(16) DEFAULT 'started'::character varying NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    finished_at timestamp with time zone,
    CONSTRAINT guidance_run_receipts_status_check CHECK (((status)::text = ANY ((ARRAY['started'::character varying, 'committed'::character varying, 'failed'::character varying])::text[])))
);


--
-- Name: guidance_run_receipts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.guidance_run_receipts_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: guidance_run_receipts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.guidance_run_receipts_id_seq OWNED BY public.guidance_run_receipts.id;


--
-- Name: history_read_cursors; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.history_read_cursors (
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    cursor_ts double precision,
    cursor_message_id text,
    processed_rows_total bigint DEFAULT 0 NOT NULL,
    last_run_at timestamp with time zone,
    last_coverage jsonb,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT history_read_cursors_check CHECK (((cursor_ts IS NULL) = (cursor_message_id IS NULL)))
);


--
-- Name: home_audio_jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.home_audio_jobs (
    sequence integer NOT NULL,
    id character varying(32) NOT NULL,
    request_key character varying(64) NOT NULL,
    payload text NOT NULL,
    status character varying(32) NOT NULL,
    created_at double precision NOT NULL,
    expires_at double precision NOT NULL,
    updated_at double precision NOT NULL,
    asset_id character varying(64),
    error text
);


--
-- Name: home_audio_jobs_sequence_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.home_audio_jobs_sequence_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: home_audio_jobs_sequence_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.home_audio_jobs_sequence_seq OWNED BY public.home_audio_jobs.sequence;


--
-- Name: home_audio_lane; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.home_audio_lane (
    id integer NOT NULL,
    owner character varying(32),
    lease_until double precision NOT NULL
);


--
-- Name: home_audio_lane_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.home_audio_lane_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: home_audio_lane_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.home_audio_lane_id_seq OWNED BY public.home_audio_lane.id;


--
-- Name: house_retirements; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.house_retirements (
    house character varying(32) NOT NULL,
    reconcile_id character varying(64) NOT NULL
);


--
-- Name: inter_bot_deliveries; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.inter_bot_deliveries (
    ordinal bigint NOT NULL,
    id character varying(96) NOT NULL,
    sender_bot_id character varying(128) NOT NULL,
    target_bot_id character varying(128) NOT NULL,
    message text NOT NULL,
    payload_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    metadata_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    user_message_id character varying(96) NOT NULL,
    turn_id character varying(128) NOT NULL,
    idempotency_key character varying(256),
    project_id character varying(128),
    task_id character varying(128),
    message_kind character varying(32),
    status character varying(24) NOT NULL,
    attempt_count integer DEFAULT 0 NOT NULL,
    max_attempts integer DEFAULT 5 NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    available_at timestamp with time zone DEFAULT now() NOT NULL,
    dispatch_started_at timestamp with time zone,
    lease_expires_at timestamp with time zone,
    delivered_at timestamp with time zone,
    next_retry_at timestamp with time zone,
    last_error text,
    response_model character varying(255),
    response_chars integer,
    claim_token character varying(96),
    claim_owner character varying(96),
    transport_accepted_at timestamp with time zone,
    target_turn_id character varying(128),
    delivery_mode character varying(24),
    session_policy character varying(32) DEFAULT 'continue'::character varying NOT NULL,
    reset_status character varying(24),
    reset_reason text,
    old_session_id character varying(36),
    new_session_id character varying(36),
    reset_at timestamp with time zone,
    overflow_recovery_count integer DEFAULT 0 NOT NULL,
    author_entity_type character varying(16) DEFAULT 'bot'::character varying NOT NULL,
    author_entity_id character varying(128) NOT NULL,
    revision bigint DEFAULT 0 NOT NULL,
    cancel_notified_at timestamp with time zone,
    lane character varying(16) DEFAULT 'normal'::character varying NOT NULL,
    sender_turn_id character varying(128),
    batch_head_id character varying(96),
    CONSTRAINT inter_bot_deliveries_author_entity_type_check CHECK (((author_entity_type)::text = ANY ((ARRAY['user'::character varying, 'bot'::character varying])::text[]))),
    CONSTRAINT inter_bot_deliveries_session_policy_check CHECK (((session_policy)::text = ANY ((ARRAY['continue'::character varying, 'reset_retain_history'::character varying, 'reset_without_history'::character varying])::text[]))),
    CONSTRAINT inter_bot_deliveries_status_check CHECK (((status)::text = ANY ((ARRAY['QUEUED'::character varying, 'STEERING'::character varying, 'DISPATCHING'::character varying, 'DELIVERED'::character varying, 'FAILED'::character varying, 'CANCELLED'::character varying])::text[])))
);


--
-- Name: inter_bot_deliveries_ordinal_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.inter_bot_deliveries_ordinal_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: inter_bot_deliveries_ordinal_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.inter_bot_deliveries_ordinal_seq OWNED BY public.inter_bot_deliveries.ordinal;


--
-- Name: job_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.job_runs (
    id character varying NOT NULL,
    job_id character varying NOT NULL,
    bot_id character varying NOT NULL,
    status public.jobstatus NOT NULL,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    duration_ms integer,
    result_json text,
    error_message text
);


--
-- Name: llm_response_records; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.llm_response_records (
    id integer NOT NULL,
    turn_id character varying(64) NOT NULL,
    response_id character varying(128) NOT NULL,
    bot_id character varying,
    user_id character varying,
    host_parent character varying(128),
    first_seen_seq integer NOT NULL,
    model character varying(128),
    fresh_input_tokens integer,
    cache_read_tokens integer,
    cache_write_tokens integer,
    output_tokens integer,
    usage_state character varying(16) NOT NULL,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    wait_ms integer,
    stream_ms integer
);


--
-- Name: llm_response_records_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.llm_response_records_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: llm_response_records_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.llm_response_records_id_seq OWNED BY public.llm_response_records.id;


--
-- Name: media_assets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.media_assets (
    id text NOT NULL,
    sha256 text NOT NULL,
    mime_type text NOT NULL,
    original_mime_type text,
    size_bytes integer NOT NULL,
    width integer,
    height integer,
    source text NOT NULL,
    owner_user_id text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    expires_at timestamp with time zone,
    kind text DEFAULT 'image'::text NOT NULL,
    filename text,
    storage_key text,
    CONSTRAINT media_assets_source_check CHECK ((source = ANY (ARRAY['chat_upload'::text, 'tool_generated'::text, 'agent_attachment'::text, 'tts_response'::text])))
);


--
-- Name: media_generations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.media_generations (
    id text NOT NULL,
    status text DEFAULT 'pending'::text NOT NULL,
    media_type text DEFAULT 'video'::text NOT NULL,
    prompt text NOT NULL,
    revised_prompt text,
    provider text DEFAULT 'xai'::text NOT NULL,
    model text DEFAULT 'grok-imagine-video'::text NOT NULL,
    aspect_ratio text DEFAULT '16:9'::text,
    duration real,
    actual_duration real,
    resolution text DEFAULT '720p'::text,
    width integer,
    height integer,
    file_path text,
    thumbnail_path text,
    file_size_bytes bigint,
    mime_type text,
    provider_job_id text,
    source_image_hash text,
    error text,
    progress integer DEFAULT 0,
    created_at timestamp with time zone DEFAULT now(),
    completed_at timestamp with time zone,
    original_prompt text,
    negative_prompt text
);


--
-- Name: memories; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.memories (
    bot_id character varying(64) NOT NULL,
    id character varying(36) NOT NULL,
    content text NOT NULL,
    tags jsonb DEFAULT '["misc"]'::jsonb NOT NULL,
    importance real DEFAULT 0.5 NOT NULL,
    source_message_ids jsonb,
    access_count integer DEFAULT 0,
    last_accessed timestamp without time zone,
    intent text,
    stakes text,
    emotional_charge real,
    recurrence_keywords jsonb,
    meaning_updated_at timestamp without time zone,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    embedding public.vector(384),
    meaning_embedding public.vector(384),
    owner_user_id text
)
PARTITION BY LIST (bot_id);


--
-- Name: memory_ownership_activations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.memory_ownership_activations (
    bot_id text NOT NULL,
    mode text NOT NULL,
    owner_user_id text,
    actor_id text NOT NULL,
    activated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    counts jsonb DEFAULT '{}'::jsonb NOT NULL,
    CONSTRAINT memory_ownership_activations_mode_check CHECK ((mode = ANY (ARRAY['new'::text, 'inferred'::text, 'attested'::text, 'mark_only'::text])))
);


--
-- Name: message_author_backfill_audit; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.message_author_backfill_audit (
    batch_id character varying(36) NOT NULL,
    bot_id character varying(100) NOT NULL,
    message_id character varying(36) NOT NULL,
    rule character varying(16) NOT NULL,
    old_type character varying(16),
    old_id character varying(100),
    new_type character varying(16) NOT NULL,
    new_id character varying(100) NOT NULL,
    applied_at timestamp with time zone DEFAULT now() NOT NULL,
    reverted_at timestamp with time zone
);


--
-- Name: message_route_eligibility; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.message_route_eligibility (
    owner_user_id character varying(128) NOT NULL,
    bot_id character varying(128) NOT NULL,
    enabled boolean DEFAULT false NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: message_routes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.message_routes (
    id character varying(96) NOT NULL,
    owner_user_id character varying(128) NOT NULL,
    actor_entity_type character varying(16) NOT NULL,
    actor_entity_id character varying(128) NOT NULL,
    idempotency_key character varying(256) NOT NULL,
    intent_hash character varying(64) NOT NULL,
    selected_bot_id character varying(128),
    model_endpoint_id integer,
    model_ref character varying(256),
    session_policy character varying(32) NOT NULL,
    scope character varying(16) NOT NULL,
    reason text DEFAULT ''::text NOT NULL,
    ranking_json jsonb,
    delivery_id character varying(96),
    state character varying(24) NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: messages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.messages (
    bot_id character varying(64) NOT NULL,
    id character varying(36) NOT NULL,
    role character varying(20) NOT NULL,
    content text NOT NULL,
    "timestamp" double precision NOT NULL,
    session_id character varying(36),
    processed boolean DEFAULT false,
    summarized boolean DEFAULT false,
    summary_metadata jsonb,
    recalled_history boolean DEFAULT false,
    attachments jsonb DEFAULT '[]'::jsonb NOT NULL,
    reasoning text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    author_entity_type character varying(16),
    author_entity_id character varying(100),
    interrupt_source_message_id character varying(36),
    interrupt_content_offset integer,
    content_tsv tsvector
)
PARTITION BY LIST (bot_id);


--
-- Name: model_endpoints; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_endpoints (
    id bigint NOT NULL,
    model_id bigint NOT NULL,
    access_path_id bigint NOT NULL,
    upstream_model_id character varying(512),
    serving_config jsonb DEFAULT '{}'::jsonb NOT NULL,
    context_window_override integer,
    tool_support_override character varying(64),
    pricing jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: model_endpoints_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.model_endpoints_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_endpoints_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.model_endpoints_id_seq OWNED BY public.model_endpoints.id;


--
-- Name: models; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.models (
    id bigint NOT NULL,
    key character varying(128) NOT NULL,
    vendor character varying(64) NOT NULL,
    display_name character varying(256) NOT NULL,
    description text,
    default_context_window integer NOT NULL,
    default_tool_support character varying(64),
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    compact_threshold integer,
    pricing_threshold integer,
    CONSTRAINT models_compact_threshold_positive CHECK (((compact_threshold IS NULL) OR (compact_threshold > 0))),
    CONSTRAINT models_context_window_positive CHECK ((default_context_window > 0)),
    CONSTRAINT models_pricing_threshold_nonnegative CHECK (((pricing_threshold IS NULL) OR (pricing_threshold >= 0)))
);


--
-- Name: models_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.models_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: models_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.models_id_seq OWNED BY public.models.id;


--
-- Name: new_session_event_outbox; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.new_session_event_outbox (
    delivery_id character varying(96) NOT NULL,
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    new_session_id character varying(36) NOT NULL,
    payload_json jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    published_at timestamp with time zone
);


--
-- Name: notify_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.notify_log (
    id character varying(32) NOT NULL,
    bot_id character varying(64) NOT NULL,
    created_at double precision NOT NULL,
    updated_at double precision NOT NULL,
    urgency character varying(16) NOT NULL,
    message_hash character varying(64) NOT NULL,
    dedupe_key character varying(128),
    result character varying(64) NOT NULL
);


--
-- Name: ops_bell_outbox; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ops_bell_outbox (
    id integer NOT NULL,
    event_key character varying(160) NOT NULL,
    job_id character varying(64) NOT NULL,
    phase character varying(16) NOT NULL,
    created_at timestamp with time zone NOT NULL,
    attempts integer DEFAULT 0 NOT NULL,
    next_attempt_at timestamp with time zone NOT NULL,
    delivered_at timestamp with time zone,
    gave_up_at timestamp with time zone,
    last_error character varying(256)
);


--
-- Name: ops_bell_outbox_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ops_bell_outbox_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ops_bell_outbox_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ops_bell_outbox_id_seq OWNED BY public.ops_bell_outbox.id;


--
-- Name: ops_host_access; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ops_host_access (
    id integer NOT NULL,
    label character varying(64) NOT NULL,
    host character varying(253) NOT NULL,
    port integer NOT NULL,
    login_user character varying(32) NOT NULL,
    known_host_keys_json text NOT NULL,
    status character varying(32) NOT NULL,
    passwordless_sudo boolean,
    last_op character varying(16),
    last_job_id character varying(64),
    last_result_json text,
    last_checked_at timestamp with time zone,
    created_by character varying(128),
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL
);


--
-- Name: ops_host_access_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ops_host_access_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ops_host_access_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ops_host_access_id_seq OWNED BY public.ops_host_access.id;


--
-- Name: ops_jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ops_jobs (
    id character varying(64) NOT NULL,
    operation_slug character varying(128) NOT NULL,
    operation_version integer DEFAULT 1 NOT NULL,
    operation_script_hash character varying(64) NOT NULL,
    args_json text NOT NULL,
    display_args_json text NOT NULL,
    caller_bot_id character varying(128),
    caller_user_id character varying(128),
    caller_turn_id character varying(128),
    caller_session_key character varying(128),
    caller_backend character varying(64),
    approval_request_id character varying(128),
    state character varying(16) DEFAULT 'queued'::character varying NOT NULL,
    host_unit_name character varying(128),
    submitted_at timestamp with time zone NOT NULL,
    dispatched_at timestamp with time zone,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    exit_code integer,
    output_tail text,
    error_text text,
    status_file_path character varying(1024),
    log_file_path character varying(1024),
    idempotency_key character varying(128) NOT NULL,
    retry_count integer DEFAULT 0 NOT NULL,
    last_reconcile_at timestamp with time zone,
    invocation_snapshot_json text,
    request_payload_json text,
    caller_actor character varying(128),
    quiesce_gate_id bigint,
    quiesce_driver_id character varying(64),
    quiesce_driver_lease_expires_at timestamp with time zone,
    quiesce_deadline_at timestamp with time zone,
    quiesce_fence_json text,
    quiesce_json text,
    receipt_callback_json text,
    receipt_callback_delivery_id character varying(128),
    worker_slot_at timestamp with time zone,
    process_spawn_hold_at timestamp with time zone,
    process_spawn_hold_deadline_at timestamp with time zone,
    process_spawn_hold_target_json text,
    process_spawn_hold_quarantine_at timestamp with time zone,
    process_spawn_hold_release_json text,
    gate_retired_at timestamp with time zone,
    gate_plan_key character varying(128),
    sequence_json text
);


--
-- Name: ops_operation_revisions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ops_operation_revisions (
    id character varying NOT NULL,
    operation_id character varying NOT NULL,
    operation_slug character varying NOT NULL,
    version integer NOT NULL,
    snapshot_json text NOT NULL,
    actor character varying,
    recorded_at timestamp with time zone NOT NULL
);


--
-- Name: ops_operations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ops_operations (
    id character varying(64) NOT NULL,
    slug character varying(128) NOT NULL,
    title character varying(256) NOT NULL,
    description text NOT NULL,
    enabled boolean DEFAULT false NOT NULL,
    executor_kind character varying(32) NOT NULL,
    target_host character varying(256) NOT NULL,
    run_as_user character varying(64),
    working_directory character varying(1024),
    command_script text NOT NULL,
    args_schema_json text NOT NULL,
    args_defaults_json text NOT NULL,
    timeout_seconds integer DEFAULT 300 NOT NULL,
    start_delay_seconds integer DEFAULT 0 NOT NULL,
    max_output_bytes integer DEFAULT 65536 NOT NULL,
    max_concurrent integer,
    risk_level character varying(16) DEFAULT 'medium'::character varying NOT NULL,
    category character varying(64),
    approval_prompt_prefix text,
    version integer DEFAULT 1 NOT NULL,
    script_hash character varying(64) NOT NULL,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    created_by character varying(128),
    updated_by character varying(128),
    soft_deleted_at timestamp with time zone,
    quiesce_enabled boolean DEFAULT false NOT NULL,
    quiesce_timeout_seconds integer DEFAULT 600 NOT NULL,
    quiesce_scope_json text,
    execution_mode character varying(16) DEFAULT 'process'::character varying NOT NULL,
    execution_tag character varying(64),
    notify_bell boolean DEFAULT false NOT NULL,
    approval_required boolean DEFAULT true NOT NULL
);


--
-- Name: ops_release_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ops_release_events (
    id character varying NOT NULL,
    release_run_id character varying(64) NOT NULL,
    sequence integer NOT NULL,
    event_type character varying(64) NOT NULL,
    state character varying(40) NOT NULL,
    detail_json text NOT NULL,
    created_at timestamp with time zone NOT NULL
);


--
-- Name: ops_release_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ops_release_runs (
    id character varying(64) NOT NULL,
    parent_job_id character varying(64) NOT NULL,
    release_request_id character varying(128) NOT NULL,
    state character varying(40) DEFAULT 'preflight'::character varying NOT NULL,
    release_task character varying(64) NOT NULL,
    bump character varying(16) NOT NULL,
    github_repository character varying(256) NOT NULL,
    workflow_path character varying(512) NOT NULL,
    canonical_branch character varying(256) NOT NULL,
    source_sha character varying(40),
    release_plan_json text,
    dispatch_started_at timestamp with time zone,
    github_run_id character varying(32),
    github_run_attempt integer,
    github_run_url text,
    rerun_count integer DEFAULT 0 NOT NULL,
    receipt_json text,
    receipt_verified_at timestamp with time zone,
    deployable boolean DEFAULT false NOT NULL,
    version character varying(32),
    tag character varying(128),
    digest character varying(80),
    image_repository character varying(256),
    deploy_approval_request_id character varying(128),
    deploy_job_id character varying(64),
    next_check_at timestamp with time zone,
    phase_deadline_at timestamp with time zone,
    error_code character varying(64),
    error_text text,
    claim_token character varying(64),
    claim_expires_at timestamp with time zone,
    reconcile_attempts integer DEFAULT 0 NOT NULL,
    event_seq integer DEFAULT 0 NOT NULL,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    finished_at timestamp with time zone
);


--
-- Name: ops_resume_notices; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ops_resume_notices (
    job_id character varying(64) NOT NULL,
    bot_id character varying(128) NOT NULL,
    kind character varying(16) NOT NULL,
    delivery_id character varying(96),
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ops_resume_notices_kind_check CHECK (((kind)::text = ANY ((ARRAY['resume'::character varying, 'escalation'::character varying, 'done'::character varying])::text[])))
);


--
-- Name: playground_artifacts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.playground_artifacts (
    id character varying(40) NOT NULL,
    user_id character varying(128) NOT NULL,
    sender_bot_id character varying(64) NOT NULL,
    title character varying(200) NOT NULL,
    note character varying(300),
    kind character varying(16) NOT NULL,
    mime character varying(200) NOT NULL,
    filename character varying(255),
    size_bytes bigint NOT NULL,
    blob_key character varying(80),
    inline_content text,
    view_token_hash character varying(64),
    created_at timestamp with time zone NOT NULL,
    opened_at timestamp with time zone,
    expired_at timestamp with time zone,
    archived_at timestamp with time zone,
    trashed_at timestamp with time zone
);


--
-- Name: prompt_occurrences; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_occurrences (
    run_id character varying NOT NULL,
    job_id character varying NOT NULL,
    occurrence_key character varying NOT NULL,
    scheduled_for timestamp with time zone NOT NULL,
    kind character varying NOT NULL,
    schedule_revision integer NOT NULL,
    snapshot_json json NOT NULL,
    session_id character varying,
    delivery_id character varying,
    state character varying NOT NULL,
    enqueue_attempts integer NOT NULL,
    last_error text,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    main_notice_json json,
    main_notice_turn_id character varying,
    CONSTRAINT prompt_occurrences_kind_check CHECK (((kind)::text = ANY ((ARRAY['scheduled'::character varying, 'manual'::character varying])::text[]))),
    CONSTRAINT prompt_occurrences_state_check CHECK (((state)::text = ANY ((ARRAY['pending'::character varying, 'queued'::character varying, 'running'::character varying, 'succeeded'::character varying, 'failed'::character varying, 'skipped'::character varying, 'cancelled'::character varying, 'unknown'::character varying])::text[])))
);


--
-- Name: prompt_schedules; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_schedules (
    job_id character varying NOT NULL,
    owner_user_id character varying NOT NULL,
    name character varying(120) NOT NULL,
    description text NOT NULL,
    prompt text NOT NULL,
    requested_model character varying,
    create_request_hash character varying,
    timing_json json NOT NULL,
    clear_context boolean NOT NULL,
    augment_memory boolean NOT NULL,
    extract_memory boolean NOT NULL,
    dedicated_session_id character varying,
    missed_policy character varying NOT NULL,
    misfire_grace_seconds integer NOT NULL,
    lifecycle character varying NOT NULL,
    revision integer NOT NULL,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    cancelled_at timestamp with time zone,
    target_mode character varying DEFAULT 'specific'::character varying NOT NULL,
    candidate_bot_ids json,
    creator_bot_id character varying,
    task_id character varying,
    task_owner_bot_id character varying,
    CONSTRAINT prompt_schedules_lifecycle_check CHECK (((lifecycle)::text = ANY ((ARRAY['active'::character varying, 'paused'::character varying, 'completed'::character varying, 'cancelled'::character varying])::text[]))),
    CONSTRAINT prompt_schedules_misfire_grace_seconds_check CHECK (((misfire_grace_seconds >= 60) AND (misfire_grace_seconds <= 604800))),
    CONSTRAINT prompt_schedules_missed_policy_check CHECK (((missed_policy)::text = ANY ((ARRAY['skip'::character varying, 'run_latest'::character varying])::text[]))),
    CONSTRAINT prompt_schedules_name_check CHECK (((length(TRIM(BOTH FROM name)) >= 1) AND (length(TRIM(BOTH FROM name)) <= 120))),
    CONSTRAINT prompt_schedules_prompt_check CHECK (((length(TRIM(BOTH FROM prompt)) >= 1) AND (length(TRIM(BOTH FROM prompt)) <= 100000))),
    CONSTRAINT prompt_schedules_revision_check CHECK ((revision > 0))
);


--
-- Name: prompt_template_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_template_versions (
    id integer NOT NULL,
    key character varying(255) NOT NULL,
    scope_type character varying(16) NOT NULL,
    scope_id character varying(128) NOT NULL,
    version integer NOT NULL,
    body text NOT NULL,
    change_note text,
    created_by character varying(255),
    created_at timestamp with time zone NOT NULL
);


--
-- Name: prompt_template_versions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_template_versions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prompt_template_versions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_template_versions_id_seq OWNED BY public.prompt_template_versions.id;


--
-- Name: prompt_templates; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_templates (
    id integer NOT NULL,
    key character varying(255) NOT NULL,
    scope_type character varying(16) NOT NULL,
    scope_id character varying(128) NOT NULL,
    title character varying(255),
    category character varying(64),
    format character varying(32) NOT NULL,
    body text NOT NULL,
    required_vars_json text,
    metadata_json text,
    updated_by character varying(255),
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL
);


--
-- Name: prompt_templates_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_templates_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prompt_templates_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_templates_id_seq OWNED BY public.prompt_templates.id;


--
-- Name: quiesce_run_sightings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.quiesce_run_sightings (
    backend character varying(64) NOT NULL,
    request_id character varying(128) NOT NULL,
    boot_id character varying(128) NOT NULL,
    pid1 boolean,
    first_seen_at timestamp with time zone DEFAULT now() NOT NULL,
    last_seen_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: recall_evidence_ledgers; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.recall_evidence_ledgers (
    handle_sha256 character(64) NOT NULL,
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    session_id character varying(128) NOT NULL,
    coverage jsonb NOT NULL,
    entries jsonb NOT NULL,
    omitted integer DEFAULT 0 NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    CONSTRAINT recall_evidence_ledgers_coverage_check CHECK ((jsonb_typeof(coverage) = 'object'::text)),
    CONSTRAINT recall_evidence_ledgers_entries_check CHECK ((jsonb_typeof(entries) = 'array'::text)),
    CONSTRAINT recall_evidence_ledgers_omitted_check CHECK ((omitted >= 0))
);


--
-- Name: response_audio; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.response_audio (
    id text NOT NULL,
    turn_id text NOT NULL,
    assistant_message_id text,
    bot_id text NOT NULL,
    session_id text,
    owner_user_id text NOT NULL,
    status text NOT NULL,
    media_asset_id text,
    duration_ms integer,
    size_bytes integer,
    lease_token text,
    lease_expires_at timestamp with time zone,
    error text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT response_audio_status_check CHECK ((status = ANY (ARRAY['recording'::text, 'finalized'::text, 'partial'::text, 'failed'::text])))
);


--
-- Name: runtime_settings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.runtime_settings (
    id integer NOT NULL,
    scope_type character varying(16) NOT NULL,
    scope_id character varying(128) NOT NULL,
    key character varying(128) NOT NULL,
    value_json text NOT NULL,
    updated_at timestamp with time zone NOT NULL
);


--
-- Name: runtime_settings_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.runtime_settings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: runtime_settings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.runtime_settings_id_seq OWNED BY public.runtime_settings.id;


--
-- Name: scheduled_jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scheduled_jobs (
    id character varying NOT NULL,
    job_type public.jobtype NOT NULL,
    bot_id character varying NOT NULL,
    enabled boolean NOT NULL,
    interval_minutes integer NOT NULL,
    last_run_at timestamp with time zone,
    next_run_at timestamp with time zone,
    created_at timestamp with time zone,
    config_json text
);


--
-- Name: session_resumption_receipts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.session_resumption_receipts (
    bot_id character varying(128) NOT NULL,
    trigger_message_id character varying(128) NOT NULL,
    turn_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    session_id character varying(36) NOT NULL,
    previous_session_id character varying(36) NOT NULL,
    accepted_at double precision NOT NULL,
    session_started_at double precision,
    title text
);


--
-- Name: sessions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sessions (
    id character varying(36) NOT NULL,
    bot_id character varying(64) NOT NULL,
    started_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    ended_at timestamp without time zone,
    status character varying(16) DEFAULT 'active'::character varying NOT NULL,
    session_metadata jsonb,
    user_id character varying(64),
    archived_at timestamp without time zone
);


--
-- Name: skill_assignments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.skill_assignments (
    scope_type text NOT NULL,
    scope_id text NOT NULL,
    skill_ref text NOT NULL,
    mode text NOT NULL,
    "position" integer DEFAULT 0 NOT NULL,
    updated_by text,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT skill_assignments_mode_vocab CHECK ((mode = ANY (ARRAY['enable'::text, 'disable'::text, 'inject'::text, 'available'::text, 'off'::text, 'injected'::text]))),
    CONSTRAINT skill_assignments_position_check CHECK (("position" >= 0)),
    CONSTRAINT skill_assignments_scope_vocab CHECK (((scope_type = ANY (ARRAY['bot'::text, 'tag'::text, 'global'::text])) AND ((scope_type <> 'global'::text) OR (scope_id = '*'::text))))
);


--
-- Name: skill_managed_bots; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.skill_managed_bots (
    bot_slug text NOT NULL,
    adopted_by text,
    adopted_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: system_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.system_events (
    id bigint NOT NULL,
    event_key character varying(256) NOT NULL,
    kind character varying(64) NOT NULL,
    source character varying(32) NOT NULL,
    user_id character varying(128),
    visibility character varying(16) NOT NULL,
    bot_id character varying(128),
    actor_type character varying(32),
    actor_id character varying(128),
    subject_type character varying(32),
    subject_id character varying(128),
    tone character varying(16) NOT NULL,
    summary character varying(500) NOT NULL,
    payload jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    published_at timestamp with time zone,
    CONSTRAINT ck_system_events_feed_owner CHECK ((((visibility)::text = 'audit_only'::text) OR (user_id IS NOT NULL))),
    CONSTRAINT ck_system_events_tone CHECK (((tone)::text = ANY ((ARRAY['info'::character varying, 'success'::character varying, 'warning'::character varying, 'error'::character varying])::text[]))),
    CONSTRAINT ck_system_events_visibility CHECK (((visibility)::text = ANY ((ARRAY['user_feed'::character varying, 'audit_only'::character varying])::text[])))
);


--
-- Name: system_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.system_events ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.system_events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: system_gates; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.system_gates (
    id bigint NOT NULL,
    owner_kind character varying(16) NOT NULL,
    owner_id character varying(128) NOT NULL,
    mode character varying(8) NOT NULL,
    scope jsonb NOT NULL,
    reason text,
    allow_actors jsonb DEFAULT '[]'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    released_at timestamp with time zone,
    admit_user boolean DEFAULT false,
    CONSTRAINT system_gates_mode_check CHECK (((mode)::text = ANY ((ARRAY['hold'::character varying, 'deny'::character varying])::text[]))),
    CONSTRAINT system_gates_owner_kind_check CHECK (((owner_kind)::text = ANY ((ARRAY['job'::character varying, 'operator'::character varying])::text[])))
);


--
-- Name: system_gates_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.system_gates ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.system_gates_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: talk_reply_subscriptions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.talk_reply_subscriptions (
    message_id character varying(96) NOT NULL,
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    source_type character varying(16) NOT NULL,
    source_id character varying(128) NOT NULL,
    backend text NOT NULL,
    conversation_token text NOT NULL,
    reply_to bigint,
    state character varying(16) DEFAULT 'PENDING'::character varying NOT NULL,
    last_error text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    next_check_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT talk_reply_subscriptions_source_type_check CHECK (((source_type)::text = ANY ((ARRAY['delivery'::character varying, 'turn'::character varying])::text[]))),
    CONSTRAINT talk_reply_subscriptions_state_check CHECK (((state)::text = ANY ((ARRAY['PENDING'::character varying, 'SENDING'::character varying, 'SENT'::character varying, 'UNCERTAIN'::character varying])::text[])))
);


--
-- Name: tool_approval_decisions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tool_approval_decisions (
    id character varying NOT NULL,
    created_at timestamp with time zone NOT NULL,
    backend character varying NOT NULL,
    tool_name character varying NOT NULL,
    action character varying NOT NULL,
    severity character varying NOT NULL,
    subject text NOT NULL,
    policy_id character varying,
    policy_version integer,
    policy_snapshot_json text,
    bundle_etag character varying NOT NULL,
    invocation_hash character varying NOT NULL,
    bot_id character varying,
    user_id character varying,
    turn_id character varying,
    source character varying NOT NULL,
    session_key character varying,
    session_id character varying,
    request_id character varying,
    tool_use_id character varying,
    outcome character varying
);


--
-- Name: tool_approval_policies; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tool_approval_policies (
    id character varying(64) NOT NULL,
    enabled boolean NOT NULL,
    backend_scope character varying(64) NOT NULL,
    tool_name character varying(128) NOT NULL,
    matcher_type character varying(16) NOT NULL,
    pattern text NOT NULL,
    field character varying(128) NOT NULL,
    action character varying(24) NOT NULL,
    severity character varying(16) NOT NULL,
    category character varying(64),
    approval_prompt text,
    order_index integer NOT NULL,
    version integer NOT NULL,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    created_by character varying(128),
    updated_by character varying(128)
);


--
-- Name: tool_approval_policy_revisions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tool_approval_policy_revisions (
    id character varying(128) NOT NULL,
    policy_id character varying NOT NULL,
    version integer NOT NULL,
    change character varying NOT NULL,
    actor character varying,
    created_at timestamp with time zone NOT NULL,
    snapshot_json text NOT NULL
);


--
-- Name: tool_approval_requests; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tool_approval_requests (
    id character varying(128) NOT NULL,
    created_at timestamp with time zone NOT NULL,
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    turn_id character varying(128) NOT NULL,
    trigger_message_id character varying(128),
    session_key character varying(128),
    backend character varying(64) NOT NULL,
    tool_name character varying(128) NOT NULL,
    tool_arguments_json text NOT NULL,
    subject text NOT NULL,
    grant_key character varying(64) NOT NULL,
    policy_id character varying(64),
    severity character varying(16) NOT NULL,
    prompt text NOT NULL,
    status character varying(24) NOT NULL,
    resolved_at timestamp with time zone,
    resolved_by character varying(128),
    resolved_turn_id character varying(128),
    request_kind character varying(16) DEFAULT 'harness'::character varying NOT NULL,
    tool_use_id character varying(128),
    mcp_server character varying(128),
    invocation_hash character varying(64),
    caller_context_json text,
    continuation_capable boolean DEFAULT false NOT NULL,
    execution_state character varying(16) DEFAULT 'not_applicable'::character varying NOT NULL,
    execution_attempts integer DEFAULT 0 NOT NULL,
    execution_started_at timestamp with time zone,
    execution_finished_at timestamp with time zone,
    execution_error text,
    result_json text,
    result_is_error boolean,
    continuation_state character varying(16) DEFAULT 'not_needed'::character varying NOT NULL,
    continuation_attempts integer DEFAULT 0 NOT NULL,
    continuation_last_error text,
    continuation_next_attempt_at timestamp with time zone,
    continuation_delivered_at timestamp with time zone,
    resolution_message text DEFAULT ''::text NOT NULL,
    continuation_owner character varying(16) DEFAULT 'client'::character varying NOT NULL,
    operations_snapshot_json text,
    execution_claim_token character varying(64),
    continuation_claim_token character varying(64),
    continuation_id character varying(128),
    grant_state character varying(24) DEFAULT 'pending'::character varying NOT NULL,
    grant_error text,
    resolved_via text DEFAULT 'user'::text NOT NULL,
    CONSTRAINT ck_tool_approval_requests_pending_mcp_routable CHECK ((((request_kind)::text <> 'mcp'::text) OR ((status)::text <> 'pending'::text) OR (NOT ((bot_id IS NULL) OR (btrim((bot_id)::text) = ''::text) OR (lower(btrim((bot_id)::text)) = ANY (ARRAY['unknown'::text, 'none'::text, 'null'::text, 'undefined'::text])) OR (user_id IS NULL) OR (btrim((user_id)::text) = ''::text) OR (lower(btrim((user_id)::text)) = ANY (ARRAY['unknown'::text, 'none'::text, 'null'::text, 'undefined'::text])) OR (turn_id IS NULL) OR (btrim((turn_id)::text) = ''::text) OR (lower(btrim((turn_id)::text)) = ANY (ARRAY['unknown'::text, 'none'::text, 'null'::text, 'undefined'::text])) OR (trigger_message_id IS NULL) OR (btrim((trigger_message_id)::text) = ''::text) OR (lower(btrim((trigger_message_id)::text)) = ANY (ARRAY['unknown'::text, 'none'::text, 'null'::text, 'undefined'::text])) OR (session_key IS NULL) OR (btrim((session_key)::text) = ''::text) OR (lower(btrim((session_key)::text)) = ANY (ARRAY['unknown'::text, 'none'::text, 'null'::text, 'undefined'::text])) OR (backend IS NULL) OR (btrim((backend)::text) = ''::text) OR (lower(btrim((backend)::text)) = ANY (ARRAY['unknown'::text, 'none'::text, 'null'::text, 'undefined'::text])) OR (tool_use_id IS NULL) OR (btrim((tool_use_id)::text) = ''::text) OR (lower(btrim((tool_use_id)::text)) = ANY (ARRAY['unknown'::text, 'none'::text, 'null'::text, 'undefined'::text])) OR (mcp_server IS NULL) OR (btrim((mcp_server)::text) = ''::text) OR (lower(btrim((mcp_server)::text)) = ANY (ARRAY['unknown'::text, 'none'::text, 'null'::text, 'undefined'::text])) OR (invocation_hash IS NULL) OR ((invocation_hash)::text !~ '^[0-9a-f]{64}$'::text) OR (caller_context_json IS NULL) OR (btrim(caller_context_json) = ''::text) OR (continuation_capable IS NOT TRUE)))))
);


--
-- Name: tool_call_records; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tool_call_records (
    id integer NOT NULL,
    created_at timestamp with time zone NOT NULL,
    turn_id character varying,
    bot_id character varying,
    user_id character varying,
    call_id character varying,
    tool_name character varying(128) NOT NULL,
    arguments_json text,
    result_text text,
    iteration integer NOT NULL,
    started_at double precision,
    ended_at double precision,
    duration_ms double precision,
    text_offset integer,
    is_error boolean,
    tool_use_id character varying,
    parent_tool_use_id character varying,
    approval_request_id character varying,
    approval_status character varying,
    preapproved boolean,
    result_total_chars integer,
    result_total_bytes bigint,
    result_sha256 character varying(64),
    result_content_type character varying(128),
    result_complete boolean,
    result_payload_available boolean DEFAULT false NOT NULL,
    result_blob_key character varying(128),
    result_total_tokens integer,
    attachments_json text,
    exec_started_at double precision,
    skill_context_tokens integer,
    result_accounting integer,
    result_media_blocks integer,
    exec_tracked boolean DEFAULT false NOT NULL,
    response_id character varying(128)
);


--
-- Name: tool_call_records_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.tool_call_records_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: tool_call_records_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.tool_call_records_id_seq OWNED BY public.tool_call_records.id;


--
-- Name: tool_call_result_payloads; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tool_call_result_payloads (
    tool_call_record_id integer NOT NULL,
    content_text text NOT NULL,
    content_type character varying(128) NOT NULL,
    total_chars integer NOT NULL,
    total_bytes bigint NOT NULL,
    sha256 character varying(64) NOT NULL,
    complete boolean NOT NULL,
    created_at timestamp with time zone NOT NULL
);


--
-- Name: tool_call_result_payloads_tool_call_record_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.tool_call_result_payloads_tool_call_record_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: tool_call_result_payloads_tool_call_record_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.tool_call_result_payloads_tool_call_record_id_seq OWNED BY public.tool_call_result_payloads.tool_call_record_id;


--
-- Name: turn_changed_files; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.turn_changed_files (
    id integer NOT NULL,
    created_at timestamp with time zone NOT NULL,
    turn_id character varying(128) NOT NULL,
    bot_id character varying,
    user_id character varying,
    trigger_message_id character varying,
    repo_key character varying(128) NOT NULL,
    repo_label character varying(255),
    path text NOT NULL,
    old_path text,
    change_kind character varying(16) NOT NULL,
    additions integer,
    deletions integer,
    "binary" boolean NOT NULL,
    truncated boolean NOT NULL,
    before_sha256 character varying(64),
    before_bytes bigint,
    before_blob_key character varying(128),
    after_sha256 character varying(64),
    after_bytes bigint,
    after_blob_key character varying(128),
    content_type character varying(128),
    source_tool_call_ids text,
    commit_requested_at timestamp with time zone,
    in_repo boolean,
    ignored_version character varying(64),
    workspace_repo_id character varying(128),
    workspace_repo_name character varying(255),
    workspace_id character varying(160),
    workspace_kind character varying(16),
    repo_relative_path text,
    git_branch character varying(255),
    git_head_sha character varying(64),
    git_base_sha character varying(64),
    identity_status character varying(16),
    provenance character varying(32),
    restored_version character varying(64),
    updated_at timestamp with time zone
);


--
-- Name: turn_changed_files_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.turn_changed_files_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: turn_changed_files_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.turn_changed_files_id_seq OWNED BY public.turn_changed_files.id;


--
-- Name: turn_diff_payloads; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.turn_diff_payloads (
    sha256 character varying(64) NOT NULL,
    content text NOT NULL,
    total_bytes bigint NOT NULL,
    created_at timestamp with time zone NOT NULL
);


--
-- Name: turn_egress_requests; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.turn_egress_requests (
    id integer NOT NULL,
    turn_id character varying(64) NOT NULL,
    request_id character varying(128) NOT NULL,
    seq integer NOT NULL,
    attempt integer NOT NULL,
    provider character varying(64),
    transport character varying(64),
    model character varying(128),
    body_bytes integer NOT NULL,
    body_sha256 character varying(64) NOT NULL,
    body_json text,
    body_state character varying(16) NOT NULL,
    captured_at timestamp with time zone NOT NULL
);


--
-- Name: turn_egress_requests_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.turn_egress_requests_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: turn_egress_requests_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.turn_egress_requests_id_seq OWNED BY public.turn_egress_requests.id;


--
-- Name: turn_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.turn_logs (
    id character varying NOT NULL,
    created_at timestamp with time zone NOT NULL,
    request_id character varying,
    path character varying(128) NOT NULL,
    stream boolean NOT NULL,
    model character varying,
    bot_id character varying,
    user_id character varying,
    status character varying(32) NOT NULL,
    latency_ms double precision,
    user_prompt text,
    request_json text,
    response_text text,
    tool_calls_json text,
    error_text text,
    trigger_message_id character varying,
    agent_session_key character varying(128),
    agent_request_id character varying(128),
    animation character varying(255),
    token_usage_json text,
    end_reason character varying(32),
    question_id character varying(128),
    parent_turn_id character varying(128),
    ended_at timestamp with time zone,
    reasoning text,
    tts_scrubbed boolean,
    assistant_message_id character varying,
    agent_backend character varying(64),
    agent_steer_caps character varying(64),
    task_ref character varying(128),
    task_attribution_source character varying(32),
    quota_snapshot_json text,
    terminal_protocol_state character varying(32),
    history_session_id character varying,
    follower_owner character varying,
    follower_lease_until timestamp with time zone,
    terminal_snapshot_json text,
    terminal_history_projected_at timestamp with time zone,
    terminal_publication_deadline timestamp with time zone,
    terminal_published_stream_id character varying,
    terminal_published_at timestamp with time zone
);


--
-- Name: utility_usage; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.utility_usage (
    id bigint NOT NULL,
    created_at timestamp with time zone NOT NULL,
    caller character varying(64) NOT NULL,
    bot_id character varying(64),
    model character varying(128),
    input_tokens integer DEFAULT 0 NOT NULL,
    cache_read_tokens integer DEFAULT 0 NOT NULL,
    cache_write_tokens integer DEFAULT 0 NOT NULL,
    output_tokens integer DEFAULT 0 NOT NULL,
    estimated boolean NOT NULL,
    elapsed_ms integer NOT NULL,
    outcome character varying(32) NOT NULL
);


--
-- Name: utility_usage_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.utility_usage_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: utility_usage_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.utility_usage_id_seq OWNED BY public.utility_usage.id;


--
-- Name: worker_lifecycle_records; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.worker_lifecycle_records (
    id integer NOT NULL,
    turn_id character varying(64) NOT NULL,
    task_id character varying(128) NOT NULL,
    tool_use_id character varying(128),
    bot_id character varying,
    user_id character varying,
    description text,
    task_type character varying(64),
    provider character varying(64),
    status character varying(16) NOT NULL,
    summary text,
    last_tool_name character varying(128),
    usage json,
    failure json,
    first_event_ts double precision,
    last_event_ts double precision,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL
);


--
-- Name: worker_lifecycle_records_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.worker_lifecycle_records_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: worker_lifecycle_records_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.worker_lifecycle_records_id_seq OWNED BY public.worker_lifecycle_records.id;


--
-- Name: working_memory_changes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.working_memory_changes (
    seq bigint NOT NULL,
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    ns_version bigint NOT NULL,
    item_id uuid NOT NULL,
    revision integer NOT NULL,
    kind character varying(16) NOT NULL,
    actor_kind character varying(8) NOT NULL,
    actor_id character varying(128) NOT NULL,
    session_id character varying(64),
    turn_id character varying(128),
    idempotency_key uuid,
    create_fingerprint character(64),
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT working_memory_changes_actor_kind_check CHECK (((actor_kind)::text = ANY ((ARRAY['bot'::character varying, 'user'::character varying])::text[]))),
    CONSTRAINT working_memory_changes_check CHECK ((((kind)::text = 'create'::text) OR ((idempotency_key IS NULL) AND (create_fingerprint IS NULL)))),
    CONSTRAINT working_memory_changes_kind_check CHECK (((kind)::text = ANY ((ARRAY['create'::character varying, 'update'::character varying, 'archive'::character varying, 'removed'::character varying, 'purged'::character varying])::text[])))
);


--
-- Name: working_memory_changes_seq_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.working_memory_changes_seq_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: working_memory_changes_seq_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.working_memory_changes_seq_seq OWNED BY public.working_memory_changes.seq;


--
-- Name: working_memory_items; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.working_memory_items (
    id uuid NOT NULL,
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    kind character varying(32) NOT NULL,
    initiative_task_id character varying(128),
    objective text NOT NULL,
    status character varying(16) NOT NULL,
    next_action text,
    blockers jsonb DEFAULT '[]'::jsonb NOT NULL,
    checkpoint text,
    priority integer,
    refs jsonb DEFAULT '[]'::jsonb NOT NULL,
    revision integer NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    archived_at timestamp with time zone,
    removed_at timestamp with time zone,
    canonical_task_id character varying(128),
    task_short_id character varying(64),
    task_title character varying(160),
    task_status character varying(32),
    CONSTRAINT working_memory_items_check CHECK ((((kind)::text = 'initiative_link'::text) = (initiative_task_id IS NOT NULL))),
    CONSTRAINT working_memory_items_check1 CHECK ((((status)::text = ANY ((ARRAY['done'::character varying, 'cancelled'::character varying])::text[])) = (archived_at IS NOT NULL))),
    CONSTRAINT working_memory_items_kind_check CHECK (((kind)::text = ANY ((ARRAY['personal'::character varying, 'initiative_link'::character varying])::text[]))),
    CONSTRAINT working_memory_items_status_check CHECK (((status)::text = ANY ((ARRAY['planned'::character varying, 'active'::character varying, 'blocked'::character varying, 'done'::character varying, 'cancelled'::character varying])::text[])))
);


--
-- Name: working_memory_namespaces; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.working_memory_namespaces (
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    version bigint DEFAULT 0 NOT NULL
);


--
-- Name: working_memory_task_purge_audit; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.working_memory_task_purge_audit (
    item_id uuid NOT NULL,
    task_digest character(64),
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: working_memory_task_suppressions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.working_memory_task_suppressions (
    bot_id character varying(128) NOT NULL,
    user_id character varying(128) NOT NULL,
    task_digest character(64) NOT NULL,
    key_version smallint DEFAULT 1 NOT NULL,
    key_check character(64),
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: access_paths id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.access_paths ALTER COLUMN id SET DEFAULT nextval('public.access_paths_id_seq'::regclass);


--
-- Name: bot_profile_changes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bot_profile_changes ALTER COLUMN id SET DEFAULT nextval('public.bot_profile_changes_id_seq'::regclass);


--
-- Name: bot_profiles id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bot_profiles ALTER COLUMN id SET DEFAULT nextval('public.bot_profiles_id_seq'::regclass);


--
-- Name: entity_profile_attributes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_profile_attributes ALTER COLUMN id SET DEFAULT nextval('public.entity_profile_attributes_id_seq'::regclass);


--
-- Name: entity_profiles id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_profiles ALTER COLUMN id SET DEFAULT nextval('public.entity_profiles_id_seq'::regclass);


--
-- Name: gpu_handoff_plans generation; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_handoff_plans ALTER COLUMN generation SET DEFAULT nextval('public.gpu_handoff_plans_generation_seq'::regclass);


--
-- Name: gpu_handoff_state id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_handoff_state ALTER COLUMN id SET DEFAULT nextval('public.gpu_handoff_state_id_seq'::regclass);


--
-- Name: gpu_video_calibrations generation; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_video_calibrations ALTER COLUMN generation SET DEFAULT nextval('public.gpu_video_calibrations_generation_seq'::regclass);


--
-- Name: guidance_proposals id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.guidance_proposals ALTER COLUMN id SET DEFAULT nextval('public.guidance_proposals_id_seq'::regclass);


--
-- Name: guidance_run_receipts id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.guidance_run_receipts ALTER COLUMN id SET DEFAULT nextval('public.guidance_run_receipts_id_seq'::regclass);


--
-- Name: home_audio_jobs sequence; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.home_audio_jobs ALTER COLUMN sequence SET DEFAULT nextval('public.home_audio_jobs_sequence_seq'::regclass);


--
-- Name: home_audio_lane id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.home_audio_lane ALTER COLUMN id SET DEFAULT nextval('public.home_audio_lane_id_seq'::regclass);


--
-- Name: inter_bot_deliveries ordinal; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inter_bot_deliveries ALTER COLUMN ordinal SET DEFAULT nextval('public.inter_bot_deliveries_ordinal_seq'::regclass);


--
-- Name: llm_response_records id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.llm_response_records ALTER COLUMN id SET DEFAULT nextval('public.llm_response_records_id_seq'::regclass);


--
-- Name: model_endpoints id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_endpoints ALTER COLUMN id SET DEFAULT nextval('public.model_endpoints_id_seq'::regclass);


--
-- Name: models id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.models ALTER COLUMN id SET DEFAULT nextval('public.models_id_seq'::regclass);


--
-- Name: ops_bell_outbox id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_bell_outbox ALTER COLUMN id SET DEFAULT nextval('public.ops_bell_outbox_id_seq'::regclass);


--
-- Name: ops_host_access id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_host_access ALTER COLUMN id SET DEFAULT nextval('public.ops_host_access_id_seq'::regclass);


--
-- Name: prompt_template_versions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_template_versions ALTER COLUMN id SET DEFAULT nextval('public.prompt_template_versions_id_seq'::regclass);


--
-- Name: prompt_templates id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_templates ALTER COLUMN id SET DEFAULT nextval('public.prompt_templates_id_seq'::regclass);


--
-- Name: runtime_settings id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.runtime_settings ALTER COLUMN id SET DEFAULT nextval('public.runtime_settings_id_seq'::regclass);


--
-- Name: tool_call_records id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_call_records ALTER COLUMN id SET DEFAULT nextval('public.tool_call_records_id_seq'::regclass);


--
-- Name: tool_call_result_payloads tool_call_record_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_call_result_payloads ALTER COLUMN tool_call_record_id SET DEFAULT nextval('public.tool_call_result_payloads_tool_call_record_id_seq'::regclass);


--
-- Name: turn_changed_files id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_changed_files ALTER COLUMN id SET DEFAULT nextval('public.turn_changed_files_id_seq'::regclass);


--
-- Name: turn_egress_requests id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_egress_requests ALTER COLUMN id SET DEFAULT nextval('public.turn_egress_requests_id_seq'::regclass);


--
-- Name: utility_usage id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.utility_usage ALTER COLUMN id SET DEFAULT nextval('public.utility_usage_id_seq'::regclass);


--
-- Name: worker_lifecycle_records id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worker_lifecycle_records ALTER COLUMN id SET DEFAULT nextval('public.worker_lifecycle_records_id_seq'::regclass);


--
-- Name: working_memory_changes seq; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.working_memory_changes ALTER COLUMN seq SET DEFAULT nextval('public.working_memory_changes_seq_seq'::regclass);


--
-- Name: access_paths access_paths_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.access_paths
    ADD CONSTRAINT access_paths_key_key UNIQUE (key);


--
-- Name: access_paths access_paths_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.access_paths
    ADD CONSTRAINT access_paths_pkey PRIMARY KEY (id);


--
-- Name: agent_content_lineage agent_content_lineage_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agent_content_lineage
    ADD CONSTRAINT agent_content_lineage_pkey PRIMARY KEY (target_id, source_id);


--
-- Name: agent_content_turns agent_content_turns_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agent_content_turns
    ADD CONSTRAINT agent_content_turns_pkey PRIMARY KEY (id);


--
-- Name: agent_content_write_decisions agent_content_write_decisions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agent_content_write_decisions
    ADD CONSTRAINT agent_content_write_decisions_pkey PRIMARY KEY (id);


--
-- Name: api_balance_ledger api_balance_ledger_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.api_balance_ledger
    ADD CONSTRAINT api_balance_ledger_pkey PRIMARY KEY (turn_id);


--
-- Name: api_balances api_balances_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.api_balances
    ADD CONSTRAINT api_balances_pkey PRIMARY KEY (provider);


--
-- Name: approval_approver_fanout approval_approver_fanout_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.approval_approver_fanout
    ADD CONSTRAINT approval_approver_fanout_pkey PRIMARY KEY (request_id, approver_bot_id);


--
-- Name: bawthub_auth_secrets bawthub_auth_secrets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bawthub_auth_secrets
    ADD CONSTRAINT bawthub_auth_secrets_pkey PRIMARY KEY (name);


--
-- Name: bot_profile_changes bot_profile_changes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bot_profile_changes
    ADD CONSTRAINT bot_profile_changes_pkey PRIMARY KEY (id);


--
-- Name: bot_profile_tags bot_profile_tags_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bot_profile_tags
    ADD CONSTRAINT bot_profile_tags_pkey PRIMARY KEY (bot_slug, tag_slug);


--
-- Name: bot_profiles bot_profiles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bot_profiles
    ADD CONSTRAINT bot_profiles_pkey PRIMARY KEY (id);


--
-- Name: bot_slug_aliases bot_slug_aliases_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bot_slug_aliases
    ADD CONSTRAINT bot_slug_aliases_pkey PRIMARY KEY (old_slug);


--
-- Name: bot_tag_catalog bot_tag_catalog_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bot_tag_catalog
    ADD CONSTRAINT bot_tag_catalog_pkey PRIMARY KEY (slug);


--
-- Name: bridge_run_accepts bridge_run_accepts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bridge_run_accepts
    ADD CONSTRAINT bridge_run_accepts_pkey PRIMARY KEY (backend, request_id);


--
-- Name: chat_pending_questions chat_pending_questions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_pending_questions
    ADD CONSTRAINT chat_pending_questions_pkey PRIMARY KEY (tool_use_id);


--
-- Name: commit_authoring_turns commit_authoring_turns_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.commit_authoring_turns
    ADD CONSTRAINT commit_authoring_turns_pkey PRIMARY KEY (repo_key, commit_sha);


--
-- Name: entity_profile_attributes entity_profile_attributes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_profile_attributes
    ADD CONSTRAINT entity_profile_attributes_pkey PRIMARY KEY (id);


--
-- Name: entity_profiles entity_profiles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_profiles
    ADD CONSTRAINT entity_profiles_pkey PRIMARY KEY (id);


--
-- Name: forgotten_messages forgotten_messages_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.forgotten_messages
    ADD CONSTRAINT forgotten_messages_pkey PRIMARY KEY (bot_id, id);


--
-- Name: gate_escalations gate_escalations_gate_id_kind_subject_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gate_escalations
    ADD CONSTRAINT gate_escalations_gate_id_kind_subject_key UNIQUE (gate_id, kind, subject);


--
-- Name: gate_escalations gate_escalations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gate_escalations
    ADD CONSTRAINT gate_escalations_pkey PRIMARY KEY (id);


--
-- Name: gate_members gate_members_gate_id_turn_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gate_members
    ADD CONSTRAINT gate_members_gate_id_turn_id_key UNIQUE (gate_id, turn_id);


--
-- Name: gate_members gate_members_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gate_members
    ADD CONSTRAINT gate_members_pkey PRIMARY KEY (id);


--
-- Name: google_home_probe_captures google_home_probe_captures_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.google_home_probe_captures
    ADD CONSTRAINT google_home_probe_captures_pkey PRIMARY KEY (request_key);


--
-- Name: google_home_probe_links google_home_probe_links_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.google_home_probe_links
    ADD CONSTRAINT google_home_probe_links_pkey PRIMARY KEY (link_key);


--
-- Name: google_home_probe_tokens google_home_probe_tokens_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.google_home_probe_tokens
    ADD CONSTRAINT google_home_probe_tokens_pkey PRIMARY KEY (digest);


--
-- Name: gpu_handoff_plans gpu_handoff_plans_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_handoff_plans
    ADD CONSTRAINT gpu_handoff_plans_pkey PRIMARY KEY (generation);


--
-- Name: gpu_handoff_resolved_video_jobs gpu_handoff_resolved_video_jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_handoff_resolved_video_jobs
    ADD CONSTRAINT gpu_handoff_resolved_video_jobs_pkey PRIMARY KEY (id);


--
-- Name: gpu_handoff_state gpu_handoff_state_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_handoff_state
    ADD CONSTRAINT gpu_handoff_state_pkey PRIMARY KEY (id);


--
-- Name: gpu_handoff_video_jobs gpu_handoff_video_jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_handoff_video_jobs
    ADD CONSTRAINT gpu_handoff_video_jobs_pkey PRIMARY KEY (id);


--
-- Name: gpu_video_calibrations gpu_video_calibrations_job_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_video_calibrations
    ADD CONSTRAINT gpu_video_calibrations_job_id_key UNIQUE (job_id);


--
-- Name: gpu_video_calibrations gpu_video_calibrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_video_calibrations
    ADD CONSTRAINT gpu_video_calibrations_pkey PRIMARY KEY (generation);


--
-- Name: guidance_proposals guidance_proposals_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.guidance_proposals
    ADD CONSTRAINT guidance_proposals_pkey PRIMARY KEY (id);


--
-- Name: guidance_run_receipts guidance_run_receipts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.guidance_run_receipts
    ADD CONSTRAINT guidance_run_receipts_pkey PRIMARY KEY (id);


--
-- Name: history_read_cursors history_read_cursors_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.history_read_cursors
    ADD CONSTRAINT history_read_cursors_pkey PRIMARY KEY (bot_id, user_id);


--
-- Name: home_audio_jobs home_audio_jobs_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.home_audio_jobs
    ADD CONSTRAINT home_audio_jobs_id_key UNIQUE (id);


--
-- Name: home_audio_jobs home_audio_jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.home_audio_jobs
    ADD CONSTRAINT home_audio_jobs_pkey PRIMARY KEY (sequence);


--
-- Name: home_audio_jobs home_audio_jobs_request_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.home_audio_jobs
    ADD CONSTRAINT home_audio_jobs_request_key_key UNIQUE (request_key);


--
-- Name: home_audio_lane home_audio_lane_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.home_audio_lane
    ADD CONSTRAINT home_audio_lane_pkey PRIMARY KEY (id);


--
-- Name: house_retirements house_retirements_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.house_retirements
    ADD CONSTRAINT house_retirements_pkey PRIMARY KEY (house);


--
-- Name: inter_bot_deliveries inter_bot_deliveries_ordinal_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inter_bot_deliveries
    ADD CONSTRAINT inter_bot_deliveries_ordinal_key UNIQUE (ordinal);


--
-- Name: inter_bot_deliveries inter_bot_deliveries_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inter_bot_deliveries
    ADD CONSTRAINT inter_bot_deliveries_pkey PRIMARY KEY (id);


--
-- Name: inter_bot_deliveries inter_bot_deliveries_turn_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inter_bot_deliveries
    ADD CONSTRAINT inter_bot_deliveries_turn_id_key UNIQUE (turn_id);


--
-- Name: inter_bot_deliveries inter_bot_deliveries_user_message_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inter_bot_deliveries
    ADD CONSTRAINT inter_bot_deliveries_user_message_id_key UNIQUE (user_message_id);


--
-- Name: job_runs job_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_runs
    ADD CONSTRAINT job_runs_pkey PRIMARY KEY (id);


--
-- Name: llm_response_records llm_response_records_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.llm_response_records
    ADD CONSTRAINT llm_response_records_pkey PRIMARY KEY (id);


--
-- Name: media_assets media_assets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.media_assets
    ADD CONSTRAINT media_assets_pkey PRIMARY KEY (id);


--
-- Name: media_generations media_generations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.media_generations
    ADD CONSTRAINT media_generations_pkey PRIMARY KEY (id);


--
-- Name: memories memories_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memories
    ADD CONSTRAINT memories_pkey PRIMARY KEY (bot_id, id);


--
-- Name: memory_ownership_activations memory_ownership_activations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memory_ownership_activations
    ADD CONSTRAINT memory_ownership_activations_pkey PRIMARY KEY (bot_id);


--
-- Name: message_author_backfill_audit message_author_backfill_audit_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.message_author_backfill_audit
    ADD CONSTRAINT message_author_backfill_audit_pkey PRIMARY KEY (batch_id, bot_id, message_id);


--
-- Name: message_route_eligibility message_route_eligibility_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.message_route_eligibility
    ADD CONSTRAINT message_route_eligibility_pkey PRIMARY KEY (owner_user_id, bot_id);


--
-- Name: message_routes message_routes_owner_user_id_actor_entity_type_actor_entity_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.message_routes
    ADD CONSTRAINT message_routes_owner_user_id_actor_entity_type_actor_entity_key UNIQUE (owner_user_id, actor_entity_type, actor_entity_id, idempotency_key);


--
-- Name: message_routes message_routes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.message_routes
    ADD CONSTRAINT message_routes_pkey PRIMARY KEY (id);


--
-- Name: messages messages_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_pkey PRIMARY KEY (bot_id, id);


--
-- Name: model_endpoints model_endpoints_model_access_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_endpoints
    ADD CONSTRAINT model_endpoints_model_access_key UNIQUE (model_id, access_path_id);


--
-- Name: model_endpoints model_endpoints_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_endpoints
    ADD CONSTRAINT model_endpoints_pkey PRIMARY KEY (id);


--
-- Name: models models_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT models_key_key UNIQUE (key);


--
-- Name: models models_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT models_pkey PRIMARY KEY (id);


--
-- Name: new_session_event_outbox new_session_event_outbox_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.new_session_event_outbox
    ADD CONSTRAINT new_session_event_outbox_pkey PRIMARY KEY (delivery_id);


--
-- Name: notify_log notify_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.notify_log
    ADD CONSTRAINT notify_log_pkey PRIMARY KEY (id);


--
-- Name: ops_bell_outbox ops_bell_outbox_event_key_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_bell_outbox
    ADD CONSTRAINT ops_bell_outbox_event_key_key UNIQUE (event_key);


--
-- Name: ops_bell_outbox ops_bell_outbox_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_bell_outbox
    ADD CONSTRAINT ops_bell_outbox_pkey PRIMARY KEY (id);


--
-- Name: ops_host_access ops_host_access_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_host_access
    ADD CONSTRAINT ops_host_access_pkey PRIMARY KEY (id);


--
-- Name: ops_jobs ops_jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_jobs
    ADD CONSTRAINT ops_jobs_pkey PRIMARY KEY (id);


--
-- Name: ops_operation_revisions ops_operation_revisions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_operation_revisions
    ADD CONSTRAINT ops_operation_revisions_pkey PRIMARY KEY (id);


--
-- Name: ops_operations ops_operations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_operations
    ADD CONSTRAINT ops_operations_pkey PRIMARY KEY (id);


--
-- Name: ops_release_events ops_release_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_release_events
    ADD CONSTRAINT ops_release_events_pkey PRIMARY KEY (id);


--
-- Name: ops_release_runs ops_release_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_release_runs
    ADD CONSTRAINT ops_release_runs_pkey PRIMARY KEY (id);


--
-- Name: ops_resume_notices ops_resume_notices_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_resume_notices
    ADD CONSTRAINT ops_resume_notices_pkey PRIMARY KEY (job_id, bot_id, kind);


--
-- Name: playground_artifacts playground_artifacts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.playground_artifacts
    ADD CONSTRAINT playground_artifacts_pkey PRIMARY KEY (id);


--
-- Name: prompt_occurrences prompt_occurrences_delivery_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_occurrences
    ADD CONSTRAINT prompt_occurrences_delivery_id_key UNIQUE (delivery_id);


--
-- Name: prompt_occurrences prompt_occurrences_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_occurrences
    ADD CONSTRAINT prompt_occurrences_pkey PRIMARY KEY (run_id);


--
-- Name: prompt_schedules prompt_schedules_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_schedules
    ADD CONSTRAINT prompt_schedules_pkey PRIMARY KEY (job_id);


--
-- Name: prompt_template_versions prompt_template_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_template_versions
    ADD CONSTRAINT prompt_template_versions_pkey PRIMARY KEY (id);


--
-- Name: prompt_templates prompt_templates_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_templates
    ADD CONSTRAINT prompt_templates_pkey PRIMARY KEY (id);


--
-- Name: quiesce_run_sightings quiesce_run_sightings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.quiesce_run_sightings
    ADD CONSTRAINT quiesce_run_sightings_pkey PRIMARY KEY (backend, request_id);


--
-- Name: recall_evidence_ledgers recall_evidence_ledgers_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.recall_evidence_ledgers
    ADD CONSTRAINT recall_evidence_ledgers_pkey PRIMARY KEY (handle_sha256);


--
-- Name: response_audio response_audio_assistant_message_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.response_audio
    ADD CONSTRAINT response_audio_assistant_message_id_key UNIQUE (assistant_message_id);


--
-- Name: response_audio response_audio_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.response_audio
    ADD CONSTRAINT response_audio_pkey PRIMARY KEY (id);


--
-- Name: response_audio response_audio_turn_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.response_audio
    ADD CONSTRAINT response_audio_turn_id_key UNIQUE (turn_id);


--
-- Name: runtime_settings runtime_settings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.runtime_settings
    ADD CONSTRAINT runtime_settings_pkey PRIMARY KEY (id);


--
-- Name: scheduled_jobs scheduled_jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scheduled_jobs
    ADD CONSTRAINT scheduled_jobs_pkey PRIMARY KEY (id);


--
-- Name: session_resumption_receipts session_resumption_receipts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.session_resumption_receipts
    ADD CONSTRAINT session_resumption_receipts_pkey PRIMARY KEY (bot_id, trigger_message_id);


--
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- Name: skill_assignments skill_assignments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.skill_assignments
    ADD CONSTRAINT skill_assignments_pkey PRIMARY KEY (scope_type, scope_id, skill_ref);


--
-- Name: skill_managed_bots skill_managed_bots_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.skill_managed_bots
    ADD CONSTRAINT skill_managed_bots_pkey PRIMARY KEY (bot_slug);


--
-- Name: system_events system_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.system_events
    ADD CONSTRAINT system_events_pkey PRIMARY KEY (id);


--
-- Name: system_gates system_gates_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.system_gates
    ADD CONSTRAINT system_gates_pkey PRIMARY KEY (id);


--
-- Name: talk_reply_subscriptions talk_reply_subscriptions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.talk_reply_subscriptions
    ADD CONSTRAINT talk_reply_subscriptions_pkey PRIMARY KEY (message_id);


--
-- Name: tool_approval_decisions tool_approval_decisions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_approval_decisions
    ADD CONSTRAINT tool_approval_decisions_pkey PRIMARY KEY (id);


--
-- Name: tool_approval_policies tool_approval_policies_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_approval_policies
    ADD CONSTRAINT tool_approval_policies_pkey PRIMARY KEY (id);


--
-- Name: tool_approval_policy_revisions tool_approval_policy_revisions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_approval_policy_revisions
    ADD CONSTRAINT tool_approval_policy_revisions_pkey PRIMARY KEY (id);


--
-- Name: tool_approval_requests tool_approval_requests_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_approval_requests
    ADD CONSTRAINT tool_approval_requests_pkey PRIMARY KEY (id);


--
-- Name: tool_call_records tool_call_records_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_call_records
    ADD CONSTRAINT tool_call_records_pkey PRIMARY KEY (id);


--
-- Name: tool_call_result_payloads tool_call_result_payloads_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_call_result_payloads
    ADD CONSTRAINT tool_call_result_payloads_pkey PRIMARY KEY (tool_call_record_id);


--
-- Name: turn_changed_files turn_changed_files_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_changed_files
    ADD CONSTRAINT turn_changed_files_pkey PRIMARY KEY (id);


--
-- Name: turn_diff_payloads turn_diff_payloads_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_diff_payloads
    ADD CONSTRAINT turn_diff_payloads_pkey PRIMARY KEY (sha256);


--
-- Name: turn_egress_requests turn_egress_requests_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_egress_requests
    ADD CONSTRAINT turn_egress_requests_pkey PRIMARY KEY (id);


--
-- Name: turn_logs turn_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_logs
    ADD CONSTRAINT turn_logs_pkey PRIMARY KEY (id);


--
-- Name: llm_response_records uq_llm_response_turn_response; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.llm_response_records
    ADD CONSTRAINT uq_llm_response_turn_response UNIQUE (turn_id, response_id);


--
-- Name: ops_jobs uq_ops_jobs_idempotency_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_jobs
    ADD CONSTRAINT uq_ops_jobs_idempotency_key UNIQUE (idempotency_key);


--
-- Name: ops_release_events uq_ops_release_event_sequence; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_release_events
    ADD CONSTRAINT uq_ops_release_event_sequence UNIQUE (release_run_id, sequence);


--
-- Name: ops_release_runs uq_ops_release_parent_job; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_release_runs
    ADD CONSTRAINT uq_ops_release_parent_job UNIQUE (parent_job_id);


--
-- Name: ops_release_runs uq_ops_release_request; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_release_runs
    ADD CONSTRAINT uq_ops_release_request UNIQUE (release_request_id);


--
-- Name: ops_operation_revisions uq_ops_revision; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ops_operation_revisions
    ADD CONSTRAINT uq_ops_revision UNIQUE (operation_id, version);


--
-- Name: prompt_occurrences uq_prompt_occurrence_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_occurrences
    ADD CONSTRAINT uq_prompt_occurrence_key UNIQUE (job_id, occurrence_key);


--
-- Name: prompt_templates uq_prompt_templates_scope_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_templates
    ADD CONSTRAINT uq_prompt_templates_scope_key UNIQUE (key, scope_type, scope_id);


--
-- Name: system_events uq_system_events_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.system_events
    ADD CONSTRAINT uq_system_events_key UNIQUE (event_key);


--
-- Name: turn_egress_requests uq_turn_egress_request_seq; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_egress_requests
    ADD CONSTRAINT uq_turn_egress_request_seq UNIQUE (request_id, seq);


--
-- Name: worker_lifecycle_records uq_worker_lifecycle_turn_task; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worker_lifecycle_records
    ADD CONSTRAINT uq_worker_lifecycle_turn_task UNIQUE (turn_id, task_id);


--
-- Name: utility_usage utility_usage_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.utility_usage
    ADD CONSTRAINT utility_usage_pkey PRIMARY KEY (id);


--
-- Name: worker_lifecycle_records worker_lifecycle_records_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worker_lifecycle_records
    ADD CONSTRAINT worker_lifecycle_records_pkey PRIMARY KEY (id);


--
-- Name: working_memory_changes working_memory_changes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.working_memory_changes
    ADD CONSTRAINT working_memory_changes_pkey PRIMARY KEY (seq);


--
-- Name: working_memory_items working_memory_items_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.working_memory_items
    ADD CONSTRAINT working_memory_items_pkey PRIMARY KEY (id);


--
-- Name: working_memory_namespaces working_memory_namespaces_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.working_memory_namespaces
    ADD CONSTRAINT working_memory_namespaces_pkey PRIMARY KEY (bot_id, user_id);


--
-- Name: working_memory_task_purge_audit working_memory_task_purge_audit_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.working_memory_task_purge_audit
    ADD CONSTRAINT working_memory_task_purge_audit_pkey PRIMARY KEY (item_id);


--
-- Name: working_memory_task_suppressions working_memory_task_suppressions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.working_memory_task_suppressions
    ADD CONSTRAINT working_memory_task_suppressions_pkey PRIMARY KEY (bot_id, user_id, key_version, task_digest);


--
-- Name: bot_profile_changes_bot_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX bot_profile_changes_bot_idx ON public.bot_profile_changes USING btree (bot_slug, created_at DESC);


--
-- Name: bot_profile_tags_tag_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX bot_profile_tags_tag_idx ON public.bot_profile_tags USING btree (tag_slug);


--
-- Name: bot_slug_aliases_bot_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX bot_slug_aliases_bot_idx ON public.bot_slug_aliases USING btree (bot_slug);


--
-- Name: bot_tag_catalog_prompt_key_key; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX bot_tag_catalog_prompt_key_key ON public.bot_tag_catalog USING btree (prompt_key) WHERE (prompt_key IS NOT NULL);


--
-- Name: idx_bot_profiles_endpoint_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_bot_profiles_endpoint_id ON public.bot_profiles USING btree (endpoint_id);


--
-- Name: idx_entity_profile_attr_entity; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_profile_attr_entity ON public.entity_profile_attributes USING btree (entity_type, entity_id, category);


--
-- Name: idx_entity_profile_unique; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_entity_profile_unique ON public.entity_profiles USING btree (entity_type, entity_id);


--
-- Name: idx_entity_profiles_email; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_profiles_email ON public.entity_profiles USING btree (email) WHERE (email IS NOT NULL);


--
-- Name: idx_job_runs_idle_pending; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_job_runs_idle_pending ON public.job_runs USING btree (job_id) WHERE (result_json ~~ '%"event_pending": true%'::text);


--
-- Name: idx_media_assets_expires_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_media_assets_expires_at ON public.media_assets USING btree (expires_at) WHERE (expires_at IS NOT NULL);


--
-- Name: idx_media_assets_immutable_sha; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_media_assets_immutable_sha ON public.media_assets USING btree (sha256) WHERE (storage_key IS NULL);


--
-- Name: idx_media_assets_owner_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_media_assets_owner_created ON public.media_assets USING btree (owner_user_id, created_at DESC);


--
-- Name: idx_media_assets_storage_key; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_media_assets_storage_key ON public.media_assets USING btree (storage_key);


--
-- Name: idx_media_generations_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_media_generations_created_at ON public.media_generations USING btree (created_at);


--
-- Name: idx_media_generations_media_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_media_generations_media_type ON public.media_generations USING btree (media_type);


--
-- Name: idx_media_generations_provider_job_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_media_generations_provider_job_id ON public.media_generations USING btree (provider_job_id);


--
-- Name: idx_media_generations_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_media_generations_status ON public.media_generations USING btree (status);


--
-- Name: idx_memories_accessed; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memories_accessed ON ONLY public.memories USING btree (last_accessed);


--
-- Name: idx_memories_embedding; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memories_embedding ON ONLY public.memories USING hnsw (embedding public.vector_cosine_ops);


--
-- Name: idx_memories_importance; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memories_importance ON ONLY public.memories USING btree (importance);


--
-- Name: idx_memories_meaning_embedding; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memories_meaning_embedding ON ONLY public.memories USING hnsw (meaning_embedding public.vector_cosine_ops);


--
-- Name: idx_memories_owner_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memories_owner_user ON ONLY public.memories USING btree (owner_user_id);


--
-- Name: idx_memories_tags_gin; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memories_tags_gin ON ONLY public.memories USING gin (tags);


--
-- Name: idx_messages_processed; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_messages_processed ON ONLY public.messages USING btree (processed);


--
-- Name: idx_messages_session_ts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_messages_session_ts ON ONLY public.messages USING btree (session_id, "timestamp" DESC);


--
-- Name: idx_messages_timestamp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_messages_timestamp ON ONLY public.messages USING btree ("timestamp");


--
-- Name: idx_model_endpoints_access_path_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_model_endpoints_access_path_id ON public.model_endpoints USING btree (access_path_id);


--
-- Name: idx_model_endpoints_model_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_model_endpoints_model_id ON public.model_endpoints USING btree (model_id);


--
-- Name: idx_playground_artifacts_opened_live; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_playground_artifacts_opened_live ON public.playground_artifacts USING btree (opened_at);


--
-- Name: idx_playground_artifacts_user_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_playground_artifacts_user_created ON public.playground_artifacts USING btree (user_id, created_at);


--
-- Name: idx_response_audio_asset; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_response_audio_asset ON public.response_audio USING btree (media_asset_id);


--
-- Name: idx_response_audio_owner; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_response_audio_owner ON public.response_audio USING btree (owner_user_id, created_at DESC);


--
-- Name: idx_runtime_settings_scope_key; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_runtime_settings_scope_key ON public.runtime_settings USING btree (scope_type, scope_id, key);


--
-- Name: idx_sessions_bot_started; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sessions_bot_started ON public.sessions USING btree (bot_id, started_at);


--
-- Name: idx_sessions_bot_user_active; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sessions_bot_user_active ON public.sessions USING btree (bot_id, user_id, status, started_at);


--
-- Name: idx_sessions_one_active_per_bot_user; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_sessions_one_active_per_bot_user ON public.sessions USING btree (bot_id, user_id) NULLS NOT DISTINCT WHERE ((status)::text = 'active'::text);


--
-- Name: idx_sessions_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sessions_status ON public.sessions USING btree (status);


--
-- Name: idx_system_events_outbox; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_system_events_outbox ON public.system_events USING btree (published_at, id);


--
-- Name: idx_system_events_user_id_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_system_events_user_id_id ON public.system_events USING btree (user_id, id);


--
-- Name: ix_api_balance_ledger_provider_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_api_balance_ledger_provider_created ON public.api_balance_ledger USING btree (provider, turn_created_at);


--
-- Name: ix_approval_approver_fanout_open; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_approval_approver_fanout_open ON public.approval_approver_fanout USING btree (request_id) WHERE (state = 'requested'::text);


--
-- Name: ix_bot_profiles_endpoint_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_bot_profiles_endpoint_id ON public.bot_profiles USING btree (endpoint_id);


--
-- Name: ix_bot_profiles_prompt_override_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_bot_profiles_prompt_override_id ON public.bot_profiles USING btree (prompt_override_id);


--
-- Name: ix_bot_profiles_slug; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ix_bot_profiles_slug ON public.bot_profiles USING btree (slug);


--
-- Name: ix_bridge_run_accepts_accepted_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_bridge_run_accepts_accepted_at ON public.bridge_run_accepts USING btree (accepted_at);


--
-- Name: ix_chat_pending_questions_bot_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_chat_pending_questions_bot_id ON public.chat_pending_questions USING btree (bot_id);


--
-- Name: ix_chat_pending_questions_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_chat_pending_questions_created_at ON public.chat_pending_questions USING btree (created_at);


--
-- Name: ix_chat_pending_questions_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_chat_pending_questions_status ON public.chat_pending_questions USING btree (status);


--
-- Name: ix_chat_pending_questions_trigger_message_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_chat_pending_questions_trigger_message_id ON public.chat_pending_questions USING btree (trigger_message_id);


--
-- Name: ix_chat_pending_questions_turn_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_chat_pending_questions_turn_id ON public.chat_pending_questions USING btree (turn_id);


--
-- Name: ix_chat_pending_questions_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_chat_pending_questions_user_id ON public.chat_pending_questions USING btree (user_id);


--
-- Name: ix_commit_authoring_turns_commit_sha; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_commit_authoring_turns_commit_sha ON public.commit_authoring_turns USING btree (commit_sha);


--
-- Name: ix_entity_profile_attributes_category; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_entity_profile_attributes_category ON public.entity_profile_attributes USING btree (category);


--
-- Name: ix_entity_profile_attributes_entity_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_entity_profile_attributes_entity_id ON public.entity_profile_attributes USING btree (entity_id);


--
-- Name: ix_entity_profile_attributes_entity_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_entity_profile_attributes_entity_type ON public.entity_profile_attributes USING btree (entity_type);


--
-- Name: ix_entity_profile_attributes_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_entity_profile_attributes_key ON public.entity_profile_attributes USING btree (key);


--
-- Name: ix_entity_profiles_email; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_entity_profiles_email ON public.entity_profiles USING btree (email);


--
-- Name: ix_entity_profiles_entity_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_entity_profiles_entity_id ON public.entity_profiles USING btree (entity_id);


--
-- Name: ix_entity_profiles_entity_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_entity_profiles_entity_type ON public.entity_profiles USING btree (entity_type);


--
-- Name: ix_gate_members_control_delivery; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_gate_members_control_delivery ON public.gate_members USING btree (control_delivery_id) WHERE (control_delivery_id IS NOT NULL);


--
-- Name: ix_gate_members_turn; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_gate_members_turn ON public.gate_members USING btree (turn_id);


--
-- Name: ix_google_home_probe_tokens_link_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_google_home_probe_tokens_link_key ON public.google_home_probe_tokens USING btree (link_key);


--
-- Name: ix_inter_bot_delivery_batch_head; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_inter_bot_delivery_batch_head ON public.inter_bot_deliveries USING btree (batch_head_id) WHERE (batch_head_id IS NOT NULL);


--
-- Name: ix_inter_bot_delivery_cancel_notice; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_inter_bot_delivery_cancel_notice ON public.inter_bot_deliveries USING btree (updated_at, ordinal) WHERE (((status)::text = 'CANCELLED'::text) AND (cancel_notified_at IS NULL) AND ((author_entity_type)::text = 'bot'::text) AND ((sender_bot_id)::text <> (target_bot_id)::text) AND ((message_kind)::text IS DISTINCT FROM 'DELIVERY_CANCELLED'::text) AND ((metadata_json -> 'prompt_schedule'::text) IS NULL));


--
-- Name: ix_inter_bot_delivery_control_head; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_inter_bot_delivery_control_head ON public.inter_bot_deliveries USING btree (target_bot_id, ordinal) WHERE (((lane)::text = 'control'::text) AND ((status)::text = 'QUEUED'::text));


--
-- Name: ix_inter_bot_delivery_control_run; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_inter_bot_delivery_control_run ON public.inter_bot_deliveries USING btree (target_bot_id, (((metadata_json -> 'control'::text) ->> 'run_key'::text))) WHERE ((lane)::text = 'control'::text);


--
-- Name: ix_inter_bot_delivery_dispatch; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_inter_bot_delivery_dispatch ON public.inter_bot_deliveries USING btree (target_bot_id, ordinal) WHERE ((status)::text = ANY ((ARRAY['QUEUED'::character varying, 'STEERING'::character varying, 'DISPATCHING'::character varying])::text[]));


--
-- Name: ix_inter_bot_delivery_restart_retry_pending; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_inter_bot_delivery_restart_retry_pending ON public.inter_bot_deliveries USING btree (updated_at, ordinal) WHERE (((status)::text = 'FAILED'::text) AND ((metadata_json ->> 'restart_retry_pending'::text) = 'true'::text));


--
-- Name: ix_inter_bot_delivery_retry; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_inter_bot_delivery_retry ON public.inter_bot_deliveries USING btree (available_at, ordinal) WHERE ((status)::text = 'QUEUED'::text);


--
-- Name: ix_inter_bot_delivery_stopped_notice; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_inter_bot_delivery_stopped_notice ON public.inter_bot_deliveries USING btree (updated_at, ordinal) WHERE (((status)::text = 'DELIVERED'::text) AND ((metadata_json ->> 'stopped_after_start'::text) = 'true'::text) AND (cancel_notified_at IS NULL));


--
-- Name: ix_job_runs_bot_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_job_runs_bot_id ON public.job_runs USING btree (bot_id);


--
-- Name: ix_job_runs_job_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_job_runs_job_id ON public.job_runs USING btree (job_id);


--
-- Name: ix_llm_response_records_turn_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_llm_response_records_turn_id ON public.llm_response_records USING btree (turn_id);


--
-- Name: ix_message_routes_pending; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_message_routes_pending ON public.message_routes USING btree (selected_bot_id) WHERE ((state)::text = 'queued'::text);


--
-- Name: ix_new_session_event_outbox_pending; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_new_session_event_outbox_pending ON public.new_session_event_outbox USING btree (created_at) WHERE (published_at IS NULL);


--
-- Name: ix_notify_log_bot_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_notify_log_bot_created ON public.notify_log USING btree (bot_id, created_at);


--
-- Name: ix_notify_log_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_notify_log_created ON public.notify_log USING btree (created_at);


--
-- Name: ix_ops_bell_outbox_job_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_bell_outbox_job_id ON public.ops_bell_outbox USING btree (job_id);


--
-- Name: ix_ops_bell_outbox_next_attempt_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_bell_outbox_next_attempt_at ON public.ops_bell_outbox USING btree (next_attempt_at);


--
-- Name: ix_ops_host_access_label; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ix_ops_host_access_label ON public.ops_host_access USING btree (label);


--
-- Name: ix_ops_jobs_approval_request_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_jobs_approval_request_id ON public.ops_jobs USING btree (approval_request_id);


--
-- Name: ix_ops_jobs_caller_bot_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_jobs_caller_bot_id ON public.ops_jobs USING btree (caller_bot_id);


--
-- Name: ix_ops_jobs_caller_turn_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_jobs_caller_turn_id ON public.ops_jobs USING btree (caller_turn_id);


--
-- Name: ix_ops_jobs_gate_attempt; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_jobs_gate_attempt ON public.ops_jobs USING btree (gate_plan_key) WHERE ((gate_plan_key IS NOT NULL) AND (((state)::text <> 'queued'::text) OR (dispatched_at IS NOT NULL)));


--
-- Name: ix_ops_jobs_operation_slug; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_jobs_operation_slug ON public.ops_jobs USING btree (operation_slug);


--
-- Name: ix_ops_jobs_state; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_jobs_state ON public.ops_jobs USING btree (state);


--
-- Name: ix_ops_jobs_submitted_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_jobs_submitted_at ON public.ops_jobs USING btree (submitted_at);


--
-- Name: ix_ops_operation_revisions_operation_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_operation_revisions_operation_id ON public.ops_operation_revisions USING btree (operation_id);


--
-- Name: ix_ops_operation_revisions_operation_slug; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_operation_revisions_operation_slug ON public.ops_operation_revisions USING btree (operation_slug);


--
-- Name: ix_ops_operations_category; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_operations_category ON public.ops_operations USING btree (category);


--
-- Name: ix_ops_operations_enabled; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_operations_enabled ON public.ops_operations USING btree (enabled);


--
-- Name: ix_ops_operations_slug; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ix_ops_operations_slug ON public.ops_operations USING btree (slug);


--
-- Name: ix_ops_release_events_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_events_created_at ON public.ops_release_events USING btree (created_at);


--
-- Name: ix_ops_release_events_event_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_events_event_type ON public.ops_release_events USING btree (event_type);


--
-- Name: ix_ops_release_events_release_run_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_events_release_run_id ON public.ops_release_events USING btree (release_run_id);


--
-- Name: ix_ops_release_events_state; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_events_state ON public.ops_release_events USING btree (state);


--
-- Name: ix_ops_release_runs_claim_expires_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_runs_claim_expires_at ON public.ops_release_runs USING btree (claim_expires_at);


--
-- Name: ix_ops_release_runs_claim_token; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_runs_claim_token ON public.ops_release_runs USING btree (claim_token);


--
-- Name: ix_ops_release_runs_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_runs_created_at ON public.ops_release_runs USING btree (created_at);


--
-- Name: ix_ops_release_runs_deploy_approval_request_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_runs_deploy_approval_request_id ON public.ops_release_runs USING btree (deploy_approval_request_id);


--
-- Name: ix_ops_release_runs_deploy_job_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_runs_deploy_job_id ON public.ops_release_runs USING btree (deploy_job_id);


--
-- Name: ix_ops_release_runs_github_run_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_runs_github_run_id ON public.ops_release_runs USING btree (github_run_id);


--
-- Name: ix_ops_release_runs_next_check_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_runs_next_check_at ON public.ops_release_runs USING btree (next_check_at);


--
-- Name: ix_ops_release_runs_parent_job_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_runs_parent_job_id ON public.ops_release_runs USING btree (parent_job_id);


--
-- Name: ix_ops_release_runs_release_request_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_runs_release_request_id ON public.ops_release_runs USING btree (release_request_id);


--
-- Name: ix_ops_release_runs_state; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_runs_state ON public.ops_release_runs USING btree (state);


--
-- Name: ix_ops_release_runs_updated_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ops_release_runs_updated_at ON public.ops_release_runs USING btree (updated_at);


--
-- Name: ix_pending_questions_bot_status_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_pending_questions_bot_status_created ON public.chat_pending_questions USING btree (bot_id, user_id, status, created_at DESC);


--
-- Name: ix_prompt_occurrences_job_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_occurrences_job_created ON public.prompt_occurrences USING btree (job_id, created_at);


--
-- Name: ix_prompt_occurrences_job_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_occurrences_job_id ON public.prompt_occurrences USING btree (job_id);


--
-- Name: ix_prompt_occurrences_state; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_occurrences_state ON public.prompt_occurrences USING btree (state);


--
-- Name: ix_prompt_schedules_owner_lifecycle; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_schedules_owner_lifecycle ON public.prompt_schedules USING btree (owner_user_id, lifecycle);


--
-- Name: ix_prompt_schedules_owner_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_schedules_owner_user_id ON public.prompt_schedules USING btree (owner_user_id);


--
-- Name: ix_prompt_template_versions_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_template_versions_key ON public.prompt_template_versions USING btree (key);


--
-- Name: ix_prompt_template_versions_scope_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_template_versions_scope_id ON public.prompt_template_versions USING btree (scope_id);


--
-- Name: ix_prompt_template_versions_scope_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_template_versions_scope_type ON public.prompt_template_versions USING btree (scope_type);


--
-- Name: ix_prompt_templates_category; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_templates_category ON public.prompt_templates USING btree (category);


--
-- Name: ix_prompt_templates_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_templates_key ON public.prompt_templates USING btree (key);


--
-- Name: ix_prompt_templates_scope_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_templates_scope_id ON public.prompt_templates USING btree (scope_id);


--
-- Name: ix_prompt_templates_scope_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_prompt_templates_scope_type ON public.prompt_templates USING btree (scope_type);


--
-- Name: ix_quiesce_run_sightings_last_seen; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_quiesce_run_sightings_last_seen ON public.quiesce_run_sightings USING btree (last_seen_at);


--
-- Name: ix_runtime_settings_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_runtime_settings_key ON public.runtime_settings USING btree (key);


--
-- Name: ix_runtime_settings_scope_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_runtime_settings_scope_id ON public.runtime_settings USING btree (scope_id);


--
-- Name: ix_runtime_settings_scope_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_runtime_settings_scope_type ON public.runtime_settings USING btree (scope_type);


--
-- Name: ix_scheduled_jobs_bot_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_scheduled_jobs_bot_id ON public.scheduled_jobs USING btree (bot_id);


--
-- Name: ix_scheduled_jobs_job_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_scheduled_jobs_job_type ON public.scheduled_jobs USING btree (job_type);


--
-- Name: ix_system_gates_owner; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_system_gates_owner ON public.system_gates USING btree (owner_id);


--
-- Name: ix_system_gates_unreleased; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_system_gates_unreleased ON public.system_gates USING btree (expires_at) WHERE (released_at IS NULL);


--
-- Name: ix_talk_reply_subscriptions_ready; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_talk_reply_subscriptions_ready ON public.talk_reply_subscriptions USING btree (next_check_at) WHERE ((state)::text = 'PENDING'::text);


--
-- Name: ix_tool_approval_decisions_policy_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_decisions_policy_id ON public.tool_approval_decisions USING btree (policy_id);


--
-- Name: ix_tool_approval_policies_backend_scope; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_policies_backend_scope ON public.tool_approval_policies USING btree (backend_scope);


--
-- Name: ix_tool_approval_policies_enabled; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_policies_enabled ON public.tool_approval_policies USING btree (enabled);


--
-- Name: ix_tool_approval_policies_order_index; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_policies_order_index ON public.tool_approval_policies USING btree (order_index);


--
-- Name: ix_tool_approval_policies_tool_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_policies_tool_name ON public.tool_approval_policies USING btree (tool_name);


--
-- Name: ix_tool_approval_policy_revisions_policy_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_policy_revisions_policy_id ON public.tool_approval_policy_revisions USING btree (policy_id);


--
-- Name: ix_tool_approval_requests_bot_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_bot_id ON public.tool_approval_requests USING btree (bot_id);


--
-- Name: ix_tool_approval_requests_continuation_next_attempt_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_continuation_next_attempt_at ON public.tool_approval_requests USING btree (continuation_next_attempt_at);


--
-- Name: ix_tool_approval_requests_continuation_state; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_continuation_state ON public.tool_approval_requests USING btree (continuation_state);


--
-- Name: ix_tool_approval_requests_execution_state; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_execution_state ON public.tool_approval_requests USING btree (execution_state);


--
-- Name: ix_tool_approval_requests_grant_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_grant_key ON public.tool_approval_requests USING btree (grant_key);


--
-- Name: ix_tool_approval_requests_policy_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_policy_id ON public.tool_approval_requests USING btree (policy_id);


--
-- Name: ix_tool_approval_requests_request_kind; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_request_kind ON public.tool_approval_requests USING btree (request_kind);


--
-- Name: ix_tool_approval_requests_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_status ON public.tool_approval_requests USING btree (status);


--
-- Name: ix_tool_approval_requests_tool_use_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_tool_use_id ON public.tool_approval_requests USING btree (tool_use_id);


--
-- Name: ix_tool_approval_requests_trigger_message_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_trigger_message_id ON public.tool_approval_requests USING btree (trigger_message_id);


--
-- Name: ix_tool_approval_requests_turn_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_turn_id ON public.tool_approval_requests USING btree (turn_id);


--
-- Name: ix_tool_approval_requests_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_approval_requests_user_id ON public.tool_approval_requests USING btree (user_id);


--
-- Name: ix_tool_call_records_approval_request_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_call_records_approval_request_id ON public.tool_call_records USING btree (approval_request_id);


--
-- Name: ix_tool_call_records_blob_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_call_records_blob_key ON public.tool_call_records USING btree (result_blob_key);


--
-- Name: ix_tool_call_records_bot_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_call_records_bot_id ON public.tool_call_records USING btree (bot_id);


--
-- Name: ix_tool_call_records_call_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_call_records_call_id ON public.tool_call_records USING btree (call_id);


--
-- Name: ix_tool_call_records_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_call_records_created_at ON public.tool_call_records USING btree (created_at);


--
-- Name: ix_tool_call_records_file_path_trgm; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_call_records_file_path_trgm ON public.tool_call_records USING gin ("substring"(arguments_json, '^\{(?:"[a-z_]+": (?:true|false|null|-?[0-9]+|"[^"\\]*"), )*"file_path": "([^"\\]+)"'::text) public.gin_trgm_ops) WHERE ((tool_name)::text = ANY ((ARRAY['Read'::character varying, 'Write'::character varying, 'Edit'::character varying])::text[]));


--
-- Name: ix_tool_call_records_parent_tool_use_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_call_records_parent_tool_use_id ON public.tool_call_records USING btree (parent_tool_use_id);


--
-- Name: ix_tool_call_records_tool_use_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_call_records_tool_use_id ON public.tool_call_records USING btree (tool_use_id);


--
-- Name: ix_tool_call_records_turn_call; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_call_records_turn_call ON public.tool_call_records USING btree (turn_id, call_id);


--
-- Name: ix_tool_call_records_turn_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_call_records_turn_id ON public.tool_call_records USING btree (turn_id);


--
-- Name: ix_tool_call_records_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tool_call_records_user_id ON public.tool_call_records USING btree (user_id);


--
-- Name: ix_turn_changed_files_bot_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_changed_files_bot_id ON public.turn_changed_files USING btree (bot_id);


--
-- Name: ix_turn_changed_files_commit_requested_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_changed_files_commit_requested_at ON public.turn_changed_files USING btree (commit_requested_at);


--
-- Name: ix_turn_changed_files_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_changed_files_created_at ON public.turn_changed_files USING btree (created_at);


--
-- Name: ix_turn_changed_files_owner_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_changed_files_owner_created ON public.turn_changed_files USING btree (bot_id, user_id, created_at);


--
-- Name: ix_turn_changed_files_trigger_message_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_changed_files_trigger_message_id ON public.turn_changed_files USING btree (trigger_message_id);


--
-- Name: ix_turn_changed_files_turn_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_changed_files_turn_id ON public.turn_changed_files USING btree (turn_id);


--
-- Name: ix_turn_changed_files_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_changed_files_user_id ON public.turn_changed_files USING btree (user_id);


--
-- Name: ix_turn_egress_requests_captured_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_egress_requests_captured_at ON public.turn_egress_requests USING btree (captured_at);


--
-- Name: ix_turn_egress_requests_turn_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_egress_requests_turn_id ON public.turn_egress_requests USING btree (turn_id);


--
-- Name: ix_turn_logs_active; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_active ON public.turn_logs USING btree (bot_id, created_at) WHERE (ended_at IS NULL);


--
-- Name: ix_turn_logs_agent_request_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_agent_request_id ON public.turn_logs USING btree (agent_request_id);


--
-- Name: ix_turn_logs_agent_session_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_agent_session_key ON public.turn_logs USING btree (agent_session_key);


--
-- Name: ix_turn_logs_assistant_message_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_assistant_message_id ON public.turn_logs USING btree (assistant_message_id);


--
-- Name: ix_turn_logs_bot_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_bot_id ON public.turn_logs USING btree (bot_id);


--
-- Name: ix_turn_logs_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_created_at ON public.turn_logs USING btree (created_at);


--
-- Name: ix_turn_logs_ended_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_ended_at ON public.turn_logs USING btree (ended_at);


--
-- Name: ix_turn_logs_model; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_model ON public.turn_logs USING btree (model);


--
-- Name: ix_turn_logs_parent_turn_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_parent_turn_id ON public.turn_logs USING btree (parent_turn_id);


--
-- Name: ix_turn_logs_question_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_question_id ON public.turn_logs USING btree (question_id);


--
-- Name: ix_turn_logs_request_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_request_id ON public.turn_logs USING btree (request_id);


--
-- Name: ix_turn_logs_stream; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_stream ON public.turn_logs USING btree (stream);


--
-- Name: ix_turn_logs_task_ref; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_task_ref ON public.turn_logs USING btree (task_ref);


--
-- Name: ix_turn_logs_terminal_recovery; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_terminal_recovery ON public.turn_logs USING btree (terminal_protocol_state, follower_lease_until) WHERE (terminal_protocol_state IS NOT NULL);


--
-- Name: ix_turn_logs_trigger_message_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_trigger_message_id ON public.turn_logs USING btree (trigger_message_id);


--
-- Name: ix_turn_logs_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_turn_logs_user_id ON public.turn_logs USING btree (user_id);


--
-- Name: ix_utility_usage_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_utility_usage_created ON public.utility_usage USING btree (created_at);


--
-- Name: ix_worker_lifecycle_records_turn_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_worker_lifecycle_records_turn_id ON public.worker_lifecycle_records USING btree (turn_id);


--
-- Name: ix_working_memory_changes_item; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_working_memory_changes_item ON public.working_memory_changes USING btree (item_id);


--
-- Name: ix_working_memory_changes_ns_version; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_working_memory_changes_ns_version ON public.working_memory_changes USING btree (bot_id, user_id, ns_version);


--
-- Name: ix_working_memory_items_namespace; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_working_memory_items_namespace ON public.working_memory_items USING btree (bot_id, user_id);


--
-- Name: messages_content_trgm_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX messages_content_trgm_idx ON ONLY public.messages USING gin (content public.gin_trgm_ops);


--
-- Name: messages_content_tsv_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX messages_content_tsv_idx ON ONLY public.messages USING gin (content_tsv);


--
-- Name: recall_evidence_ledgers_expires_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX recall_evidence_ledgers_expires_idx ON public.recall_evidence_ledgers USING btree (expires_at);


--
-- Name: uq_inter_bot_delivery_idempotency; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_inter_bot_delivery_idempotency ON public.inter_bot_deliveries USING btree (author_entity_type, author_entity_id, sender_bot_id, target_bot_id, idempotency_key) WHERE (idempotency_key IS NOT NULL);


--
-- Name: uq_prompt_occurrences_active; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_prompt_occurrences_active ON public.prompt_occurrences USING btree (job_id) WHERE ((state)::text = ANY ((ARRAY['pending'::character varying, 'queued'::character varying, 'running'::character varying])::text[]));


--
-- Name: ux_turn_changed_files_turn_repo_path; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ux_turn_changed_files_turn_repo_path ON public.turn_changed_files USING btree (turn_id, repo_key, path);


--
-- Name: ux_working_memory_changes_create_key; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ux_working_memory_changes_create_key ON public.working_memory_changes USING btree (bot_id, user_id, idempotency_key) WHERE ((kind)::text = 'create'::text);


--
-- Name: ux_working_memory_live_task_links; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ux_working_memory_live_task_links ON public.working_memory_items USING btree (bot_id, user_id, canonical_task_id) WHERE (((kind)::text = 'initiative_link'::text) AND (canonical_task_id IS NOT NULL) AND (removed_at IS NULL) AND ((status)::text = ANY ((ARRAY['planned'::character varying, 'active'::character varying, 'blocked'::character varying])::text[])));


--
-- Name: bot_profiles bot_profiles_model_endpoint_check; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER bot_profiles_model_endpoint_check BEFORE INSERT OR UPDATE ON public.bot_profiles FOR EACH ROW EXECUTE FUNCTION public.bot_profiles_check_model_endpoint();


--
-- Name: bot_slug_aliases bot_slug_aliases_guard; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER bot_slug_aliases_guard BEFORE INSERT OR UPDATE OF old_slug ON public.bot_slug_aliases FOR EACH ROW EXECUTE FUNCTION public.bot_slug_aliases_guard();


--
-- Name: inter_bot_deliveries inter_bot_deliveries_batch_settle; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER inter_bot_deliveries_batch_settle BEFORE UPDATE OF status ON public.inter_bot_deliveries FOR EACH ROW EXECUTE FUNCTION public.inter_bot_deliveries_settle_batch();


--
-- Name: inter_bot_deliveries inter_bot_deliveries_revision; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER inter_bot_deliveries_revision BEFORE UPDATE ON public.inter_bot_deliveries FOR EACH ROW EXECUTE FUNCTION public.inter_bot_deliveries_bump_revision();


--
-- Name: messages messages_content_tsv_fill; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER messages_content_tsv_fill BEFORE INSERT OR UPDATE OF content ON public.messages FOR EACH ROW EXECUTE FUNCTION public.messages_content_tsv_fill();


--
-- Name: bot_profile_tags bot_profile_tags_tag_slug_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bot_profile_tags
    ADD CONSTRAINT bot_profile_tags_tag_slug_fkey FOREIGN KEY (tag_slug) REFERENCES public.bot_tag_catalog(slug) ON UPDATE RESTRICT ON DELETE RESTRICT;


--
-- Name: bot_profiles bot_profiles_endpoint_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bot_profiles
    ADD CONSTRAINT bot_profiles_endpoint_fk FOREIGN KEY (endpoint_id) REFERENCES public.model_endpoints(id) ON UPDATE CASCADE ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED;


--
-- Name: bot_slug_aliases bot_slug_aliases_bot_slug_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bot_slug_aliases
    ADD CONSTRAINT bot_slug_aliases_bot_slug_fkey FOREIGN KEY (bot_slug) REFERENCES public.bot_profiles(slug) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: bot_profiles fk_bot_profiles_last_session; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bot_profiles
    ADD CONSTRAINT fk_bot_profiles_last_session FOREIGN KEY (last_session_id) REFERENCES public.sessions(id) ON DELETE SET NULL NOT VALID;


--
-- Name: messages fk_messages_session; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE public.messages
    ADD CONSTRAINT fk_messages_session FOREIGN KEY (session_id) REFERENCES public.sessions(id);


--
-- Name: tool_call_records fk_toolcalls_turn; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tool_call_records
    ADD CONSTRAINT fk_toolcalls_turn FOREIGN KEY (turn_id) REFERENCES public.turn_logs(id) ON DELETE SET NULL;


--
-- Name: gate_escalations gate_escalations_gate_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gate_escalations
    ADD CONSTRAINT gate_escalations_gate_id_fkey FOREIGN KEY (gate_id) REFERENCES public.system_gates(id) ON DELETE CASCADE;


--
-- Name: gate_members gate_members_gate_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gate_members
    ADD CONSTRAINT gate_members_gate_id_fkey FOREIGN KEY (gate_id) REFERENCES public.system_gates(id);


--
-- Name: job_runs job_runs_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_runs
    ADD CONSTRAINT job_runs_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.scheduled_jobs(id);


--
-- Name: model_endpoints model_endpoints_access_path_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_endpoints
    ADD CONSTRAINT model_endpoints_access_path_id_fkey FOREIGN KEY (access_path_id) REFERENCES public.access_paths(id) ON DELETE RESTRICT;


--
-- Name: model_endpoints model_endpoints_model_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_endpoints
    ADD CONSTRAINT model_endpoints_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.models(id) ON DELETE CASCADE;


--
-- Name: prompt_occurrences prompt_occurrences_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_occurrences
    ADD CONSTRAINT prompt_occurrences_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.scheduled_jobs(id);


--
-- Name: prompt_occurrences prompt_occurrences_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_occurrences
    ADD CONSTRAINT prompt_occurrences_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.job_runs(id);


--
-- Name: prompt_schedules prompt_schedules_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_schedules
    ADD CONSTRAINT prompt_schedules_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.scheduled_jobs(id);


--
-- PostgreSQL database dump complete
--


