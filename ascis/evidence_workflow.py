"""Collect public evidence with the local MCP workflow."""

import anyio

from ascis.evidence import evidence_documents, evidence_summary, local_extract
from ascis.research_mcp import request_sources


async def research_event(event: dict) -> dict:
    try:
        with anyio.fail_after(18):
            discovery = await request_sources(event["query"])
    except (TimeoutError, RuntimeError, OSError, ValueError, ExceptionGroup):
        discovery = {"providers": {"MCP": "unavailable"}}
    documents = evidence_documents(discovery.get("sources", []))
    facts = [fact for document in documents for fact in local_extract(document)]
    return {**discovery, "evidence": evidence_summary(event, discovery, facts)}
