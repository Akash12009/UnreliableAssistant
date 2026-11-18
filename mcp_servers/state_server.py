"""MCP server `state-store`: session facts, items, hash-chained audit, Tier-3 HITL mutations."""
from _common import run  # noqa: F401

from mcp.server.fastmcp import FastMCP

from ua.contracts import (AuditAppendIn, AuditVerifyIn, GetFactsIn, HitlDecideIn, HitlListIn, LookupItemIn, Meta, SessionFactIn, SubmitFormIn,
                          TransitionIn)
from ua.services import StateService

mcp = FastMCP("state-store")
svc = StateService()


def _c(tool, args, meta):
    return svc.call(tool, args.model_dump(), meta.model_dump())


@mcp.tool()
def lookup_item(args: LookupItemIn, meta: Meta) -> dict:
    return _c("lookup_item", args, meta)


@mcp.tool()
def get_session_facts(args: GetFactsIn, meta: Meta) -> dict:
    return _c("get_session_facts", args, meta)


@mcp.tool()
def put_session_fact(args: SessionFactIn, meta: Meta) -> dict:
    return _c("put_session_fact", args, meta)


@mcp.tool(description="TIER_3_MUTATE: never executes directly; returns PENDING_APPROVAL.")
def submit_form(args: SubmitFormIn, meta: Meta) -> dict:
    return _c("submit_form", args, meta)


@mcp.tool(description="TIER_3_MUTATE: never executes directly; returns PENDING_APPROVAL.")
def transition_workflow(args: TransitionIn, meta: Meta) -> dict:
    return _c("transition_workflow", args, meta)


@mcp.tool()
def audit_append(args: AuditAppendIn, meta: Meta) -> dict:
    return _c("audit_append", args, meta)


@mcp.tool(description="Reviewer-only: approve/reject a pending Tier-3 request.")
def hitl_decide(args: HitlDecideIn, meta: Meta) -> dict:
    return _c("hitl_decide", args, meta)


@mcp.tool(description="Reviewer-only: list HITL approvals for the tenant.")
def hitl_list(args: HitlListIn, meta: Meta) -> dict:
    return _c("hitl_list", args, meta)


@mcp.tool(description="Verify the SHA-256 hash chain of the tenant audit log.")
def audit_verify(args: AuditVerifyIn, meta: Meta) -> dict:
    return _c("audit_verify", args, meta)


if __name__ == "__main__":
    run(mcp, 8081)
