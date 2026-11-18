"""Pluggy hook specifications for lifecycle plugins (design.md 1.2 'Plugin contract')."""
import pluggy

hookspec = pluggy.HookspecMarker("ua")
hookimpl = pluggy.HookimplMarker("ua")


class LifecycleSpec:
    @hookspec
    def pre_execution(self, ctx):
        """Sanitise/inspect the request context. Raise Block to reject."""

    @hookspec
    def tool_dispatch(self, call):
        """Inspect a tool call just before dispatch. Raise Block to reject."""

    @hookspec
    def tool_result(self, call, result):
        """Observe a tool result (breaker bookkeeping, indirect-injection scan)."""

    @hookspec
    def post_execution(self, ctx, draft):
        """Audit a draft answer; may veto by setting draft.veto."""
