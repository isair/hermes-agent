"""hermes-agent-matrix: Matrix platform adapter for Hermes Agent."""

from hermes_agent_matrix.adapter import (  # noqa: F401
    MatrixAdapter,
    check_matrix_requirements,
    RoomID,
    _MatrixApprovalPrompt,
    _check_e2ee_deps,
    _CryptoStateStore,
    _create_matrix_session,
)


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group."""
    from hermes_agent_matrix.adapter import (
        MatrixAdapter,
        check_matrix_requirements,
    )

    ctx.register_platform(
        name="matrix",
        label="Matrix",
        adapter_factory=lambda cfg: MatrixAdapter(cfg),
        check_fn=check_matrix_requirements,
        install_hint="pip install 'mautrix[encryption]'",
        emoji="🟢",
    )

    ctx.register_platform_entry(
        name="matrix",
        adapter_class=MatrixAdapter,
        check_requirements=check_matrix_requirements,
    )
