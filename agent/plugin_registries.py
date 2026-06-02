"""Plugin capability registries.

Each plugin's ``register(ctx)`` function populates these registries via
``ctx.register_<capability>()``.  The core codebase then queries the
registries instead of importing from plugin packages directly.

This is the **only** coupling point between the core and plugins: the core
imports from ``agent.plugin_registries``, never from ``hermes_agent_*``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    runtime_checkable,
)


# ---------------------------------------------------------------------------
# Auth providers
# ---------------------------------------------------------------------------

@runtime_checkable
class AuthProvider(Protocol):
    """A plugin that can provide or check authentication credentials.

    Registered via ``ctx.register_auth_provider(name, provider)``.
    Queried by ``hermes_cli/auth_commands.py``, ``doctor.py``, etc.
    """

    @property
    def name(self) -> str: ...

    def has_credentials(self) -> bool:
        """Return True if the required credentials are present in env/config."""
        ...

    def check_env_vars(self) -> Dict[str, str | None]:
        """Return a dict of env-var-name → current-value (or None if unset).

        Used by ``hermes doctor`` to display credential status.
        """
        ...

    def resolve_token(self, **kwargs: Any) -> Any:
        """Resolve and return an auth token/credential for the provider.

        The return type is provider-specific (string, tuple, object, etc.).
        """
        ...

    def refresh_token(self, **kwargs: Any) -> Any:
        """Refresh an existing token.  Raises if refresh is not supported."""
        ...


@dataclass
class AuthProviderEntry:
    provider: AuthProvider
    """The auth provider instance."""

    cli_group: str = ""
    """CLI argument group name (e.g. 'Anthropic', 'AWS / Bedrock')."""

    setup_subcommands: bool = False
    """Whether this provider adds CLI auth subcommands (login, logout, etc.)."""


# ---------------------------------------------------------------------------
# Transport builders
# ---------------------------------------------------------------------------

@runtime_checkable
class TransportBuilder(Protocol):
    """A plugin that builds clients and converts messages for a model transport.

    Registered via ``ctx.register_transport(name, builder)``.
    Queried by ``agent/transports/`` and ``agent/auxiliary_client.py``.
    """

    def build_client(self, **kwargs: Any) -> Any:
        """Build and return a provider-specific API client."""
        ...

    def build_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Build the kwargs dict for a provider-specific API call."""
        ...

    def convert_messages(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        """Convert internal message format to provider-specific format."""
        ...

    def convert_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        """Convert internal tool format to provider-specific format."""
        ...

    def normalize_response(self, response: Any, **kwargs: Any) -> Any:
        """Normalize a provider-specific response into the internal format."""
        ...


# ---------------------------------------------------------------------------
# Platform adapters
# ---------------------------------------------------------------------------

@dataclass
class PlatformAdapterEntry:
    """A registered platform adapter.

    Registered via ``ctx.register_platform(name, entry)``.
    Queried by ``gateway/run.py`` and ``tools/send_message_tool.py``.
    """
    name: str
    """Platform identifier (e.g. 'telegram', 'slack')."""

    adapter_class: Type
    """The adapter class (e.g. TelegramAdapter)."""

    check_requirements: Callable[[], bool]
    """Check if the platform's dependencies are installed and configured."""

    available_flag: str = ""
    """Name of the module-level AVAILABLE boolean, if any."""

    constants: Dict[str, Any] = field(default_factory=dict)
    """Platform-specific constants (e.g. FEISHU_DOMAIN, LARK_DOMAIN)."""

    helper_functions: Dict[str, Callable] = field(default_factory=dict)
    """Platform-specific helper functions (e.g. probe_bot, qr_register)."""


# ---------------------------------------------------------------------------
# Tool providers
# ---------------------------------------------------------------------------

@dataclass
class ToolProviderEntry:
    """A registered tool provider.

    Registered via ``ctx.register_tool_provider(name, entry)``.
    Queried by ``tools/`` modules.
    """
    name: str
    """Tool identifier (e.g. 'tts', 'stt', 'fal', 'daytona')."""

    tool_functions: Dict[str, Callable] = field(default_factory=dict)
    """Tool functions keyed by name (e.g. 'text_to_speech_tool', 'transcribe_audio')."""

    check_fn: Optional[Callable] = None
    """Check if the tool's dependencies are available."""

    constants: Dict[str, Any] = field(default_factory=dict)
    """Tool-specific constants (e.g. MAX_FILE_SIZE)."""

    config_functions: Dict[str, Callable] = field(default_factory=dict)
    """Config/utility functions (e.g. _get_provider, _load_stt_config)."""

    environment_classes: Dict[str, Type] = field(default_factory=dict)
    """Environment classes for terminal backends (e.g. DaytonaEnvironment)."""


# ---------------------------------------------------------------------------
# Model metadata providers
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadataEntry:
    """A registered model metadata provider.

    Registered via ``ctx.register_model_metadata(name, entry)``.
    Queried by ``agent/model_metadata.py`` and CLI model commands.
    """
    name: str
    """Provider identifier (e.g. 'anthropic', 'bedrock')."""

    get_context_length: Optional[Callable[[str], int | None]] = None
    """Return the context length for a model name, or None if unknown."""

    list_models: Optional[Callable[[], List[str]]] = None
    """Return a list of known model IDs for this provider."""

    constants: Dict[str, Any] = field(default_factory=dict)
    """Provider-specific constants (e.g. _COMMON_BETAS, betas lists)."""


# ---------------------------------------------------------------------------
# Credential pool entries
# ---------------------------------------------------------------------------

@dataclass
class CredentialPoolEntry:
    """A registered credential pool provider.

    Registered via ``ctx.register_credential_pool(name, entry)``.
    Queried by ``agent/credential_pool.py``.
    """
    name: str
    """Provider identifier (e.g. 'anthropic')."""

    read_credentials: Optional[Callable] = None
    """Read stored credentials."""

    write_credentials: Optional[Callable] = None
    """Write/store credentials."""

    refresh_credentials: Optional[Callable] = None
    """Refresh stored credentials."""

    read_oauth: Optional[Callable] = None
    """Read OAuth credentials."""


# ---------------------------------------------------------------------------
# Provider resolvers
# ---------------------------------------------------------------------------

@runtime_checkable
class ProviderResolver(Protocol):
    """A plugin that resolves an auxiliary client for a specific provider.

    Registered via ``ctx.register_provider_resolver(provider_name, resolver)``.
    Queried by ``agent/auxiliary_client.py`` in ``resolve_provider_client()``.
    """

    def __call__(
        self,
        *,
        model: str | None = None,
        explicit_api_key: str | None = None,
        explicit_base_url: str | None = None,
        async_mode: bool = False,
        is_vision: bool = False,
        main_runtime: dict | None = None,
        api_mode: str | None = None,
    ) -> tuple[Any, str] | tuple[None, None]:
        """Return ``(client, default_model)`` or ``(None, None)`` if unavailable."""
        ...


# ---------------------------------------------------------------------------
# Credential pool hooks
# ---------------------------------------------------------------------------

@dataclass
class CredentialPoolHook:
    """Provider-specific credential pool operations.

    Registered via ``ctx.register_credential_pool_hook(provider_name, hook)``.
    Queried by ``agent/credential_pool.py``.
    """

    sync_from_credentials_file: Optional[Callable] = None
    """Sync a pool entry from an external credentials file (e.g. ~/.claude/.credentials.json)."""

    refresh_oauth: Optional[Callable] = None
    """Refresh an OAuth token for a pool entry."""

    should_include_in_pool: Optional[Callable] = None
    """Return True if this provider's credentials should be included in the pool."""

    needs_refresh: Optional[Callable] = None
    """Return True if an OAuth entry needs a token refresh."""

    source_priority: Optional[Callable] = None
    """Return integer priority for a credential source (lower = preferred)."""

    discover_credentials: Optional[Callable] = None
    """Discover external credentials and upsert into the pool entries.

    Signature: (entries: list, provider: str, is_suppressed: Callable) -> (changed: bool, active_sources: set)
    """

    env_var_order: Optional[list] = None
    """Override env var scan order for this provider (e.g. ['ANTHROPIC_TOKEN', 'CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_API_KEY'])."""

    detect_auth_type: Optional[Callable] = None
    """Given a token string, return the auth type for this provider.

    Signature: (token: str) -> str  (e.g. AUTH_TYPE_OAUTH or AUTH_TYPE_API_KEY)
    """


# ---------------------------------------------------------------------------
# Pricing providers
# ---------------------------------------------------------------------------

# Re-export PricingEntry from usage_pricing — that's the canonical definition
# with Decimal fields. The registry stores these directly keyed by (provider, model).
# Lazy import to avoid circular dependency (usage_pricing imports registries at runtime).
def _get_pricing_entry_class():
    from agent.usage_pricing import PricingEntry
    return PricingEntry


# ---------------------------------------------------------------------------
# Provider overlays
# ---------------------------------------------------------------------------

@dataclass
class ProviderOverlayEntry:
    """A provider overlay registered by a plugin.

    Registered via ``ctx.register_provider_overlay(provider_name, entry)``.
    Queried by ``hermes_cli/providers.py``.

    This mirrors the fields of ``HermesOverlay`` so that providers.py
    can merge plugin-registered overlays seamlessly.
    """

    provider_name: str
    """Primary provider name (e.g. 'anthropic', 'bedrock')."""

    transport: str = "openai_chat"
    """Transport type: openai_chat | anthropic_messages | codex_responses | bedrock_converse"""

    is_aggregator: bool = False
    """Whether this provider aggregates multiple model providers."""

    auth_type: str = "api_key"
    """Auth type: api_key | oauth_device_code | oauth_external | aws_sdk | external_process"""

    extra_env_vars: Tuple[str, ...] = ()
    """Environment variable names that indicate this provider is configured."""

    base_url_override: str = ""
    """Override if models.dev URL is wrong/missing."""

    base_url_env_var: str = ""
    """Env var for user-custom base URL."""

    display_name: str = ""
    """Human-readable name for the provider (e.g. 'Anthropic', 'AWS Bedrock')."""

    aliases: List[str] = field(default_factory=list)
    """Alternative names that resolve to this provider."""


# ---------------------------------------------------------------------------
# The global registries (singleton)
# ---------------------------------------------------------------------------

class PluginRegistries:
    """Central store for all plugin-registered capabilities.

    A single instance is created at import time and shared across the
    process.  Plugins populate it during ``register()``; the core
    queries it at runtime.
    """

    def __init__(self) -> None:
        self.auth_providers: Dict[str, AuthProviderEntry] = {}
        self.transport_builders: Dict[str, TransportBuilder] = {}
        self._transports: Dict[str, type] = {}
        self.platform_adapters: Dict[str, PlatformAdapterEntry] = {}
        self.tool_providers: Dict[str, ToolProviderEntry] = {}
        self.model_metadata: Dict[str, ModelMetadataEntry] = {}
        self.credential_pools: Dict[str, CredentialPoolEntry] = {}
        self._provider_services: Dict[str, Dict[str, Any]] = {}
        self._provider_resolvers: Dict[str, Callable] = {}
        self._credential_pool_hooks: Dict[str, CredentialPoolHook] = {}
        self._pricing_providers: Dict[tuple, Any] = {}
        self._provider_overlays: Dict[str, ProviderOverlayEntry] = {}

    # -- registration methods (called from PluginContext) --------------------

    def register_auth_provider(
        self,
        name: str,
        provider: AuthProvider,
        *,
        cli_group: str = "",
        setup_subcommands: bool = False,
    ) -> None:
        self.auth_providers[name] = AuthProviderEntry(
            provider=provider,
            cli_group=cli_group,
            setup_subcommands=setup_subcommands,
        )

    def register_transport(self, name: str, builder: TransportBuilder) -> None:
        self.transport_builders[name] = builder

    def register_platform(self, entry: PlatformAdapterEntry) -> None:
        self.platform_adapters[entry.name] = entry

    def register_tool_provider(self, entry: ToolProviderEntry) -> None:
        self.tool_providers[entry.name] = entry

    def register_model_metadata(self, entry: ModelMetadataEntry) -> None:
        self.model_metadata[entry.name] = entry

    def register_credential_pool(self, entry: CredentialPoolEntry) -> None:
        self.credential_pools[entry.name] = entry

    def register_provider_resolver(self, name: str, resolver: Callable) -> None:
        """Register a provider resolver callable.

        The resolver is called by ``resolve_provider_client()`` to create an
        auxiliary client for a specific provider.  Signature::

            def resolver(
                *,
                model: str | None,
                explicit_api_key: str | None,
                explicit_base_url: str | None,
                async_mode: bool,
                is_vision: bool,
                main_runtime: dict | None,
                api_mode: str | None,
            ) -> tuple[Any, str] | tuple[None, None]:
                ...

        Returns ``(client, default_model)`` or ``(None, None)``.
        """
        self._provider_resolvers[name] = resolver

    def register_credential_pool_hook(self, name: str, hook: CredentialPoolHook) -> None:
        """Register a credential pool hook for provider-specific pool operations."""
        self._credential_pool_hooks[name] = hook

    def register_pricing_provider(self, name: str, entries: List[tuple]) -> None:
        """Register pricing entries for a provider.

        Each entry is a (provider, model, PricingEntry) tuple so the
        lookup key matches the (provider, model) pattern used by
        _OFFICIAL_DOCS_PRICING.
        """
        for prov, model, entry in entries:
            self._pricing_providers[(prov, model)] = entry

    def register_provider_overlay(self, entry: ProviderOverlayEntry) -> None:
        """Register a provider overlay entry from a plugin."""
        self._provider_overlays[entry.provider_name] = entry

    # -- query helpers -------------------------------------------------------

    def get_auth_provider(self, name: str) -> AuthProviderEntry | None:
        return self.auth_providers.get(name)

    def get_transport(self, name: str) -> TransportBuilder | None:
        return self.transport_builders.get(name)

    def get_platform(self, name: str) -> PlatformAdapterEntry | None:
        return self.platform_adapters.get(name)

    def get_tool_provider(self, name: str) -> ToolProviderEntry | None:
        return self.tool_providers.get(name)

    def get_model_metadata(self, name: str) -> ModelMetadataEntry | None:
        return self.model_metadata.get(name)

    def get_credential_pool(self, name: str) -> CredentialPoolEntry | None:
        return self.credential_pools.get(name)

    def get_provider_resolver(self, name: str) -> Callable | None:
        """Return the registered resolver for a provider, or None."""
        return self._provider_resolvers.get(name)

    def get_credential_pool_hook(self, name: str) -> CredentialPoolHook | None:
        """Return the registered credential pool hook for a provider, or None."""
        return self._credential_pool_hooks.get(name)

    def get_pricing_entry(self, provider: str, model: str) -> Any:
        """Return a registered pricing entry for (provider, model), or None."""
        return self._pricing_providers.get((provider, model))

    def all_pricing_entries(self) -> Dict[tuple, Any]:
        """Return all registered pricing entries (keyed by (provider, model))."""
        return dict(self._pricing_providers)

    def get_provider_overlay(self, name: str) -> ProviderOverlayEntry | None:
        """Return a registered provider overlay, or None."""
        return self._provider_overlays.get(name)

    def all_provider_overlays(self) -> Dict[str, ProviderOverlayEntry]:
        """Return all registered provider overlays."""
        return dict(self._provider_overlays)

    def all_auth_providers(self) -> List[AuthProviderEntry]:
        return list(self.auth_providers.values())

    def all_platforms(self) -> List[PlatformAdapterEntry]:
        return list(self.platform_adapters.values())

    def all_tool_providers(self) -> List[ToolProviderEntry]:
        return list(self.tool_providers.values())

    # -- provider services (model-provider namespace) -----------------------

    def register_provider_services(self, name: str, services: Dict[str, Any]) -> None:
        """Register a namespace dict of provider-specific services.

        This is the escape hatch for model-provider plugins that expose many
        symbols (anthropic has 50+).  Each plugin registers its public surface
        as a flat dict of ``{symbol_name: callable_or_value}``.  Core code
        looks up specific symbols instead of importing from the plugin
        package directly.

        Each callable value is stored as a *lazy module-attribute reference*
        so that ``unittest.mock.patch("pkg.mod.fn")`` works correctly in
        tests — the registry re-reads ``mod.fn`` on every lookup instead of
        capturing the function object at register time.

        Example::

            registries.register_provider_services("anthropic", {
                "build_anthropic_client": build_anthropic_client,
                "resolve_anthropic_token": resolve_anthropic_token,
                "_is_oauth_token": _is_oauth_token,
                ...
            })
        """
        import sys

        def _make_lazy(fn: Any) -> Any:
            """Return a lazy wrapper that re-reads fn from its module each call.

            This makes mock.patch() on the module attribute work transparently —
            the registry never caches the function object, just the reference path.
            """
            if not callable(fn):
                return fn
            module = getattr(fn, "__module__", None)
            qualname = getattr(fn, "__qualname__", None)
            if not module or not qualname or "." in qualname:
                # non-simple attribute (lambda, nested fn, class method) — store directly
                return fn

            class _LazyRef:
                __slots__ = ("_mod", "_attr", "_fallback")

                def __init__(self, mod: str, attr: str, fallback: Any) -> None:
                    self._mod = mod
                    self._attr = attr
                    self._fallback = fallback

                def _resolve(self) -> Any:
                    mod = sys.modules.get(self._mod)
                    return getattr(mod, self._attr, self._fallback) if mod else self._fallback

                def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    return self._resolve()(*args, **kwargs)

                def __getattr__(self, name: str) -> Any:
                    if name.startswith("_"):
                        raise AttributeError(name)
                    return getattr(self._resolve(), name)

                def __repr__(self) -> str:  # pragma: no cover
                    return f"<LazyRef {self._mod}.{self._attr}>"

                # Allow isinstance checks and hasattr to pass through
                def __bool__(self) -> bool:
                    return True

            return _LazyRef(module, qualname, fn)

        self._provider_services[name] = {k: _make_lazy(v) for k, v in services.items()}

    def get_provider_service(self, provider: str, name: str) -> Any:
        """Look up a single symbol from a provider's service namespace.

        Returns ``None`` if the provider is not registered or the symbol
        doesn't exist.
        """
        ns = self._provider_services.get(provider)
        if ns is None:
            return None
        return ns.get(name)

    def get_provider_namespace(self, provider: str) -> Dict[str, Any]:
        """Return the full service namespace dict for a provider (empty dict if unregistered)."""
        return self._provider_services.get(provider, {})


# Module-level singleton — the one and only instance.
registries = PluginRegistries()
