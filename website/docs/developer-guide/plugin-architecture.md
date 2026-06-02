---
sidebar_position: 2
title: "Plugin Architecture"
description: "How the plugin system works — workspace layout, capability registries, dependency isolation, and the hermetic core boundary"
---

# Plugin Architecture

Since v0.14, Hermes Agent is built on a **plugin-first architecture**: every
optional capability — model providers, platform adapters, TTS/STT, terminal
backends, image generation — lives in its own installable Python package under
`plugins/`. The core codebase (`agent/`, `hermes_cli/`, `gateway/`, `tools/`)
**never** imports from a plugin package directly. Instead, plugins register
their capabilities into typed registries during `register()`, and the core
queries those registries at runtime.

This page covers the structural design. For the user-facing guide to creating
plugins, see [Build a Hermes Plugin](/guides/build-a-hermes-plugin). For
enabling/disabling plugins, see [Plugins](/user-guide/features/plugins).

## Why everything is a plugin

Before v0.14, optional capabilities were wired into core with
`tools/lazy_deps.py` — a runtime `pip install` helper called `ensure()`. On
NixOS (and any sealed-venv environment), `ensure()` can't work because the
venv is immutable at build time. The old design also meant:

- **Single source of truth was split** — deps were declared in `pyproject.toml`
  extras AND in `LAZY_DEPS` dicts inside plugin code.
- **Core was coupled to plugins** — `from hermes_agent_bedrock import
  has_aws_credentials` in `hermes_cli/auth_commands.py` meant adding a new
  provider required editing core files.
- **Testing was fragile** — `ensure()` mocking was complex and tests regularly
  passed locally but failed in CI (or vice versa) because of venv state leaks.

The plugin-first architecture fixes all three:

| Problem | Fix |
|---------|-----|
| `ensure()` doesn't work on NixOS | Dependencies are installed by the package manager. No runtime `pip install`. |
| Dual source of truth for deps | Each plugin's `pyproject.toml` is the **only** place its deps are declared. |
| Core imports plugins directly | Core queries typed registries. Plugins register themselves. |
| Flaky `ensure()` tests | Gone. If a plugin isn't installed, `ImportError` — same as any Python package. |

## Workspace layout

All plugin packages live under `plugins/` as members of a
[uv workspace](https://docs.astral.sh/uv/concepts/workspaces/). Each plugin
is a standard Python package with its own `pyproject.toml`:

```
plugins/
├── model-providers/
│   ├── anthropic/
│   │   ├── pyproject.toml          # package: hermes-agent-anthropic
│   │   ├── plugin.yaml             # directory-scanner manifest (dev mode)
│   │   └── hermes_agent_anthropic/
│   │       ├── __init__.py          # register(), public re-exports
│   │       ├── adapter.py           # Anthropic-specific client building
│   │       └── ...
│   ├── bedrock/
│   │   ├── pyproject.toml          # package: hermes-agent-bedrock
│   │   └── hermes_agent_bedrock/
│   │       └── ...
│   └── azure-foundry/
│       ├── pyproject.toml          # package: hermes-agent-azure
│       └── hermes_agent_azure/
│           └── ...
├── platforms/
│   ├── telegram/
│   │   ├── pyproject.toml          # package: hermes-agent-telegram
│   │   └── hermes_agent_telegram/
│   │       └── ...
│   ├── slack/
│   ├── discord/
│   ├── feishu/
│   ├── dingtalk/
│   └── matrix/
├── tts/
│   ├── pyproject.toml              # package: hermes-agent-tts
│   └── hermes_agent_tts/
├── stt/
│   ├── pyproject.toml              # package: hermes-agent-stt
│   └── hermes_agent_stt/
├── image_gen/
│   └── fal_pkg/
│       ├── pyproject.toml          # package: hermes-agent-fal
│       └── hermes_agent_fal/
├── terminals/
│   ├── daytona/
│   ├── modal/
│   └── vercel/
└── ...
```

The root `pyproject.toml` declares the workspace:

```toml
[tool.uv.workspace]
members = [
    "plugins/model-providers/anthropic",
    "plugins/model-providers/bedrock",
    "plugins/model-providers/azure-foundry",
    "plugins/platforms/telegram",
    "plugins/platforms/slack",
    # ... all 21 workspace members
]
```

And each plugin depends on the main `hermes-agent` package for shared
utilities:

```toml
# plugins/platforms/telegram/pyproject.toml
[project]
name = "hermes-agent-telegram"
dependencies = [
    "hermes-agent",
    "python-telegram-bot>=22.0",
]

[tool.uv.sources]
hermes-agent = { workspace = true }
```

### Single source of truth for dependencies

A plugin's `pyproject.toml` is the **only** place its runtime dependencies are
declared. The root `pyproject.toml` maps extras to workspace members:

```toml
[project.optional-dependencies]
telegram = ["hermes-agent-telegram"]
slack = ["hermes-agent-slack"]
anthropic = ["hermes-agent-anthropic"]
all = [
    "hermes-agent-telegram",
    "hermes-agent-slack",
    "hermes-agent-anthropic",
    # ... all plugins
]
```

When you `uv sync --extra telegram`, uv resolves the workspace member
`hermes-agent-telegram` and installs it (with its own deps) into the venv.

There is no `LAZY_DEPS` dict, no `ensure()`, no duplicate pin lists. The
`pyproject.toml` is the truth; `uv.lock` is the resolution.

## The hermetic core boundary

The core codebase (`agent/`, `hermes_cli/`, `gateway/`, `tools/`) must never
import from a `hermes_agent_*` plugin package. This is enforced by convention
and should be checked in CI.

### How core accesses plugin capabilities

Instead of direct imports, the core queries **typed registries** in
`agent/plugin_registries.py`:

```python
# ❌ OLD — core directly imports plugin
from hermes_agent_bedrock import has_aws_credentials

# ✅ NEW — core queries the registry
from agent.plugin_registries import registries

bedrock_auth = registries.get_auth_provider("bedrock")
if bedrock_auth and bedrock_auth.provider.has_credentials():
    ...
```

### Registry types

| Registry | What it stores | Populated by | Queried by |
|----------|---------------|---------------|------------|
| `auth_providers` | Auth check/resolve functions | Model-provider plugins | `hermes_cli/auth.py`, `auth_commands.py`, `doctor.py` |
| `transport_builders` | Client builders + message converters | Model-provider plugins | `agent/transports/`, `auxiliary_client.py` |
| `platform_adapters` | Adapter classes + `check_requirements()` | Platform plugins | `gateway/run.py`, `tools/send_message_tool.py` |
| `tool_providers` | Tool functions + constants | TTS, STT, FAL, terminal plugins | `tools/voice_mode.py`, `image_generation_tool.py`, `terminal_tool.py` |
| `model_metadata` | Context lengths, model IDs, betas | Model-provider plugins | `agent/model_metadata.py`, `hermes_cli/models.py` |
| `credential_pools` | Credential read/write/refresh | Model-provider plugins | `agent/credential_pool.py` |

Each registry entry is a dataclass or protocol instance with well-typed fields.
The `PluginRegistries` singleton lives at `agent.plugin_registries.registries`.

### Plugin registration

Each plugin's `register(ctx)` function populates the registries:

```python
# plugins/model-providers/bedrock/hermes_agent_bedrock/__init__.py
def register(ctx):
    from agent.plugin_registries import AuthProviderEntry, ModelMetadataEntry

    ctx.register_auth_provider(
        name="bedrock",
        provider=BedrockAuthProvider(),
        cli_group="AWS / Bedrock",
    )
    ctx.register_model_metadata(ModelMetadataEntry(
        name="bedrock",
        list_models=bedrock_model_ids_or_none,
        get_context_length=get_bedrock_context_length,
    ))
```

The `PluginContext` (`hermes_cli/plugins.py`) delegates each
`register_*()` call to the matching method on the global `PluginRegistries`
singleton. This keeps the existing PluginManager lifecycle intact — plugins
are still discovered and loaded the same way, they just register into more
registries.

### Existing specialized registries

Some plugin categories already had registries before the refactor. These
continue to work alongside the new generic registries:

| Registry | Module | Used by |
|----------|--------|---------|
| `platform_registry` | `gateway/platform_registry.py` | `ctx.register_platform()` |
| `tts_registry` | `agent/tts_registry.py` | `ctx.register_tts_provider()` |
| `transcription_registry` | `agent/transcription_registry.py` | `ctx.register_transcription_provider()` |
| `image_gen_provider` | `agent/image_gen_provider.py` | `ctx.register_image_gen_provider()` |
| `video_gen_provider` | `agent/video_gen_provider.py` | `ctx.register_video_gen_provider()` |
| `context_engine` | `agent/context_engine.py` | `ctx.register_context_engine()` |
| `memory_manager` | `agent/memory_manager.py` | `MemoryProvider` subclasses |

The new `plugin_registries` module covers the capabilities that **didn't** have
a registry before: auth, transport building, model metadata, credential
pooling, and tool-provider registration.

## Plugin discovery

Plugins are discovered through **three** mechanisms (same as before the
refactor, but now with workspace awareness):

1. **Directory scanner** — scans `plugins/` (bundled), `~/.hermes/plugins/`
   (user), `.hermes/plugins/` (project) for directories with `plugin.yaml`.
   This is the primary path for dev-mode and for user-installed plugins.

2. **Entry points** — packages that declare
   `[project.entry-points."hermes_agent.plugins"]` in their `pyproject.toml`.
   This is the primary path for `pip install`-ed plugins and for NixOS
   installs where the venv already contains the installed packages.

3. **uv workspace members** — the 21 builtin plugins are workspace members,
   so `uv sync --extra <name>` installs them into the venv. At runtime, the
   entry-point scanner finds them because each plugin declares the
   `hermes_agent.plugins` entry point in its `pyproject.toml`.

On NixOS, `loadWorkspace` discovers all workspace members from `uv.lock`
automatically, and `mkVirtualEnv { hermes-agent = ["all"] }` installs all
plugin packages as transitive deps.

## Building and publishing

### Dev / source installs

```bash
uv sync --all-extras    # install all plugins + their deps
uv sync --extra telegram  # install just the telegram plugin
```

### Wheel publishing (custom build backend)

The root `pyproject.toml` uses a custom PEP 517 build backend
(`_build_backend.py`) that wraps `setuptools.build_meta`. At wheel build time
it:

1. Reads each plugin's `pyproject.toml` from the workspace.
2. Inlines the plugin's runtime dependencies into the corresponding
   `[project.optional-dependencies]` extra.
3. Delegates to `setuptools` to build the wheel.

This means the published wheel has `telegram = ["python-telegram-bot>=22.0",
...]` instead of `telegram = ["hermes-agent-telegram"]` — because the
individual plugin packages aren't on PyPI.

Source installs and NixOS use workspace resolution directly and never hit the
build-backend rewrite path.

### NixOS

```nix
services.hermes-agent = {
  enable = true;
  # All plugins are included by default via "all" extra.
  # Select specific plugins with:
  extraDependencyGroups = [ "telegram" "anthropic" ];
};
```

`loadWorkspace` discovers all workspace members from `uv.lock`. No structural
changes to the Nix files are needed — the existing `mkVirtualEnv` + `extraDependencyGroups`
mechanism already handles it.

## Tests

Plugin test files live in the plugin's own `tests/` directory:

```
plugins/platforms/telegram/
├── tests/
│   ├── conftest.py
│   ├── test_telegram_format.py
│   └── ...
└── hermes_agent_telegram/
    └── ...
```

The test runner (`scripts/run_tests_parallel.py`) discovers tests under both
`tests/` (core) and `plugins/` (plugins). The root `conftest.py` provides
shared fixtures for both.

Running a plugin's tests requires the plugin to be installed:

```bash
uv sync --extra telegram
scripts/run_tests.sh plugins/platforms/telegram/tests/
```

If the plugin isn't installed, its tests fail with `ModuleNotFoundError` —
which is correct. You can't run telegram tests without the telegram package.

## Migration checklist (for adding a new plugin)

When a new optional capability is added to Hermes:

1. **Create a plugin package** under `plugins/<category>/<name>/` with:
   - `pyproject.toml` (name, version, deps, entry point declaration)
   - `plugin.yaml` (for directory-scanner discovery in dev)
   - `hermes_agent_<name>/__init__.py` with `register(ctx)`
   - `hermes_agent_<name>/tests/` for plugin-specific tests

2. **Add to workspace** — add the directory to `[tool.uv.workspace].members`
   and `[tool.uv.sources]` in the root `pyproject.toml`.

3. **Add an extra** — add `name = ["hermes-agent-<name>"]` to
   `[project.optional-dependencies]` and include it in `all`.

4. **Register capabilities** — in `register(ctx)`, call the appropriate
   `ctx.register_*()` methods to populate the typed registries.

5. **No core edits** — the core code should not need to change. If it does,
   that's a sign the registry surface is incomplete and needs a new
   `register_*()` method on `PluginContext`.

6. **Run `uv lock`** — resolve the new workspace member.

7. **Add NixOS support** — if the plugin has native deps, add an override
   in `nix/python.nix`. Otherwise `loadWorkspace` handles it automatically.

## The rule

> **If it can be a plugin, it must be a plugin.**

Adding optional capabilities to core files is a code review rejection. If the
plugin surface doesn't support what you need, extend the surface (new
registry type, new hook, new `ctx` method) — don't inline the capability.
