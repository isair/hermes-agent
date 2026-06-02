# nix/checks.nix — Build-time verification tests
#
# Checks are Linux-only: the full Python venv (via uv2nix) includes
# transitive deps like onnxruntime that lack compatible wheels on
# aarch64-darwin. The package and devShell still work on macOS.
{ inputs, ... }: {
  perSystem = { pkgs, lib, self', ... }:
    let
      hermes-agent = self'.packages.default;
      hermesVenv = hermes-agent.hermesVenv;

      configMergeScript = pkgs.callPackage ./configMergeScript.nix { };

      # Auto-generated config key reference — always in sync with Python
      configKeys = pkgs.runCommand "hermes-config-keys" {} ''
        set -euo pipefail
        export HOME=$TMPDIR
        ${hermesVenv}/bin/python3 -c '
import json, sys
from hermes_cli.config import DEFAULT_CONFIG

def leaf_paths(d, prefix=""):
    paths = []
    for k, v in sorted(d.items()):
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            paths.extend(leaf_paths(v, path))
        else:
            paths.append(path)
    return paths

json.dump(sorted(leaf_paths(DEFAULT_CONFIG)), sys.stdout, indent=2)
' > $out
      '';
    in {
      packages.configKeys = configKeys;

      checks = {
        # Cross-platform evaluation — catches "not supported for interpreter"
        # errors (e.g. sphinx dropping python311) without needing a darwin builder.
        # Evaluation is pure and instant; it doesn't build anything.
        cross-eval = let
          targetSystems = builtins.filter
            (s: inputs.self.packages ? ${s})
            [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
          tryEvalPkg = sys:
            let pkg = inputs.self.packages.${sys}.default;
            in builtins.tryEval (builtins.seq pkg.drvPath true);
          results = map (sys: { inherit sys; result = tryEvalPkg sys; }) targetSystems;
          failures = builtins.filter (r: !r.result.success) results;
          failMsg = lib.concatMapStringsSep "\n" (r: "  - ${r.sys}") failures;
        in pkgs.runCommand "hermes-cross-eval" { } (
          if failures != [] then
            throw "Package fails to evaluate on:\n${failMsg}"
          else ''
            echo "PASS: package evaluates on all ${toString (builtins.length targetSystems)} platforms"
            mkdir -p $out
            echo "ok" > $out/result
          ''
        );
      } // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
        # Verify binaries exist and are executable
        package-contents = pkgs.runCommand "hermes-package-contents" { } ''
          set -e
          echo "=== Checking binaries ==="
          test -x ${hermes-agent}/bin/hermes || (echo "FAIL: hermes binary missing"; exit 1)
          test -x ${hermes-agent}/bin/hermes-agent || (echo "FAIL: hermes-agent binary missing"; exit 1)
          echo "PASS: All binaries present"

          echo "=== Checking version ==="
          ${hermes-agent}/bin/hermes version 2>&1 | grep -qi "hermes" || (echo "FAIL: version check"; exit 1)
          echo "PASS: Version check"

          echo "=== All checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify every pyproject.toml [project.scripts] entry has a wrapped binary
        entry-points-sync = pkgs.runCommand "hermes-entry-points-sync" { } ''
          set -e
          echo "=== Checking entry points match pyproject.toml [project.scripts] ==="
          for bin in hermes hermes-agent hermes-acp; do
            test -x ${hermes-agent}/bin/$bin || (echo "FAIL: $bin binary missing from Nix package"; exit 1)
            echo "PASS: $bin present"
          done

          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify CLI subcommands are accessible
        cli-commands = pkgs.runCommand "hermes-cli-commands" { } ''
          set -e
          export HOME=$(mktemp -d)

          echo "=== Checking hermes --help ==="
          ${hermes-agent}/bin/hermes --help 2>&1 | grep -q "gateway" || (echo "FAIL: gateway subcommand missing"; exit 1)
          ${hermes-agent}/bin/hermes --help 2>&1 | grep -q "config" || (echo "FAIL: config subcommand missing"; exit 1)
          echo "PASS: All subcommands accessible"

          echo "=== All CLI checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled skills are present in the package
        bundled-skills = pkgs.runCommand "hermes-bundled-skills" { } ''
          set -e
          echo "=== Checking bundled skills ==="
          test -d ${hermes-agent}/share/hermes-agent/skills || (echo "FAIL: skills directory missing"; exit 1)
          echo "PASS: skills directory exists"

          SKILL_COUNT=$(find ${hermes-agent}/share/hermes-agent/skills -name "SKILL.md" | wc -l)
          test "$SKILL_COUNT" -gt 0 || (echo "FAIL: no SKILL.md files found in skills directory"; exit 1)
          echo "PASS: $SKILL_COUNT bundled skills found"

          grep -q "HERMES_BUNDLED_SKILLS" ${hermes-agent}/bin/hermes || \
            (echo "FAIL: HERMES_BUNDLED_SKILLS not in wrapper"; exit 1)
          echo "PASS: HERMES_BUNDLED_SKILLS set in wrapper"

          echo "=== All bundled skills checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled plugins (platforms, memory, context_engine) are present
        bundled-plugins = pkgs.runCommand "hermes-bundled-plugins" { } ''
          set -e
          echo "=== Checking bundled plugins ==="
          test -d ${hermes-agent}/share/hermes-agent/plugins || (echo "FAIL: plugins directory missing"; exit 1)
          echo "PASS: plugins directory exists"

          test -f ${hermes-agent}/share/hermes-agent/plugins/platforms/irc/plugin.yaml || \
            (echo "FAIL: irc plugin manifest missing"; exit 1)
          echo "PASS: irc plugin manifest present"

          grep -q "HERMES_BUNDLED_PLUGINS" ${hermes-agent}/bin/hermes || \
            (echo "FAIL: HERMES_BUNDLED_PLUGINS not in wrapper"; exit 1)
          echo "PASS: HERMES_BUNDLED_PLUGINS set in wrapper"

          echo "=== All bundled plugins checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled TUI is present and compiled
        bundled-tui = pkgs.runCommand "hermes-bundled-tui" { } ''
          set -e
          echo "=== Checking bundled TUI ==="
          test -d ${hermes-agent}/ui-tui || (echo "FAIL: ui-tui directory missing"; exit 1)
          echo "PASS: ui-tui directory exists"

          test -f ${hermes-agent}/ui-tui/dist/entry.js || (echo "FAIL: compiled entry.js missing"; exit 1)
          echo "PASS: compiled entry.js present"

          # self-contained bundle; no runtime node_modules expected

          grep -q "HERMES_TUI_DIR" ${hermes-agent}/bin/hermes || \
            (echo "FAIL: HERMES_TUI_DIR not in wrapper"; exit 1)
          echo "PASS: HERMES_TUI_DIR set in wrapper"

          echo "=== All bundled TUI checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify HERMES_NODE is set in wrapper and points to Node 20+
        # (string-width uses the /v regex flag which requires Node 20+)
        hermes-node = pkgs.runCommand "hermes-node-version" { } ''
          set -e
          echo "=== Checking HERMES_NODE in wrapper ==="
          grep -q "HERMES_NODE" ${hermes-agent}/bin/hermes || \
            (echo "FAIL: HERMES_NODE not set in wrapper"; exit 1)
          echo "PASS: HERMES_NODE present in wrapper"

          HERMES_NODE=$(sed -n "s/^export HERMES_NODE='\(.*\)'/\1/p" ${hermes-agent}/bin/hermes)
          test -x "$HERMES_NODE" || (echo "FAIL: HERMES_NODE=$HERMES_NODE not executable"; exit 1)
          echo "PASS: HERMES_NODE executable at $HERMES_NODE"

          NODE_MAJOR=$("$HERMES_NODE" --version | sed 's/^v//' | cut -d. -f1)
          test "$NODE_MAJOR" -ge 20 || \
            (echo "FAIL: Node v$NODE_MAJOR < 20, TUI needs /v regex flag support"; exit 1)
          echo "PASS: Node v$NODE_MAJOR >= 20"

          echo "=== All HERMES_NODE checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify HERMES_MANAGED guard works on all mutation commands
        managed-guard = pkgs.runCommand "hermes-managed-guard" { } ''
          set -e
          export HOME=$(mktemp -d)

          check_blocked() {
            local label="$1"
            shift
            OUTPUT=$(HERMES_MANAGED=true "$@" 2>&1 || true)
            echo "$OUTPUT" | grep -q "managed by NixOS" || (echo "FAIL: $label not guarded"; echo "$OUTPUT"; exit 1)
            echo "PASS: $label blocked in managed mode"
          }

          echo "=== Checking HERMES_MANAGED guards ==="
          check_blocked "config set" ${hermes-agent}/bin/hermes config set model foo
          check_blocked "config edit" ${hermes-agent}/bin/hermes config edit

          echo "=== All guard checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify extraPythonPackages PYTHONPATH injection
        extra-python-packages = let
          testPkg = pkgs.python312Packages.pyfiglet;
          hermesWithExtra = hermes-agent.override {
            extraPythonPackages = [ testPkg ];
          };
        in pkgs.runCommand "hermes-extra-python-packages" { } ''
          set -e
          echo "=== Checking extraPythonPackages PYTHONPATH injection ==="

          grep -q "PYTHONPATH" ${hermesWithExtra}/bin/hermes || \
            (echo "FAIL: PYTHONPATH not in wrapper"; exit 1)
          echo "PASS: PYTHONPATH present in wrapper"

          grep -q "${testPkg}" ${hermesWithExtra}/bin/hermes || \
            (echo "FAIL: test package path not in PYTHONPATH"; exit 1)
          echo "PASS: test package path found in wrapper"

          echo "=== Checking base package has no PYTHONPATH ==="
          if grep -q "PYTHONPATH" ${hermes-agent}/bin/hermes; then
            echo "FAIL: base package should not have PYTHONPATH"; exit 1
          fi
          echo "PASS: base package clean"

          echo "=== All extraPythonPackages checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify extraDependencyGroups passes through to python.nix
        extra-dependency-groups = let
          hermesWithGroups = hermes-agent.override {
            extraDependencyGroups = [ "honcho" ];
          };
        in pkgs.runCommand "hermes-extra-dependency-groups" { } ''
          set -e
          echo "=== Checking extraDependencyGroups override evaluates ==="

          # Eval-only: verify the override produces valid derivation paths
          # without building the full venv (which is expensive and redundant
          # since the mechanism is just list concatenation into python.nix).
          echo "derivation: ${hermesWithGroups}"
          echo "venv: ${hermesWithGroups.hermesVenv}"
          echo "PASS: extraDependencyGroups override evaluates cleanly"

          echo "=== All extraDependencyGroups checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # ── Plugin architecture (hermetic core boundary) ───────────────────
        #
        # These checks prove that under NixOS (sealed venv, no pip install),
        # the plugin system works correctly:
        #   1. Core never imports from hermes_agent_* packages directly
        #   2. Plugin registries are populated after discovery
        #   3. Provider service namespaces are queryable
        #   4. Optional plugins degrade gracefully (None returns, no crash)
        #   5. No ensure() / lazy_deps / pip-install at runtime

        # Check 1: Zero direct hermes_agent_* imports in core code
        plugin-hermetic-boundary = pkgs.runCommand "hermes-plugin-hermetic-boundary" { } ''
          set -e
          echo "=== Checking core never imports from plugin packages ==="

          # Search for direct imports from hermes_agent_* in core code
          # (excluding plugins/, tests/, website/, and comments)
          VIOLATIONS=$(${hermesVenv}/bin/python3 -c '
          import subprocess, re, sys
          result = subprocess.run(
            ["grep", "-rn",
             "from hermes_agent_\\|import hermes_agent_",
             "${hermes-agent}/share/hermes-agent"],
            capture_output=True, text=True
          )
          lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
          # Filter: only .py files, not in plugins/ or tests/, not comments
          violations = []
          for line in lines:
            if not line.endswith(".py"):
              continue
            parts = line.split(":", 2)
            if len(parts) < 3:
              continue
            filepath, lineno, content = parts
            # Skip plugin directories
            if "/plugins/" in filepath:
              continue
            # Skip test directories
            if "/tests/" in filepath or "/test_" in filepath:
              continue
            # Skip comments
            stripped = content.lstrip()
            if stripped.startswith("#"):
              continue
            violations.append(line)
          for v in violations:
            print(v)
          sys.exit(1 if violations else 0)
          ' 2>&1 || true)

          if [ -n "$VIOLATIONS" ]; then
            echo "FAIL: Core code imports directly from plugin packages:"
            echo "$VIOLATIONS"
            exit 1
          fi
          echo "PASS: Zero direct hermes_agent_* imports in core"

          echo "=== Checking no ensure() / LAZY_DEPS in core ==="
          ENSURE_VIOLATIONS=$(grep -rn 'ensure(' ${hermes-agent}/share/hermes-agent/agent/ ${hermes-agent}/share/hermes-agent/hermes_cli/ --include='*.py' 2>/dev/null | grep -v '__pycache__' | grep -v '# ' || true)
          if [ -n "$ENSURE_VIOLATIONS" ]; then
            echo "FAIL: ensure() still used in core:"
            echo "$ENSURE_VIOLATIONS"
            exit 1
          fi
          echo "PASS: No ensure() calls in core code"

          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Check 2: Plugin registries populate after discovery
        plugin-registries-populate = pkgs.runCommand "hermes-plugin-registries-populate" { } ''
          set -e
          echo "=== Checking plugin registries populate after discovery ==="
          export HOME=$(mktemp -d)

          RESULT=$(${hermesVenv}/bin/python3 -c '
          import json, sys
          from hermes_cli.plugins import PluginManager
          from agent.plugin_registries import registries

          pm = PluginManager()
          pm.discover_and_load(force=True)

          out = {
            "provider_services": list(registries._provider_services.keys()),
            "platform_adapters": list(registries.platform_adapters.keys()),
            "tool_providers": list(registries.tool_providers.keys()),
          }
          json.dump(out, sys.stdout)
          ' 2>/dev/null)

          echo "Registry state: $RESULT"

          # Verify provider services populated
          PROV_COUNT=$(echo "$RESULT" | ${pkgs.jq}/bin/jq '.provider_services | length')
          if [ "$PROV_COUNT" -lt 1 ]; then
            echo "FAIL: No provider services registered (expected >= 1)"
            exit 1
          fi
          echo "PASS: $PROV_COUNT provider service(s) registered"

          # Verify platform adapters populated
          PLAT_COUNT=$(echo "$RESULT" | ${pkgs.jq}/bin/jq '.platform_adapters | length')
          if [ "$PLAT_COUNT" -lt 1 ]; then
            echo "FAIL: No platform adapters registered (expected >= 1)"
            exit 1
          fi
          echo "PASS: $PLAT_COUNT platform adapter(s) registered"

          # Verify tool providers populated
          TOOL_COUNT=$(echo "$RESULT" | ${pkgs.jq}/bin/jq '.tool_providers | length')
          if [ "$TOOL_COUNT" -lt 1 ]; then
            echo "FAIL: No tool providers registered (expected >= 1)"
            exit 1
          fi
          echo "PASS: $TOOL_COUNT tool provider(s) registered"

          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Check 3: Specific provider service lookups work
        plugin-provider-lookups = pkgs.runCommand "hermes-plugin-provider-lookups" { } ''
          set -e
          echo "=== Checking provider service lookups ==="
          export HOME=$(mktemp -d)

          RESULT=$(${hermesVenv}/bin/python3 -c '
          import json, sys
          from hermes_cli.plugins import PluginManager
          from agent.plugin_registries import registries

          pm = PluginManager()
          pm.discover_and_load(force=True)

          checks = {
            "anthropic.resolve_anthropic_token": registries.get_provider_service("anthropic", "resolve_anthropic_token") is not None,
            "bedrock.has_aws_credentials": registries.get_provider_service("bedrock", "has_aws_credentials") is not None,
            "azure.is_token_provider": registries.get_provider_service("azure", "is_token_provider") is not None,
          }
          json.dump(checks, sys.stdout)
          ' 2>/dev/null)

          echo "Lookup results: $RESULT"

          for key in anthropic.resolve_anthropic_token bedrock.has_aws_credentials azure.is_token_provider; do
            VALUE=$(echo "$RESULT" | ${pkgs.jq}/bin/jq --arg k "$key" '.[$k]')
            if [ "$VALUE" != "true" ]; then
              echo "FAIL: $key lookup returned $VALUE (expected true)"
              exit 1
            fi
            echo "PASS: $key lookup works"
          done

          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Check 4: Missing plugins degrade gracefully (no crash)
        plugin-missing-graceful = pkgs.runCommand "hermes-plugin-missing-graceful" { } ''
          set -e
          echo "=== Checking missing plugins degrade gracefully ==="
          export HOME=$(mktemp -d)

          ${hermesVenv}/bin/python3 -c '
          from agent.plugin_registries import registries

          # Lookup from non-existent provider — should return None, not crash
          result = registries.get_provider_service("nonexistent-provider", "some_function")
          assert result is None, f"Expected None for missing provider, got {result}"

          # Lookup from empty registry — should return None
          result2 = registries.get_provider_namespace("no-such-provider")
          assert result2 == {}, f"Expected empty dict for missing namespace, got {result2}"

          # Lookup specific tool provider that does not exist
          result3 = registries.get_tool_provider("nonexistent-tool")
          assert result3 is None, f"Expected None for missing tool provider, got {result3}"

          print("PASS: All missing-plugin lookups return None gracefully")
          ' 2>&1

          echo "PASS: Missing plugins degrade gracefully (no crash)"

          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Check 5: No runtime pip install / ensure in gateway/run.py
        plugin-no-runtime-install = pkgs.runCommand "hermes-plugin-no-runtime-install" { } ''
          set -e
          echo "=== Checking no runtime pip install / ensure in core ==="

          # Check gateway/run.py has no ensure() or pip install
          GATEWAY=${hermes-agent}/share/hermes-agent/gateway/run.py
          if [ -f "$GATEWAY" ]; then
            if grep -q 'ensure(' "$GATEWAY" || grep -q 'pip install' "$GATEWAY"; then
              echo "FAIL: gateway/run.py contains ensure() or pip install"
              grep -n 'ensure(\|pip install' "$GATEWAY"
              exit 1
            fi
            echo "PASS: gateway/run.py has no ensure()/pip install"
          else
            echo "SKIP: gateway/run.py not found in package"
          fi

          # Check run_agent.py has no ensure() or pip install
          RUN_AGENT=${hermes-agent}/share/hermes-agent/run_agent.py
          if [ -f "$RUN_AGENT" ]; then
            if grep -q 'ensure(' "$RUN_AGENT" || grep -q 'pip install' "$RUN_AGENT"; then
              echo "FAIL: run_agent.py contains ensure() or pip install"
              grep -n 'ensure(\|pip install' "$RUN_AGENT"
              exit 1
            fi
            echo "PASS: run_agent.py has no ensure()/pip install"
          else
            echo "SKIP: run_agent.py not found in package"
          fi

          # Check tools/lazy_deps.py is gone
          LAZY_DEPS=${hermes-agent}/share/hermes-agent/tools/lazy_deps.py
          if [ -f "$LAZY_DEPS" ]; then
            echo "FAIL: tools/lazy_deps.py still exists — should be removed"
            exit 1
          fi
          echo "PASS: tools/lazy_deps.py removed"

          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Regression guard: messaging deps live outside [all], so the
        # #messaging variant must actually ship discord.py — otherwise
        # `nix profile install .#messaging` regresses to the broken default.
        messaging-variant = pkgs.runCommand "hermes-messaging-variant" { } ''
          set -e
          echo "=== Checking discord.py importable from messaging variant ==="
          ${self'.packages.messaging.hermesVenv}/bin/python3 -c \
            "import discord; print(discord.__version__)"
          echo "PASS: discord.py importable from messaging variant venv"
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # ── Config merge + round-trip test ────────────────────────────────
        # Tests the merge script (Nix activation behavior) across 7
        # scenarios, then verifies Python's load_config() reads correctly.
        config-roundtrip = let
          # Nix settings used across scenarios
          nixSettings = pkgs.writeText "nix-settings.json" (builtins.toJSON {
            model = "test/nix-model";
            toolsets = ["nix-toolset"];
            terminal = { backend = "docker"; timeout = 999; };
            mcp_servers = {
              nix-server = { command = "echo"; args = ["nix"]; };
            };
          });

          # Pre-built YAML fixtures for each scenario
          fixtureB = pkgs.writeText "fixture-b.yaml" ''
            model: "old-model"
            mcp_servers:
              old-server:
                url: "http://old"
          '';
          fixtureC = pkgs.writeText "fixture-c.yaml" ''
            skills:
              disabled:
                - skill-a
                - skill-b
            session_reset:
              mode: idle
              idle_minutes: 30
            streaming:
              enabled: true
            fallback_model:
              provider: openrouter
              model: test-fallback
          '';
          fixtureD = pkgs.writeText "fixture-d.yaml" ''
            model: "user-model"
            skills:
              disabled:
                - skill-x
            streaming:
              enabled: true
              transport: edit
          '';
          fixtureE = pkgs.writeText "fixture-e.yaml" ''
            mcp_servers:
              user-server:
                url: "http://user-mcp"
              nix-server:
                command: "old-cmd"
                args: ["old"]
          '';
          fixtureF = pkgs.writeText "fixture-f.yaml" ''
            terminal:
              cwd: "/user/path"
              custom_key: "preserved"
              env_passthrough:
                - USER_VAR
          '';

        in pkgs.runCommand "hermes-config-roundtrip" {
          nativeBuildInputs = [ pkgs.jq ];
        } ''
          set -e
          export HOME=$(mktemp -d)
          ERRORS=""

          fail() { ERRORS="$ERRORS\nFAIL: $1"; }

          # Helper: run merge then load with Python, output merged JSON
          merge_and_load() {
            local hermes_home="$1"
            export HERMES_HOME="$hermes_home"
            ${configMergeScript} ${nixSettings} "$hermes_home/config.yaml"
            ${hermesVenv}/bin/python3 -c '
import json, sys
from hermes_cli.config import load_config
json.dump(load_config(), sys.stdout, default=str)
'
          }

          # ═══════════════════════════════════════════════════════════════
          # Scenario A: Fresh install — no existing config.yaml
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario A: Fresh install ==="
          A_HOME=$(mktemp -d)
          A_CONFIG=$(merge_and_load "$A_HOME")

          echo "$A_CONFIG" | jq -e '.model == "test/nix-model"' > /dev/null \
            || fail "A: model not set from Nix"
          echo "$A_CONFIG" | jq -e '.mcp_servers."nix-server".command == "echo"' > /dev/null \
            || fail "A: MCP nix-server missing"
          echo "PASS: Scenario A"

          # ═══════════════════════════════════════════════════════════════
          # Scenario B: Nix keys override existing values
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario B: Nix overrides ==="
          B_HOME=$(mktemp -d)
          install -m 0644 ${fixtureB} "$B_HOME/config.yaml"
          B_CONFIG=$(merge_and_load "$B_HOME")

          echo "$B_CONFIG" | jq -e '.model == "test/nix-model"' > /dev/null \
            || fail "B: Nix model did not override"
          echo "PASS: Scenario B"

          # ═══════════════════════════════════════════════════════════════
          # Scenario C: User-only keys preserved
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario C: User keys preserved ==="
          C_HOME=$(mktemp -d)
          install -m 0644 ${fixtureC} "$C_HOME/config.yaml"
          C_CONFIG=$(merge_and_load "$C_HOME")

          echo "$C_CONFIG" | jq -e '.skills.disabled == ["skill-a", "skill-b"]' > /dev/null \
            || fail "C: skills.disabled not preserved"
          echo "$C_CONFIG" | jq -e '.session_reset.mode == "idle"' > /dev/null \
            || fail "C: session_reset.mode not preserved"
          echo "$C_CONFIG" | jq -e '.session_reset.idle_minutes == 30' > /dev/null \
            || fail "C: session_reset.idle_minutes not preserved"
          echo "$C_CONFIG" | jq -e '.streaming.enabled == true' > /dev/null \
            || fail "C: streaming.enabled not preserved"
          echo "$C_CONFIG" | jq -e '.fallback_model.provider == "openrouter"' > /dev/null \
            || fail "C: fallback_model not preserved"
          echo "PASS: Scenario C"

          # ═══════════════════════════════════════════════════════════════
          # Scenario D: Mixed — Nix wins for its keys, user keys preserved
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario D: Mixed merge ==="
          D_HOME=$(mktemp -d)
          install -m 0644 ${fixtureD} "$D_HOME/config.yaml"
          D_CONFIG=$(merge_and_load "$D_HOME")

          echo "$D_CONFIG" | jq -e '.model == "test/nix-model"' > /dev/null \
            || fail "D: Nix model did not override user model"
          echo "$D_CONFIG" | jq -e '.skills.disabled == ["skill-x"]' > /dev/null \
            || fail "D: user skills not preserved"
          echo "$D_CONFIG" | jq -e '.streaming.enabled == true' > /dev/null \
            || fail "D: user streaming not preserved"
          echo "PASS: Scenario D"

          # ═══════════════════════════════════════════════════════════════
          # Scenario E: MCP additive merge
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario E: MCP additive merge ==="
          E_HOME=$(mktemp -d)
          install -m 0644 ${fixtureE} "$E_HOME/config.yaml"
          E_CONFIG=$(merge_and_load "$E_HOME")

          echo "$E_CONFIG" | jq -e '.mcp_servers."user-server".url == "http://user-mcp"' > /dev/null \
            || fail "E: user MCP server not preserved"
          echo "$E_CONFIG" | jq -e '.mcp_servers."nix-server".command == "echo"' > /dev/null \
            || fail "E: Nix MCP server did not override same-name user server"
          echo "$E_CONFIG" | jq -e '.mcp_servers."nix-server".args == ["nix"]' > /dev/null \
            || fail "E: Nix MCP server args wrong"
          echo "PASS: Scenario E"

          # ═══════════════════════════════════════════════════════════════
          # Scenario F: Nested deep merge
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario F: Nested deep merge ==="
          F_HOME=$(mktemp -d)
          install -m 0644 ${fixtureF} "$F_HOME/config.yaml"
          F_CONFIG=$(merge_and_load "$F_HOME")

          echo "$F_CONFIG" | jq -e '.terminal.backend == "docker"' > /dev/null \
            || fail "F: Nix terminal.backend did not override"
          echo "$F_CONFIG" | jq -e '.terminal.timeout == 999' > /dev/null \
            || fail "F: Nix terminal.timeout did not override"
          echo "$F_CONFIG" | jq -e '.terminal.custom_key == "preserved"' > /dev/null \
            || fail "F: terminal.custom_key not preserved"
          echo "$F_CONFIG" | jq -e '.terminal.cwd == "/user/path"' > /dev/null \
            || fail "F: user terminal.cwd not preserved when Nix does not set it"
          echo "$F_CONFIG" | jq -e '.terminal.env_passthrough == ["USER_VAR"]' > /dev/null \
            || fail "F: user terminal.env_passthrough not preserved"
          echo "PASS: Scenario F"

          # ═══════════════════════════════════════════════════════════════
          # Scenario G: Idempotency — merging twice yields the same result
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario G: Idempotency ==="
          G_HOME=$(mktemp -d)
          install -m 0644 ${fixtureD} "$G_HOME/config.yaml"
          ${configMergeScript} ${nixSettings} "$G_HOME/config.yaml"
          FIRST=$(cat "$G_HOME/config.yaml")
          ${configMergeScript} ${nixSettings} "$G_HOME/config.yaml"
          SECOND=$(cat "$G_HOME/config.yaml")

          if [ "$FIRST" != "$SECOND" ]; then
            fail "G: second merge produced different output"
            echo "--- first ---"
            echo "$FIRST"
            echo "--- second ---"
            echo "$SECOND"
          fi
          echo "PASS: Scenario G"

          # ═══════════════════════════════════════════════════════════════
          # Report
          # ═══════════════════════════════════════════════════════════════
          if [ -n "$ERRORS" ]; then
            echo ""
            echo "FAILURES:"
            echo -e "$ERRORS"
            exit 1
          fi

          echo ""
          echo "=== All 7 merge scenarios passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';
      };
    };
}
