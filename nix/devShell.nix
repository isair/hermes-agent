# nix/devShell.nix — Dev shell that delegates setup to each package
#
# Each package in inputsFrom might expose passthru.devShellHook — a bash snippet
# with stamp-checked setup logic. This file collects and runs them all.
#
# The npm root-level hook (rootDevShellHook) runs AFTER the per-package hooks
# so that npm install (from a package.json change) can update the lockfile first,
# then the root hook detects the change and runs npm ci + fix-lockfiles in the
# same shell entry.
{ ... }:
{
  perSystem =
    { pkgs, self', ... }:
    let
      packages = builtins.attrValues self'.packages;
      hermesNpmLib = self'.packages.default.passthru.hermesNpmLib;
      fixLockfilesExe = pkgs.lib.getExe self'.packages.fix-lockfiles;
    in
    {
      devShells.default = pkgs.mkShell {
        inputsFrom = packages;
        packages = with pkgs; [
          uv
        ];
        shellHook =
          let
            hooks = map (p: p.passthru.devShellHook or "") packages;
            combined = pkgs.lib.concatStringsSep "\n" (builtins.filter (h: h != "") hooks);
          in
          ''
            echo "Hermes Agent dev shell"
            ${combined}
            ${hermesNpmLib.rootDevShellHook fixLockfilesExe}
            echo "Ready. Run 'hermes' to start."
          '';
      };
    };
}
