{
  description = "pkasolver — microstate pKa prediction via Graph Neural Networks";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-25.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { pkgs, system, ... }:
        let
          python = pkgs.python311;

          # Separate nixpkgs instance with allowUnfree for CUDA packages.
          pkgs-unfree = import inputs.nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };

          # System libraries required by rdkit, cairosvg, svgutils, and the
          # X11 drawing backend.
          sharedLibDeps = with pkgs; [
            cairo
            pango
            glib
            zlib
            stdenv.cc.cc.lib
            xorg.libXrender
            xorg.libX11
            xorg.libXext
          ];

          # Shell hook shared across all variants.
          # Sets LD_LIBRARY_PATH; runs uv sync only in interactive shells
          # (not during direnv's environment export phase).
          commonHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath sharedLibDeps}:$LD_LIBRARY_PATH"
            export UV_PYTHON="${python}/bin/python"
            # Only run uv sync in interactive shells, not during direnv export.
            # direnv sets IN_NIX_SHELL=impure during its eval phase but PS1 is unset.
            if [ -n "''${PS1:-}" ]; then
              uv sync --extra dev --extra cpu --quiet
              source .venv/bin/activate
              echo "Ready. Python: $(python --version), pkasolver: $(python -c 'import pkasolver; print(pkasolver.__version__)' 2>/dev/null || echo 'not installed')"
            fi
          '';

        in
        {
          # ── Dev shells ──────────────────────────────────────────────────

          devShells = {

            # CPU (default) ───────────────────────────────────────────────
            default = pkgs.mkShell {
              name = "pkasolver-cpu";
              packages = [ python pkgs.uv pkgs.git ] ++ sharedLibDeps;
              shellHook = commonHook;
            };

            cpu = pkgs.mkShell {
              name = "pkasolver-cpu";
              packages = [ python pkgs.uv pkgs.git ] ++ sharedLibDeps;
              shellHook = commonHook;
            };

            # CUDA ────────────────────────────────────────────────────────
            # Nix provides the CUDA toolkit; PyTorch CUDA wheels come via uv.
            # Uses a separate pkgs instance with allowUnfree = true since the
            # CUDA toolkit is licensed under the CUDA EULA.
            cuda =
              let cudaPkgs = pkgs-unfree.cudaPackages; in
              pkgs.mkShell {
                name = "pkasolver-cuda";
                packages = [
                  python pkgs.uv pkgs.git
                  cudaPkgs.cudatoolkit
                  cudaPkgs.cudnn
                ] ++ sharedLibDeps;
                shellHook = commonHook + ''
                  export CUDA_HOME="${cudaPkgs.cudatoolkit}"
                  echo "CUDA shell active. To install CUDA-enabled PyTorch:"
                  echo "  uv pip install torch --index-url https://download.pytorch.org/whl/cu121"
                '';
              };

            # ROCm ────────────────────────────────────────────────────────
            # Nix provides ROCm runtime; PyTorch ROCm wheels come via uv.
            rocm = pkgs.mkShell {
              name = "pkasolver-rocm";
              packages = [
                python pkgs.uv pkgs.git
                pkgs.rocmPackages.rocm-runtime
                pkgs.rocmPackages.rocm-smi
              ] ++ sharedLibDeps;
              shellHook = commonHook + ''
                echo "ROCm shell active. To install ROCm-enabled PyTorch:"
                echo "  uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"
              '';
            };
          };

          # ── Formatter ───────────────────────────────────────────────────
          formatter = pkgs.nixpkgs-fmt;
        };
    };
}
