{
  description = "pkasolver — microstate pKa prediction via Graph Neural Networks";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, nixpkgs-unstable, flake-utils, uv2nix, pyproject-nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pkgs-unstable = nixpkgs-unstable.legacyPackages.${system};

        # ── Python interpreter ───────────────────────────────────────────
        python = pkgs.python311;

        # ── System libraries required by the Python deps ─────────────────
        # rdkit, cairosvg, and svgutils need these at runtime.
        sharedLibDeps = with pkgs; [
          cairo
          pango
          glib
          zlib
          stdenv.cc.cc.lib
        ];

        # ── Base dev shell (CPU-only PyTorch) ────────────────────────────
        # uv manages the Python virtualenv; Nix provides system libraries.
        # Run `uv sync` inside the shell to set up the venv.
        baseShell = pkgs.mkShell {
          name = "pkasolver-cpu";
          packages = [
            python
            pkgs.uv
            pkgs.git
          ] ++ sharedLibDeps;

          shellHook = ''
            echo "pkasolver dev shell (CPU)"
            echo "Run: uv sync --extra dev  to set up the virtualenv"
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath sharedLibDeps}:$LD_LIBRARY_PATH"
            # Point uv at the Nix-provided Python so it uses the right interpreter
            export UV_PYTHON="${python}/bin/python"
          '';
        };

        # ── CUDA dev shell ────────────────────────────────────────────────
        # Requires CUDA drivers on the host. Nix provides the toolkit headers
        # and libraries; PyTorch CUDA wheels come via uv/pip.
        cudaShell =
          let
            cudaPkgs = pkgs.cudaPackages;
          in
          pkgs.mkShell {
            name = "pkasolver-cuda";
            packages = [
              python
              pkgs.uv
              pkgs.git
              cudaPkgs.cudatoolkit
              cudaPkgs.cudnn
            ] ++ sharedLibDeps;

            shellHook = ''
              echo "pkasolver dev shell (CUDA)"
              echo "Run: uv sync --extra dev  to set up the virtualenv"
              echo "Then install CUDA PyTorch:"
              echo "  uv pip install torch --index-url https://download.pytorch.org/whl/cu121"
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath sharedLibDeps}:$LD_LIBRARY_PATH"
              export CUDA_HOME="${cudaPkgs.cudatoolkit}"
              export UV_PYTHON="${python}/bin/python"
            '';
          };

        # ── ROCm dev shell ────────────────────────────────────────────────
        # Requires ROCm drivers on the host. Nix provides ROCm libraries;
        # PyTorch ROCm wheels come via uv/pip.
        rocmShell = pkgs.mkShell {
          name = "pkasolver-rocm";
          packages = [
            python
            pkgs.uv
            pkgs.git
            pkgs.rocmPackages.rocm-runtime
            pkgs.rocmPackages.rocm-smi
          ] ++ sharedLibDeps;

          shellHook = ''
            echo "pkasolver dev shell (ROCm)"
            echo "Run: uv sync --extra dev  to set up the virtualenv"
            echo "Then install ROCm PyTorch:"
            echo "  uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath sharedLibDeps}:$LD_LIBRARY_PATH"
            export UV_PYTHON="${python}/bin/python"
          '';
        };

      in
      {
        # ── Dev shells ────────────────────────────────────────────────────
        devShells = {
          default = baseShell;
          cpu = baseShell;
          cuda = cudaShell;
          rocm = rocmShell;
        };

        # ── Formatter ─────────────────────────────────────────────────────
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
