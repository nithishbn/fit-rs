{
  description = "EC50 Curve Fitting Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Define Rust toolchain with specific components
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
          targets = [ "x86_64-unknown-linux-gnu" ];
        };

        # Development dependencies
        buildInputs = with pkgs; [
          # Core Rust toolchain
          rustToolchain

          # Essential build tools
          pkg-config
          openssl

          # Development tools
          bacon           # Background Rust code checker (replaces cargo-watch)

          # System dependencies for the project
          gcc
          clang
          llvm

          # Optional: Docker for containerization
          docker
          docker-compose
        ];

        # Library dependencies
        nativeBuildInputs = with pkgs; [
          pkg-config
        ];

        # Runtime library dependencies
        runtimeInputs = with pkgs; [
          openssl.dev
          fontconfig.dev
          freetype.dev
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          inherit buildInputs nativeBuildInputs;

          # Environment variables
          shellHook = ''
            echo "🦀 Welcome to the EC50 Curve Fitting Development Environment!"
            echo ""
            echo "Available tools:"
            echo "  • rustc $(rustc --version)"
            echo "  • cargo $(cargo --version)"
            echo "  • bacon - Background code checker with file watching"
            echo ""
            echo "Quick commands:"
            echo "  bacon           # Start background checker with file watching"
            echo "  cargo build     # Build the project"
            echo "  cargo test      # Run tests"
            echo ""
            echo "Stan model compilation enabled at runtime."
            echo "For build-time compilation, use: cargo build --features build-time-compile"
            echo ""

            # Set up environment for OpenSSL
            export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
            export OPENSSL_DIR="${pkgs.openssl.dev}"
            export OPENSSL_LIB_DIR="${pkgs.openssl.out}/lib"
            export OPENSSL_INCLUDE_DIR="${pkgs.openssl.dev}/include"

            # Set up Rust environment
            export RUST_SRC_PATH="${rustToolchain}/lib/rustlib/src/rust/library"
            export RUST_LOG="info"

            # Bacon configuration
            export BACON_PREFS="$PWD/.bacon-prefs"
          '';

          # Additional environment variables
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeInputs;
          FONTCONFIG_FILE = "${pkgs.fontconfig.out}/etc/fonts/fonts.conf";
        };

        # Optional: Define packages for building
        packages = {
          default = pkgs.rustPlatform.buildRustPackage {
            pname = "fit-rs";
            version = "0.1.0";
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;

            nativeBuildInputs = nativeBuildInputs;
            buildInputs = runtimeInputs;

            # Build both binaries
            buildFeatures = [ ];

            meta = with pkgs.lib; {
              description = "EC50 curve fitting with Bayesian inference";
              homepage = "https://github.com/nithishbn/fit-rs";
              license = licenses.mit;
              maintainers = [ ];
            };
          };
        };

        # Development apps
        apps = {
          # Background checker with file watching
          bacon-check = {
            type = "app";
            program = "${pkgs.writeShellScript "bacon-check" ''
              echo "Starting bacon background checker..."
              exec ${pkgs.bacon}/bin/bacon
            ''}";
          };
        };
      });
}
