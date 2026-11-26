{
  description = "tonepoet-true-peak - streaming true-peak and loudness measurement for decoded PCM audio";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        # Pinned rather than "latest", so the compiler changes only when this
        # line does. Rust 1.93 and 1.94 crash in the LLVM loop vectorizer while
        # compiling this crate at opt-level 3; 1.95 and later do not.
        rustToolchain = pkgs.rust-bin.stable."1.99.1".default.override {
          extensions = [ "rust-src" "clippy" "rustfmt" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ rustToolchain ];
        };
      });
}
