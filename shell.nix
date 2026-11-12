# Optional: `nix-shell` gives you uv and curl; then run ./setup.sh (uv installs the pinned Python itself).
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell { packages = [ pkgs.uv pkgs.curl ]; }
