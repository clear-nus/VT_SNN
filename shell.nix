let
  pkgs = import <nixpkgs> {};
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    python37Packages.python-language-server
    python37Packages.pyls-black
    python37Packages.black
  ];
}
