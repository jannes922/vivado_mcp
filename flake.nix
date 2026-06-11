{
  description = "MCP server for AMD/Xilinx Vivado FPGA development";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (pkgs: rec {
        vivado-mcp = pkgs.python3Packages.buildPythonApplication {
          pname = "vivado-mcp";
          version = "0.1.0";
          pyproject = true;
          src = self;

          build-system = [ pkgs.python3Packages.setuptools ];

          dependencies = with pkgs.python3Packages; [
            mcp
            pexpect
            psutil
          ];

          nativeCheckInputs = [ pkgs.python3Packages.pytestCheckHook ];

          pythonImportsCheck = [ "vivado_mcp" ];

          meta = {
            description = "MCP server for AMD/Xilinx Vivado FPGA development";
            homepage = "https://github.com/coreyhahn/vivado_mcp";
            license = nixpkgs.lib.licenses.mit;
            mainProgram = "vivado-mcp";
          };
        };
        default = vivado-mcp;
      });

      apps = forAllSystems (pkgs: {
        default = {
          type = "app";
          program = nixpkgs.lib.getExe self.packages.${pkgs.stdenv.hostPlatform.system}.default;
        };
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: with ps; [ mcp pexpect psutil pytest ]))
          ];
        };
      });

      checks = forAllSystems (pkgs: {
        package = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      });
    };
}
