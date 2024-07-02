{
  description = "training environement for MLNanoshaper";

  # Nixpkgs / NixOS version to use.
  inputs.nixpkgs.url = "nixpkgs/nixos-24.05";

  outputs = { self, nixpkgs }:
	let
    	pkgs = nixpkgs.legacyPackages.x86_64-linux.pkgs;
	in 
    {
	devShells.x86_64-linux.default = pkgs.mkShell {
      packages = with pkgs; [
	  	parallel
      ];
	  };
	};
}
