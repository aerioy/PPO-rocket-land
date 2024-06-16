{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  packages = with pkgs; [ git gh ];

  languages.javascript.enable = true;
  services.postgres = {
    enable = true;
    listen_addresses = "0.0.0.0";
    initialScript = "CREATE ROLE postgres SUPERUSER; ALTER ROLE postgres WITH LOGIN;";
    initialDatabases = [{ name = "dbthing"; }];
  };
 
  enterShell = ''
  '';

  # https://devenv.sh/tests/
  enterTest = ''
  '';
}
