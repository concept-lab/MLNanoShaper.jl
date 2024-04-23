using TOML

conf = TOML.parsefile("param/param.toml")
datadir ="$(homedir())/datasets/proteins/"
