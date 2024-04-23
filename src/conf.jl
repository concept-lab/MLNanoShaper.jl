using TOML

params = "$( dirname(dirname(@__FILE__)))/param/param.toml"
conf = TOML.parsefile(params)
datadir ="$(homedir())/datasets/proteins/"
