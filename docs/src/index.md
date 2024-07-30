```@meta
CurrentModule = MLNanoShaper
```

# MLNanoShaper

Documentation for [MLNanoShaper](https://github.com/hack-hard/MLNanoShaper.jl). The main way of interfacing with the training is through the cli.

In order to launch training with default values:
```
~/.julia/bin/mlnanoshaper train 
```

There are 2 ways to modify parameters.
- modifying default values in `param/param.toml`. 
- modifying some values using flags in the command line interface 


```@index
```

```@autodocs
Modules = [MLNanoShaper,MLNanoShaperRunner]
```
