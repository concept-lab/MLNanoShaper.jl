# MLNanoShaper

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://concept-lab.github.io/MLNanoShaper.jl/dev/)
[![Build Status](https://github.com/concept-lab/MLNanoShaper.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/hack-hard/MLNanoShaper.jl/actions/workflows/CI.yml?query=branch%3Amain)

MLnanoshaper is an experimental sofware designed to compute proteins surface like the Nanoshaper program but with less computations using Machine Learning.
Instead of having a probe (a virtual water molecule) that rolls over the surface, like in Nanoshaper, we will have a machine learning algorithme will generate a scalar "energy" field from the atoms. We will then define the surface as an implicit surface ie $\\{x/ f(x)= 0\\}$ that will be triangulated using marching cubes.


# To use this project
- install julia
```
julia --project MLNanoShaperRunner/build/build.jl
```
- clone this repository:
```
git clone https://github.com/concept-lab/MLNanoShaper.jl
```
- clone the submodule
```
cd MLNanoShaper.jl
git submodule init
git submodule update
```
- download dependency:
```
julia -e 'using Pkg; Pkg.activate(".");Pkg.instantiate()'
```
# make dirs
```
mkdir ~/datasets
mkdir ~/datasets/models
mkdir ~/datasets/logs
mkdir ~/datasets/pqr
```
# install dataset
```
curl https://zenodo.org/records/12772809/files/shrec.tar.gz --output ~/datasets/shrec.tar.gz 
tar -xzf ~/datasets/shrec.tar.gz -C ~/datasets/pqr
```

# To train 
install parallel and run commands
```
julia --project ./build/build.jl install
./scripts/training.bash
```

By default, logs and weights should be stored in ~/datasets/logs/ and ~/datasets/models


# Notebooks
The ./scripts/ folder contains some Pluto Notebooks. Theses are for creating figures and for debugging. To execute these notebooks, you should install pluto and launch server with 
```
julia -e "using Pluto; Pluto.run()"
```

Each notebooks will use the environement in scripts.

# Interfaceing with C interface
We have an example of code in MLNanoShaperRunner/example. Here is the commands to compile the example.
## Compiling the C interface.
```
julia --project MLNanoShaperRunner/build/build.jl
```

## Make a dicectory for compilation
```
mkdir MLNanoShaperRunner/build/build
```

## Compiling the C code
``` 
clang MLNanoShaperRunner/examples/dummy_example.c \
    -I MLNanoShaperRunner/build/lib/include \
    -L MLNanoShaperRunner/build/lib/lib/ \
    -l MLNanoShaperRunner \
    -o MLNanoShaperRunner/build/build/test
```

## Copy artifacts
The julia code needs access to some artifacts to load correcly
```
cp -r MLNanoShaperRunner/build/lib/share MLNanoShaperRunner/build/build/share
cp -r MLNanoShaperRunner/build/lib/lib/* MLNanoShaperRunner/build/build
cp MLNanoShaperRunner/examples/tiny* MLNanoShaperRunner/build/build

```

## Run the code
```
    MLNanoShaperRunner/build/build/test
```
## Weights
weights are on `the https://zenodo.org/records/13222088`
