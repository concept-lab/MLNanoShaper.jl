# MLNanoShaper

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://concept-lab.github.io/MLNanoShaper.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://concept-lab.github.io/MLNanoShaper.jl/dev/)
[![Build Status](https://github.com/concept-lab/MLNanoShaper.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/hack-hard/MLNanoShaper.jl/actions/workflows/CI.yml?query=branch%3Amain)

MLnanoshaper is an experimental sofware designed to compute proteins surface like the Nanoshaper program but with less computations using Machine Learning.
Instead of having a probe (a virtual water molecule) that rolls over the surface, like in Nanoshaper, we will have a machine learning algorithme will generate a scalar "energy" field from the atoms. We will then define the surface as an implicit surface ie $\\{x/ f(x)= 0\\}$ that will be triangulated using marching cubes.


# To use this project
- clone this repository:
```
    git clone https://github.com/concept-lab/MLNanoShaper.jl
```
- clone the submodule
```
    git submodule init
    git submodule update
```
- download dependency:
```
    julia -e "using Pkg; Pkg.activate(.);Pkg.instantiate(.)"
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
    curl https://zenodo.org/records/12772809/files/shrec.tar.gz --output datasets/shrec.tar.gz 
    tar -xzf shrec.tar.gz datasets/pqr
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
    julia -e"using Pluto; Pluto.run()"
```

Each notebooks will use the environement in scripts.

# Interfaceing with C interface
We have an example of code in MLNanoShaperRunner/example. Here is the commands to compile the example.
## Compiling the C interface.
```
    julia --project MLNanoShaperRunner/build/build.jl
```

## Copy the code to examples
```
    mkdir MLNanoShaperRunner/build/build
    cp -r MLNanoShaperRunner/build/lib/lib MLNanoShaperRunner/build/build/
```

## Compiling the C code
``` 
    clang -I MLNanoShaperRunner/build/build \
        -rpath MLNanoShaperRunner/build/build \
        -L MLNanoShaperRunner/build/build \
        -l MLNanoShaperRunner \
        MLNanoShaperRunner/examples/dummy_example.c \
        -o MLNanoShaperRunner/examples/test
```

## Copy artifacts
The julia code needs access to some artifacts to load correcly
```
    cp -r ~/.julia/artifacts/d5b30ebced3f3de269a3489e10c85c81eae13b0d/ MLNanoShaperRunner/build/build/julia/artifacts
    cp -r ~/.julia/artifacts/abf4b5086b4eb867021118c85b2cc11a15b764a9/ MLNanoShaperRunner/build/build/julia/artifacts
    cp -r ~/.julia/artifacts/9cfa1f93276d8e380806650071f8447e8e38301f/ MLNanoShaperRunner/build/build/julia/artifacts
    cp -r ~/.julia/artifacts/69059e078be18d2f90e9876662a7672df0784b19/ MLNanoShaperRunner/build/build/julia/artifacts
```

## Run the code
```
    MLNanoShaperRunner/build/build/test
```

