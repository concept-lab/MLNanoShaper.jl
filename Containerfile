FROM docker.io/library/julia
VOLUME ~/datasets
RUN mkdir ~/datasets/pqr && mkdir ~/datasets/logs && ~/datasets/models
RUN curl https://zenodo.org/records/14886938/files/shrec.tar.gz --output ~/datasets/shrec.tar.gz \
  &&  tar -xzf ~/datasets/shrec.tar.gz -C ~/datasets \
  &&  rm ~/datasets/shrec.tar.gz
COPY git@github.com/concept-lab/MLNanoShaper.jl .
COPY git@github.com/concept-lab/MLNanoShaperRunner.jl MLNanoShaper.jl
RUN julia --project -e 'using Pkg; pkg"instantiate"'
RUN julia --project=MLNanoShaperRunner -e 'using Pkg; pkg"instantiate"'
RUN julia --project=scripts -e 'using Pkg; Pkg.instantiate(); using CUDA; CUDA.set_runtime_version!(v"12.6.2"); CUDA.precompile_runtime()'
CMD julia
