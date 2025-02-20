FROM docker.io/library/julia
RUN apt-get update && apt-get install parallel -y
# RUN mkdir ~/datasets && mkdir ~/datasets/models && mkdir ~/datasets/logs && mkdir ~/datasets/pqr
# RUN curl https://zenodo.org/records/14886938/files/shrec.tar.gz --output ~/datasets/shrec.tar.gz \
  # &&  tar -xzf ~/datasets/shrec.tar.gz -C ~/datasets \
  # &&  rm ~/datasets/shrec.tar.gz
COPY . .
RUN julia --project -e 'using Pkg; pkg"instantiate"'
RUN julia --project=MLNanoShaperRunner -e 'using Pkg; pkg"instantiate"'
RUN julia --project=scripts -e 'using Pkg; Pkg.instantiate(); using CUDA; CUDA.set_runtime_version!(v"12.6.2"); CUDA.precompile_runtime()'
CMD scripts/training.bash
