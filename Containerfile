FROM docker.io/library/julia
RUN mkdir ~/datasets &&  mkdir ~/datasets/pqr && mkdir ~/datasets/logs && mkdir ~/datasets/models
RUN curl https://zenodo.org/records/14886938/files/shrec.tar.gz --output ~/datasets/shrec.tar.gz \
  &&  tar -xzf ~/datasets/shrec.tar.gz -C ~/datasets \
  &&  rm ~/datasets/shrec.tar.gz
ADD https://github.com/concept-lab/MLNanoShaper.jl.git /MLNanoShaper.jl
ADD https://github.com/concept-lab/MLNanoShaperRunner.jl.git /MLNanoShaper.jl/MLNanoShaperRunner
RUN julia --project=MLNanoShaper.jl/scripts -e 'using Pkg; Pkg.instantiate()'
# RUN julia --project=MLNanoShaper.jl -e 'using Pkg;Pkg.instantiate()'
# RUN julia --project=MLNanoShaper.jl/MLNanoShaperRunner -e 'using Pkg;Pkg.instantiate()'
# RUN julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
CMD julia --project=MLNanoShaper.jl/scripts
