FROM docker.io/library/julia
RUN apt-get update && apt-get install parallel -y
COPY . .
RUN mkdir ~/datasets
RUN mkdir ~/datasets/models
RUN mkdir ~/datasets/logs
RUN mkdir ~/datasets/pqr
RUN curl https://zenodo.org/records/12772809/files/shrec.tar.gz --output ~/datasets/shrec.tar.gz; tar -xzf ~/datasets/shrec.tar.gz -C ~/datasets/pqr
RUN julia --project -e 'using Pkg; pkg"instantiate"'
RUN julia --project=MLNanoShaperRunner -e 'using Pkg; pkg"instantiate";pkg"resolve"'
RUN julia --project=MLNanoShaperRunner -e 'using Pkg; pkg"resolve"'
RUN julia --project -e 'using Pkg;pkg"resolve"'
RUN julia --project build/build.jl install
CMD scripts/training.bash
