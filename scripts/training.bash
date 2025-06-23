#!/bin/bash

parallel --dry-run \
  	julia --project=scripts -p 6 scripts/cli.jl\
		--model {1}\
		--cutoff_radius {2}\
    --batch_size {3}\
    --learning_rate 1e-4\
		--loss {4}\
		--on_gpu\
	::: tiny_angular_dense light_angular_dense\
	::: 3.0 4.0 5.0\
	:::+ 1000000 100000 50000\
	::: categorical continuous
