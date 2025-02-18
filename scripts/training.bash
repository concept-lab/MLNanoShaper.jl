#!/bin/bash
batch_name=smooth_fixed_1
echo training $batch_name

parallel --dry-run --jobs 1\
  	julia --project=scripts -p 12 scripts/cli.jl\
		--model {1}\
		--cutoff_radius {2}\
		--learning_rate {3}\
		--van_der_waals_channel {4}\
		--loss {5}\
	::: tiny_angular_dense\
	::: 3.0\
	::: 1e-4\
 	::: false\
	::: categorical continuous

# --name {2}A_${batch_name}_{5}
parallel --jobs 3\
	julia --project=scripts -p 12 scripts/cli.jl\
		--model {1}\
		--cutoff_radius {2}\
		--learning_rate {3}\
		--name {2}A_${batch_name}_{5}\
		--van_der_waals_channel {4}\
		--loss {5}\
	::: tiny_angular_dense light_angular_dense \
	::: 2.0\
	::: 1e-4\
 	::: false\
	::: categorical continuous

# parallel --jobs 2\
# 	julia --project=scripts -p 12 scripts/cli.jl\
# 		--model {1}\
# 		--cutoff_radius {2}\
# 		--learning_rate {3}\
# 		--name {2}A_$batch_name\
# 		--van_der_waals_channel {4}\
# 		--categorical {5}\
# 	::: light_angular_dense medium_angular_dense\
# 	::: 3.0\
# 	::: 2e-4\
#  	::: false\
# 	::: true false


 parallel --jobs 1\
	 julia --project=scripts -p 12 scripts/cli.jl\
		 --model {1}\
		 --cutoff_radius {2}\
		 --learning_rate {3}\
		 --name {2}A_${batch_name}_{5}\
		 --van_der_waals_channel {4}\
		 --loss {5}\
	::: medium_angular_dense\
	::: 3.0\
	::: 1e-4\
	::: false\
	::: categorical continuous

 parallel --jobs 1\
	 julia --project=scripts -p 12 scripts/cli.jl\
		 --model {1}\
		 --cutoff_radius {2}\
		 --learning_rate {3}\
		 --name {2}A_${batch_name}_{5}\
		 --van_der_waals_channel {4}\
		 --loss {5}\
	::: tiny_angular_dense light_angular_dense \
	::: 4.0\
	::: 1e-4\
	::: false\
	::: categorical continuous
