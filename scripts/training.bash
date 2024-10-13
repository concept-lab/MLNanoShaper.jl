#!/bin/bash
batch_name=smooth_14
echo training $batch_name

parallel --jobs 1\
	julia --project src/cli.jl\
		--model={1}\
		--cutoff-radius={2}\
		--learning-rate={3}\
		--name={2}A_${batch_name}_{5}\
		--model-kargs.van_der_waals_channel {4}\
		--loss {5}\
	::: tiny_angular_dense \
	::: 3.0\
	::: 1e-4\
 	::: false\
	::: categorical continuous

parallel --jobs 3\
	julia --project src/cli.jl\
		--model={1}\
		--cutoff-radius={2}\
		--learning-rate={3}\
		--name={2}A_${batch_name}_{5}\
		--model-kargs.van_der_waals_channel {4}\
		--loss {5}\
	::: tiny_angular_dense light_angular_dense \
	::: 2.0\
	::: 1e-4\
 	::: false\
	::: categorical continuous

# parallel --jobs 2\
# 	julia --project src/cli.jl\
# 		--model={1}\
# 		--cutoff-radius={2}\
# 		--learning-rate={3}\
# 		--name={2}A_$batch_name\
# 		--model-kargs.van_der_waals_channel {4}\
# 		--categorical {5}\
# 	::: light_angular_dense medium_angular_dense\
# 	::: 3.0\
# 	::: 2e-4\
#  	::: false\
# 	::: true false


 parallel --jobs 1\
	 julia --project src/cli.jl\
		 --model={1}\
		 --cutoff-radius={2}\
		 --learning-rate={3}\
		 --name={2}A_${batch_name}_{5}\
		 --model-kargs.van_der_waals_channel {4}\
		 --loss {5}\
	::: medium_angular_dense\
	::: 3.0\
	::: 1e-4\
	::: false\
	::: categorical continuous

 parallel --jobs 1\
	 julia --project src/cli.jl\
		 --model={1}\
		 --cutoff-radius={2}\
		 --learning-rate={3}\
		 --name={2}A_${batch_name}_{5}\
		 --model-kargs.van_der_waals_channel {4}\
		 --loss {5}\
	::: tiny_angular_dense light_angular_dense \
	::: 4.0\
	::: 1e-4\
	::: false\
	::: categorical continuous
