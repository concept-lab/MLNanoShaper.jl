#!/bin/bash
batch_name=small_grid_4
parallel --jobs 4\
	~/.julia/bin/mlnanoshaper train\
		--model={1}\
		--cutoff-radius={2}\
		--learning-rate={3}\
		--name={2}A_$batch_name\
		--model-kargs.van_der_wal_channel {4}\
		--categorical {5}\
	::: tiny_angular_dense light_angular_dense medium_andular_dense\
	::: 2.0\
	::: 1e-5\
 	::: true false\
	::: true false

parallel --jobs 3\
	~/.julia/bin/mlnanoshaper train\
		--model={1}\
		--cutoff-radius={2}\
		--learning-rate={3}\
		--name={2}A_$batch_name\
		--model-kargs.van_der_wal_channel {4}\
		--categorical {5}\
	::: tiny_angular_dense \
	::: 3.0\
	::: 2e-6\
 	::: true false\
	::: true false

parallel --jobs 2\
	~/.julia/bin/mlnanoshaper train\
		--model={1}\
		--cutoff-radius={2}\
		--learning-rate={3}\
		--name={2}A_$batch_name\
		--model-kargs.van_der_wal_channel {4}\
		--categorical {5}\
	::: light_angular_dense medium_andular_dense\
	::: 3.0\
	::: 2e-6\
 	::: true false\
	::: true false


 parallel --jobs 1\
	 ~/.julia/bin/mlnanoshaper train\
		 --model={1}\
		 --cutoff-radius={2}\
		 --learning-rate={3}\
		 --name={2}A_$batch_name\
		 --model-kargs.van_der_wal_channel{4}\
		 --categorical {5}\
	::: tiny_angular_dense light_angular_dense medium_angular_dense\
	::: 4.0\
	::: 1e-7\
	::: true false\
	:::  true false
