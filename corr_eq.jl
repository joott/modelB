cd("/share/tmschaef/jkott/modelB")
using DelimitedFiles
using Random
using Glob
using Statistics

function corr(dataset, L)
	G = zeros(div(L,2)+1)
	Threads.@threads for i in 1:div(L,2)+1
		G[i] = mean(dataset[:,2i].^2 .+ dataset[:,1+2i].^2)
	end
	G
end

function variance(x)
	S=zeros(length(x[1]))
	mean = average(x)
	for i in 1:length(x)
		S .= S .+ (x[i] .- mean) .^2
	end
	S./length(x)
end

function average(x)
	sum(x)/length(x)
end

function bootstrap(fs, M)
	len = length(fs)
	bs_tot = []
	for i in 1:M
		bsD = []
		for j in 1:len
			samp = fs[rand(1:len)]
			push!(bsD, samp)
		end
		push!(bs_tot, average(bsD))
	end
	mean = average(bs_tot)
	var = variance(bs_tot)
	(mean, sqrt.(var))
end

function collect_data(L, m)
	dfs = glob("dynamics_L_$(L)_m_$(m)_id_*")
	datasets = []
	for file in dfs
		df = readdlm(file,' ')
		G = corr(df, L)
		push!(datasets, G)
	end

	(G, Gerr) = bootstrap(datasets, 100)
	k = [0:div(L,2);] * (2pi/L) |> x -> 2*sin.(x/2)
	
	(k, G, Gerr)
end

function save_data(L, m)
	(k, G, err) = collect_data(L, m)

	output_file = open("data_$(L).jl","w")

	write(output_file, "k_$L = ")
	show(output_file, k)
	write(output_file, "\n\n")

	write(output_file, "G_$L = ")
	show(output_file, G)
	write(output_file, "\n\n")

	write(output_file, "Gerr_$L = ")
	show(output_file, err)

	close(output_file)
end

save_data(parse(Int, ARGS[1]), parse(Int, ARGS[2]))
