cd(@__DIR__)
using DelimitedFiles
using Random
using Glob
using LinearAlgebra

function autocor_loc_2(x, beg, max, n=2)
	C = zeros(ComplexF64,max+1)
	N = zeros(Int64,max+1)
	for tau in 0:max
		for i in beg:length(x)-max
			j = i + tau
			@inbounds C[tau+1] = C[tau+1] +  (x[i]*conj(x[j]))^n
			@inbounds N[tau+1] = N[tau+1] + 1
		end
	end
	real(C) ./ N
end

function corr(dataset, L)
	F(i) = autocor_loc_2(dataset[:,2+2i].+(-1)^(i==L÷2)*1.0im.*dataset[:,3+2i], 1, L^4÷40, 1)
	corrs = hcat([F(i) for i in 0:L÷2]...)
	corrs
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
    len = length(fs[1,:])
    bs_tot = []
    for i in 1:M
        bsD = []
        for j in 1:len
            samp = fs[:,rand(1:len)]
            push!(bsD, samp)
        end
        push!(bs_tot, average(bsD))
    end
    mean = average(bs_tot)
    var = variance(bs_tot)
    output = cat(mean, sqrt.(var); dims=3)
    output
end

function collect_data(L)
	dfs = glob("data/dynamics_L_$(L)_id_*")
	datasets = cat([corr(readdlm(file, ' '), L) for file in dfs]...; dims=3)

	len = length(datasets[1,:,1]) # num of modes
	C = hcat([bootstrap(datasets[:,i,:], 100) for i in 1:len]...)

	step = 10
	Δt = 0.04
	t = [0:length(C[:,1,1])-1;] .* (step*Δt)
	k = [0:L÷2;] .* (2π/L)
	(t, k, C[:,:,1], C[:,:,2])
end

function save_data(L)
	(t, k, C, err) = collect_data(L)

	output_file = open("data_k_$L.jl","w")

	write(output_file, "t_$L = ")
	show(output_file, t)
	write(output_file, "\n\n")

	write(output_file, "k_$L = ")
	show(output_file, k)
	write(output_file, "\n\n")

	write(output_file, "C_$L = ")
	show(output_file, C)
	write(output_file, "\n\n")

	write(output_file, "Cerr_$L = ")
	show(output_file, err)
	write(output_file, "\n\n")

	close(output_file)
end

save_data(parse(Int, ARGS[1]))