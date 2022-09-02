cd(@__DIR__)

using Distributions
using Printf
using Random
using JLD2

ENV["JULIA_CUDA_USE_BINARYBUILDER"] = false

Random.seed!(parse(Int, ARGS[3]))

const L = parse(Int, ARGS[2])
const λ = 4.0f0
const Γ = 1.0f0
const T = 1.0f0

const Δt = 0.04f0/Γ
const Rate = Float32(sqrt(2.0*Δt*Γ))

function ΔH(x, φ, φt, m²)
    @inbounds φold = φ[x[1], x[2], x[3]]
    Δφ = φt - φold
    Δφ² = φt^2 - φold^2

    @inbounds ∑nn = φ[x[1]%L+1, x[2], x[3]] + φ[x[1], x[2]%L+1, x[3]] + φ[x[1], x[2], x[3]%L+1]
    @inbounds ∑nn += φ[(x[1]+L-2)%L+1, x[2], x[3]] + φ[x[1], (x[2]+L-2)%L+1, x[3]] + φ[x[1], x[2], (x[3]+L-2)%L+1]

    3Δφ² - Δφ * ∑nn + 0.5m² * Δφ² + 0.25λ * (φt^4 - φold^4)
end

function step(m², φ, x)
    δ = Rate*rand(Normal())
    @inbounds φnew = φ[x[1], x[2], x[3]] + δ

    P = min(1.0f0, exp(-ΔH(x, φ, φnew, m²)/T))
    r = rand(Float32)
    if (r < P)
        @inbounds φ[x[1], x[2], x[3]] = φnew
    end
end

function sweep(m², φ, L)
	Threads.@threads for i in 1:L
		for j in 1:L
			for k in 1:L
				if (i+j+k)%2 == 0
					step(m², φ, (i,j,k))
				end
			end
		end
	end

	Threads.@threads for i in 1:L
		for j in 1:L
			for k in 1:L
				if (i+j+k)%2 !=0
					step(m², φ, (i,j,k))
				end
			end
		end
	end
end

function thermalize(m², φ, L, N=10000)
    for i in 1:N
        sweep(m², φ, L)
    end
end

M(φ) = 2/L^3*sum(φ[:,:,1:L÷2])

# near critical value
m² = -2.0

maxt = L^2*500
skip = 10
batch = parse(Int, ARGS[4])

for series in 1:16
	df = load("/share/tmschaef/jkott/modelB/KZ/IC_sym_L_$L"*"_id_"*ARGS[1]*"_series_$series.jld2")

	for run in (16batch-15):16batch
		ϕ = df["ϕ"]

		thermalize(m², ϕ, L, 1.5 * 10^4)

		open("/share/tmschaef/jkott/modelB/KZ/cumulants/static/sum_L_$L"*"_id_"*ARGS[1]*"_series_$series"*"_run_$run.dat","w") do io 
			for i in 0:div(maxt,skip)
				Printf.@printf(io, "%i %f\n", i*skip, M(ϕ))
				thermalize(m², ϕ, L, skip)
			end
		end
	end
end