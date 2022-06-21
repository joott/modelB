cd(@__DIR__)

using Distributions
using Printf
using FFTW
using JLD2
using Random
using CUDA
using BenchmarkTools

# ENV["JULIA_CUDA_USE_BINARYBUILDER"] = false

# Random.seed!(parse(Int, ARGS[3]))

# const L = parse(Int, ARGS[2]) # must be a multiple of 4
L = 32
const λ = 4.0e0
const Γ = 1.0e0
const T = 1.0e0
const z = 3.9e0

const Δt = 0.04e0/Γ
const Rate = Float64(sqrt(2.0*Δt*Γ))
ξ = Normal(0.0e0, 1.0e0)

# KZ protocol variables
const m²c, m²0, m²e = -2.28587, -2.0e0, -4.0e0
const m_a, m_b = begin
    t_c = 0.5 * 10^-3 * L^z * (1 - m²0/m²c)
    τ_Q = t_c * m²c / (m²c - m²0)

    m²c/τ_Q, m²0
end

const t_e = (m²e - m_b)/m_a
##

function hotstart(n)
	rand(ξ, n, n, n)
end

function ΔH(x, ϕ, q, m², L)
	@inbounds ϕold = ϕ[x...]
	ϕt = ϕold + q
	Δϕ = ϕt - ϕold
	Δϕ² = ϕt^2 - ϕold^2

    @inbounds ∑nn = ϕ[x[1]%L+1, x[2], x[3]] + ϕ[x[1], x[2]%L+1, x[3]] + ϕ[x[1], x[2], x[3]%L+1] + ϕ[(x[1]+L-2)%L+1, x[2], x[3]] + ϕ[x[1], (x[2]+L-2)%L+1, x[3]] + ϕ[x[1], x[2], (x[3]+L-2)%L+1]

	return 3Δϕ² - Δϕ * ∑nn + 0.5m² * Δϕ² + 0.25λ * (ϕt^4 - ϕold^4)
end

function step(m², ϕ, x1, x2, L)
	norm = cos(2π*rand())*sqrt(-2*log(rand()))
	q = Rate*norm

	δH = ΔH(x1, ϕ, q, m², L) + ΔH(x2, ϕ, -q, m², L) + q^2
	P = min(1.0f0, exp(-δH))
	r = rand()
	
	@inbounds ϕ[x1...] += q * (r<P)
	@inbounds ϕ[x2...] -= q * (r<P)
end

function sweep(m², ϕ, threads, blocks)
	#=
	n=0 : (i,j,k)->(x,y,z)
	n=1 : (i,j,k)->(y,z,x)
	n=2 : (i,j,k)->(z,x,y)
	pairs are in i direction
	=#
	for m in 1:4
		kernel_i(m², ϕ, L, m; threads, blocks)
		kernel_j(m², ϕ, L, m; threads, blocks)
		kernel_k(m², ϕ, L, m; threads, blocks)
	end
end

function gpu_sweep_i(m², ϕ, L, m)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x

    for l in index:stride:L^3÷4-1
        i = l ÷ L^2
        j = (l÷L) % L
        k = l%L
        
        x1 = ((4i + 2j + m%2)%L+1, (j + k + m÷2)%L+1, k%L+1)
        @inbounds x2 = (x1[1]%L+1, x1[2], x1[3])

		step(m², ϕ, x1, x2, L)
    end
    return
end

function gpu_sweep_j(m², ϕ, L, m)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x

    for l in index:stride:L^3÷4-1
        i = l ÷ L^2
        j = (l÷L) % L
        k = l%L
        
        x1 = (k%L+1, (4i + 2j + m%2)%L+1, (j + k + m÷2)%L+1)
        @inbounds x2 = (x1[1], x1[2]%L+1, x1[3])

		step(m², ϕ, x1, x2, L)
    end
    return
end

function gpu_sweep_k(m², ϕ, L, m)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x

    for l in index:stride:L^3÷4-1
        i = l ÷ L^2
        j = (l÷L) % L
        k = l%L

        x1 = ((j + k + m÷2)%L+1, k%L+1, (4i + 2j + m%2)%L+1)
        @inbounds x2 = (x1[1], x1[2], x1[3]%L+1)

		step(m², ϕ, x1, x2, L)
    end
    return
end

function thermalize(m², ϕ, threads, blocks, N=10000)
	for i in 1:N
		sweep(m², ϕ, threads, blocks)
	end
end

function m²(t)
    m_a*t + m_b
end

ϕ = hotstart(L)
ϕ .= ϕ .- shuffle(ϕ)
ϕ = CuArray(ϕ)

N = L^3÷4

kernel_i = @cuda launch=false gpu_sweep_i(m²(0), ϕ, L, 1)
kernel_j = @cuda launch=false gpu_sweep_j(m²(0), ϕ, L, 1)
kernel_k = @cuda launch=false gpu_sweep_k(m²(0), ϕ, L, 1)
config = launch_configuration(kernel_i.fun)
threads = min(N, config.threads)
blocks = cld(N, threads)

maxt = trunc(Int, t_e / Δt)+1
skip = 1

for series in 1:16
	df = load("/share/tmschaef/jkott/modelB/KZ/IC_sym_L_$L"*"_id_"*ARGS[1]*"_series_$series.jld2")

	for run in 1:16
		ϕ = CuArray(df["ϕ"])

		thermalize(m²(0), ϕ, threads, blocks, 25*L^2)

		open("/share/tmschaef/jkott/modelB/dynamics_L_$L"*"_id_"*ARGS[1]*"_series_$series"*"_run_$run.dat","w") do io 
			for i in 0:maxt
				ϕk = fft(ϕ)

				Printf.@printf(io, "%i", skip*i)
				for kx in 1:L÷2+1
					Printf.@printf(io, " %f %f", real(ϕk[kx,1,1]), imag(ϕk[kx,1,1]))
				end 

				Printf.@printf(io,  "\n")
				flush(io)
				thermalize(m²(i * Δt), ϕ, threads, blocks, skip)
			end
		end
	end
end
