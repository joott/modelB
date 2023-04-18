cd("/share/tmschaef/jkott/modelB/KZ")

using Distributions
using Printf
using JLD2
using Random
using CUDA
using CUDA.CUFFT
using CodecZlib

Random.seed!(parse(Int, ARGS[3]))
CUDA.seed!(parse(Int, ARGS[3]))

const L = parse(Int, ARGS[2]) # must be a multiple of 4
const λ = 4.0e0
const Γ = 1.0e0
const T = 1.0e0
const z = 4.0e0

const Δt = 0.04e0/Γ
const Rate = Float64(sqrt(2.0*Δt*Γ))

m² = [-1.8, -1.9, -2.0, -2.1, -2.15, -2.2, -2.22, -2.24, -2.26]
ξ = [3, 3, 3, 4, 5, 7, 8, 11, 22]

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

function CorrFunc(ϕ, L)
	C = zeros(L÷2+1)

	for r in 0:(L÷2), i in 1:L, j in 1:L, k in 1:L
        C[r+1] = C[r+1] + ϕ[mod(i+r-1,L)+1,j,k]*ϕ[i,j,k] + ϕ[i,mod(j+r-1,L)+1,k]*ϕ[i,j,k] + ϕ[i,j,mod(k+r-1,L)+1]*ϕ[i,j,k]
	end
	C
end

function thermalize(m², ϕ, threads, blocks, N=10000)
	for i in 0:N-1
		sweep(m², ϕ, threads, blocks)
	end
end

ϕ = CUDA.zeros(Float64, L, L, L)

const N = L^3÷4
const m_id = parse(Int, ARGS[1])
const τ_C = trunc(Int, (4 * 10^-3 * ξ[m_id]^z) / Δt)
const n_corr = 100

kernel_i = @cuda launch=false gpu_sweep_i(m²[m_id], ϕ, L, 1)
kernel_j = @cuda launch=false gpu_sweep_j(m²[m_id], ϕ, L, 1)
kernel_k = @cuda launch=false gpu_sweep_k(m²[m_id], ϕ, L, 1)
config = launch_configuration(kernel_i.fun)
threads = min(N, config.threads)
blocks = cld(N, threads)

# df = load("IC_sym_L_$L"*"_id_1_series_$(ARGS[1]).jld2")
df = load("corr/phi_L_$(L)_m2_$(m_id).jld2") # reuse previous configuration
ϕ .= CuArray(df["ϕ"])

open("corr/corr_L_$(L)_m2_$(m_id).dat","a") do io 
    for i in 1:n_corr
        thermalize(m²[m_id], ϕ, threads, blocks, L^4)
        C = CorrFunc(Array(ϕ), L)

        for x in 1:L÷2+1
            Printf.@printf(io, "%f %f %f\n", x-1, C[x]/(3*L^3), m²[m_id])
        end
        jldsave("corr/phi_L_$(L)_m2_$(m_id).jld2", true; ϕ=Array(ϕ), m2=m²[m_id])
        flush(io)
    end
end
