cd(@__DIR__)

using Distributions
using Printf
using FFTW
using JLD2
using Random
using LinearAlgebra
using CUDA
using BenchmarkTools

Random.seed!(parse(Int, ARGS[3]))

const L = parse(Int, ARGS[2]) # must be a multiple of 4
const λ = 4.0e0
const Γ = 1.0e0
const T = 1.0e0

const Δt = 0.04e0/Γ
const Rate = Float64(sqrt(2.0*Δt*Γ))
ξ = Normal(0.0e0, 1.0e0)

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

function step(m², ϕ, x1, x2, L, R, r_i)
	q = Rate*R[r_i]

	δH = ΔH(x1, ϕ, q, m², L) + ΔH(x2, ϕ, -q, m², L) + q^2
	P = min(1.0f0, exp(-δH))
	r = rand()
	
	@inbounds ϕ[x1...] += q * (r<P)
	@inbounds ϕ[x2...] -= q * (r<P)
end

function sweep(m², ϕ, R, kernel, threads, blocks)
	#=
	n=0 : (i,j,k)->(x,y,z)
	n=1 : (i,j,k)->(y,z,x)
	n=2 : (i,j,k)->(z,x,y)
	pairs are in i direction
	=#
	for n in 0:2, m in 1:4
		kernel(m², ϕ, L, n, m, R; threads, blocks)
	end
end

function gpu_sweep(m², ϕ, L, n, m, R)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x

    for l in index:stride:L^3÷4-1
        i = l ÷ L^2
        j = (l÷L) % L
        k = l%L
        
        tmp = (4i + 2j + m%2, j + k + m÷2, k)
		(x, y, z) = (tmp[n+1], tmp[(n+1)%3+1], tmp[(n+2)%3+1])
		x1 = (x%L+1, y%L+1, z%L+1)
        x2 = ((x + (n==0))%L+1, (y + (n==1))%L+1, (z + (n==2))%L+1)

		r_i = L^3÷4 * (n + 3(m-1)) + l + 1
		step(m², ϕ, x1, x2, L, R, r_i)
    end
    return
end

function thermalize(m², ϕ, kernel, threads, blocks, N=10000)
	R = CUDA.randn(N*3L^3)
	for i in 1:N
		sweep(m², ϕ, R[range((i-1)*3L^3+1, length=3L^3)], kernel, threads, blocks)
	end
end

m² = -2.28587

ϕ = hotstart(L)
ϕ .= ϕ .- shuffle(ϕ)
ϕ = CuArray(ϕ)

N = 1024

R = CUDA.randn(3L^3)

kernel = @cuda launch=false gpu_sweep(m², ϕ, L, 0, 1, R)
config = launch_configuration(kernel.fun)
threads = min(N, config.threads)
blocks = cld(N, threads)

maxt = L^2

for i in 0:maxt
	thermalize(m², ϕ, kernel, threads, blocks, 4*L^2)
	jldsave("/share/tmschaef/jkott/modelB/KZ/IC_crit_L_$L"*"_id_"*ARGS[1]*".jld2", true; ϕ=ϕ, m2=m², i=i)
end