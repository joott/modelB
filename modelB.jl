cd(@__DIR__)

using Plots
using Distributions
using Printf
using BenchmarkTools
using StaticArrays
using FFTW

const L = 8 # must be a multiple of 4
const λ = 4.0f0
const Γ = 1.0f0
const T = 1.0f0

const Δt = 0.04f0/Γ
const Rate = Float32(sqrt(2.0*Δt*Γ))
ξ = Normal(0.0f0, 1.0f0)


function hotstart(n)
    rand(ξ, n, n, n)
end

function ΔH(x, ϕ, q, m²)
    @inbounds ϕold = ϕ[x[1], x[2], x[3]]
    ϕt = ϕold + q
    Δϕ = ϕt - ϕold
    Δϕ² = ϕt^2 - ϕold^2

    @inbounds ∑nn = ϕ[x[1]%L+1, x[2], x[3]] + ϕ[x[1], x[2]%L+1, x[3]] + ϕ[x[1], x[2], x[3]%L+1]
    @inbounds ∑nn += ϕ[(x[1]+L-2)%L+1, x[2], x[3]] + ϕ[x[1], (x[2]+L-2)%L+1, x[3]] + ϕ[x[1], x[2], (x[3]+L-2)%L+1]

    3Δϕ² - Δϕ * ∑nn + 0.5m² * Δϕ² + 0.25λ * (ϕt^4 - ϕold^4)
end

function step(m², ϕ, x1, x2)
    q = Rate*rand(ξ)

    @inbounds ϕ1 = ϕ[x1[1], x1[2], x1[3]]
    @inbounds ϕ2 = ϕ[x2[1], x2[2], x2[3]]

    δH = ΔH(x1, ϕ, q, m²) + ΔH(x2, ϕ, -q, m²) + q^2
    P = min(1.0f0, exp(-δH))
    r = rand(Float32)
	
	if (r < P)
        @inbounds ϕ[x1[1], x1[2], x1[3]] += q
        @inbounds ϕ[x2[1], x2[2], x2[3]] -= q
    end
end

function sweep(m², ϕ)
    for n in 0:2, m in 1:4
        Threads.@threads for k in 1:L   
            for i in 1:L÷4, j in 1:L
                idx = [(3-n)%3+1, (4-n)%3+1, (5-n)%3+1]
                transition = [4(i-1)+2(j-1), j+k-2, k-1]

                x1 = transition[idx]
                x1[idx[1]] += m%2
                x1[idx[2]] += m<3
				x2 = copy(x1)
                x2[n+1] += 1

                step(m², ϕ, x1.%L.+1, x2.%L.+1)
            end
        end
    end
end

function thermalize(m², ϕ, N=10000)
    for i in 1:N
        sweep(m², ϕ)
    end
end

M(ϕ) = sum(ϕ)/L^3

m² = -2.285

ϕ = hotstart(L)

thermalize(m², ϕ, 100*L^4)

maxt = L^4*25

skip=20 

ϕk = fft(ϕ)

open("output_$L.dat","w") do io 
	for i in 0:maxt
		Mt = M(ϕ)

		ϕk = fft(ϕ)

		Printf.@printf(io, "%i %f", skip*i, Mt)
		for kx in 1:L
			Printf.@printf(io, " %f %f", real(ϕk[1,kx,1]), imag(ϕk[1,kx,1]))
		end 

		Printf.@printf(io,  "\n")
		thermalize(m², ϕ, skip)
	end
end
