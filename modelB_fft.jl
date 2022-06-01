cd(@__DIR__)

using Distributions
using Printf
using FFTW
using JLD2
using Random
using LinearAlgebra

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

function ΔH(x, ϕ, q, m²)
	@inbounds ϕold = ϕ[x...]
	ϕt = ϕold + q
	Δϕ = ϕt - ϕold
	Δϕ² = ϕt^2 - ϕold^2

	@inbounds ∑nn = ϕ[x[1]%L+1, x[2], x[3]] + ϕ[x[1], x[2]%L+1, x[3]] + ϕ[x[1], x[2], x[3]%L+1]
	@inbounds ∑nn += ϕ[(x[1]+L-2)%L+1, x[2], x[3]] + ϕ[x[1], (x[2]+L-2)%L+1, x[3]] + ϕ[x[1], x[2], (x[3]+L-2)%L+1]

	3Δϕ² - Δϕ * ∑nn + 0.5m² * Δϕ² + 0.25λ * (ϕt^4 - ϕold^4)
end

function step(m², ϕ, x1, x2)
	q = Rate*rand(ξ)

	δH = ΔH(x1, ϕ, q, m²) + ΔH(x2, ϕ, -q, m²) + q^2
	P = min(1.0f0, exp(-δH))
	r = rand(Float64)
	
	if (r < P)
		@inbounds ϕ[x1...] += q
		@inbounds ϕ[x2...] -= q
	end
end

function sweep(m², ϕ)
    for n in 0:7
        Threads.@threads for m in 0:L^3÷16-1
            # Truth table-esque generation of coordinates on an L x L/4 x L/4 lattice
            # Printing the output should make it clear what this part is doing
            i = 4m ÷ (L^2)
            j = (m÷L) % (L÷4)
            k = m%L
            ##

            # Convert (i,j,k) indices to LxLxL lattice coordinates with necessary offsets from n
            x1 = [4i+k+2(n÷4),4j+2k+(n%4),k]

            # Update in all six directions
            for μ in 0:5
                x2 = copy(x1)
                # μ÷3 and μ%3 create another truth table of 3 directions for +/-
                # (L-2)*(μ÷3) term is effectively turning +1 into -1 due to the later modulus operation
                x2[μ%3+1] += 1+(L-2)*(μ÷3)

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

A = collect([i,j,k] for i in 0:L-1, j in 0:L-1, k in 0:L-1)
C = []

for n in 1:3(L-1)^2
    B = A[A.⋅A .== n]
    length(B) != 0 && push!(C, B[1].+1)
end

indices = map(v -> CartesianIndex(v...), C)

df = load("/share/tmschaef/jkott/modelB/IC_L_$L"*"_id_"*ARGS[1]*".jld2")

ϕ = df["ϕ"]
m² = df["m2"]

thermalize(m², ϕ, L^4÷4)

skip=10 
maxt = 50*L^4

open("/share/tmschaef/jkott/modelB/dynamics_k_L_$L"*"_id_"*ARGS[1]*".dat","w") do io 
	for i in 0:maxt
		ϕk = fft(ϕ)

		Printf.@printf(io, "%i", skip*i)
		for k in indices
			Printf.@printf(io, " %f %f", real(ϕk[k...]), imag(ϕk[k...]))
		end 

		Printf.@printf(io,  "\n")
		flush(io)
		thermalize(m², ϕ, skip)
	end
end
