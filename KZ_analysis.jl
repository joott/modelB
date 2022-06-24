cd("/share/tmschaef/jkott/modelB/KZ/dynamics")

using Plots
using DelimitedFiles

const L = 32
const z = 3.906e0
const Δt = 0.04e0

# KZ protocol variables
const m²c, m²0, m²e = -2.28587, -2.0e0, -4.0e0
const t_c = 0.5 * 10^-3 * L^z * (1 - m²0/m²c)
const τ_Q = t_c * m²c / (m²c - m²0)
const m_a, m_b = m²c/τ_Q, m²0

const t_e = (m²e - m_b)/m_a
##

const t_c_steps = trunc(Int, t_c/Δt)

maxt = trunc(Int, t_e / Δt)+2
ts = [1//2, 3//4, 1, 5//4, 5//2, maxt//t_c_steps]

C_tot = zeros(Float32, length(ts), L÷2)

function KZ_run(fn)
    df = readdlm(fn, ' ')
    if (length(df[:,1])) != maxt
        println("YIKES AT $fn")
        return
    end
    ϕ = zeros(ComplexF32, length(ts), L÷2)
    τ = trunc.(Int, ts*t_c_steps)

    Threads.@threads for k in 1:L÷2
        ϕ[:,k] .= df[τ,2k] .+ 1.0im .* df[τ,1+2k]
    end

    for i in 1:length(ts)
        C_tot[i,:] .+= abs2.(ϕ[i,:])/4096
    end
end

function collect_files()
    for id in 1:16, series in 1:16, run in 1:16
	@show (id, series, run)
        KZ_run("KZ_L_$L"*"_id_$id"*"_series_$series"*"_run_$run.dat")
    end
end

collect_files()

output_file = open("data_KZ_$L.jl","w")

write(output_file, "C_32 = ")
show(output_file, C_tot)
close(output_file)
