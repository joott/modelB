cd("/share/tmschaef/jkott/modelB/KZ/cumulants")

using Plots
using DelimitedFiles

const L = 32
const z = 3.906e0
const Δt = 0.04e0

# KZ protocol variables
const m²c, m²0, m²e = -2.28587, -2.0e0, -3.0e0
const t_c = 0.5 * 10^-3 * L^z * (1 - m²0/m²c)
const τ_Q = t_c * m²c / (m²c - m²0)
const m_a, m_b = m²c/τ_Q, m²0

const t_e = (m²e - m_b)/m_a
##

const t_c_steps = trunc(Int, t_c/Δt)

skip = 10
maxt = (trunc(Int, t_e / Δt)+1)÷skip+1

id_max = 16
series_max = 16
run_max = 32
n_files = id_max * series_max * run_max

M4 = zeros(Float32, maxt, n_files)
M2 = zeros(Float32, maxt, n_files)

function cumulant_run(fn, i)
    df = readdlm(fn, ' ')
    M = df[:,3]

    M4[:,i] .+= M.^4 / (n_files)
    M2[:,i] .+= M.^2 / (n_files)
end

function collect_files()
    Threads.@threads for id in 1:id_max
        for series in 1:series_max, run in 1:run_max
            cumulant_run("sum_L_$L"*"_id_$id"*"_series_$series"*"_run_$run.dat", run + (series-1)*series_max + (id-1)*series_max*id_max)
        end
    end
end

collect_files()

M4_err = std(M4, dims=2) / sqrt(n_files)
M2_err = std(M2, dims=2) / sqrt(n_files)

M4 .= sum(M4, dims=2)/n_files
M2 .= sum(M2, dims=2)/n_files

c4 = M4 .- 3 * M2.^2
c4_err = sqrt.(M4_err.^2 .+ (6 * M2 .* M2_err).^2)

output_file = open("data_cumulant_$L.jl","w")

write(output_file, "c4 = ")
show(output_file, C_tot)
write(output_file, "\n\n")

write(output_file, "c4_err = ")
show(output_file, c4_err)

close(output_file)