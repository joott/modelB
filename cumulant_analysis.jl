cd("/share/tmschaef/jkott/modelB/KZ/cumulants/"*ARGS[1])

using DelimitedFiles
using Statistics

const L = 32
const z = 3.906e0
const Δt = 0.04e0
const threads = Threads.nthreads()
const factor = 2^parse(Int, ARGS[1])

# KZ protocol variables
const m²c, m²0, m²e = -2.28587, -2.0e0, -3.0e0
m_a, m_b = begin
	τ_R = 2 * 10^-3 * L^z
    τ_Q = factor * τ_R

    m²c/τ_Q, m²0
end

# In units of time
t_c = (m²c - m_b) / m_a
t_e = (m²e - m_b) / m_a

# In units of steps
const skip = 100
const maxt = div(trunc(Int, t_e / Δt) + 1, skip) + 1
##

id_max = 16
series_max = 16
run_max = 192
n_files = id_max * series_max * run_max

M2 = zeros(Float64, maxt, threads)
M2_err = zeros(Float64, maxt, threads)

function cumulant_run(fn)
    df = readdlm(fn, ' ')
    M = df[:,3]

    M2[:,Threads.threadid()] .+= M.^2 / n_files
end

function cumulant_err(fn)
    df = readdlm(fn, ' ')
    M = df[:,3]

    M2_err[:,Threads.threadid()] .+= (M.^2 .- M2).^2 / n_files
end

function collect_files()
    Threads.@threads for id in 1:id_max
        for series in 1:series_max, run in 1:run_max
            cumulant_run("sum_L_$L"*"_id_$id"*"_series_$series"*"_run_$run.dat")
        end
    end

    global M2 = sum(M2, dims=2)

    Threads.@threads for id in 1:id_max
        for series in 1:series_max, run in 1:run_max
            cumulant_err("sum_L_$L"*"_id_$id"*"_series_$series"*"_run_$run.dat")
        end
    end

    global M2_err = sum(M2_err, dims=2)
end

collect_files()

M2_err .= sqrt.(M2_err / n_files)

output_file = open("data_cumulant_$L.jl","w")

write(output_file, "c2 = ")
show(output_file, M2)
write(output_file, "\n\n")

write(output_file, "c2_err = ")
show(output_file, M2_err)

close(output_file)
