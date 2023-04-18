cd("/share/tmschaef/jkott/modelB/KZ/fft/"*ARGS[1])

using DelimitedFiles
using Statistics

const L = 32
const id_max = 16
const series_max = 16
const run_max = 64
const id_size = series_max*run_max
const n_quenches = id_max*id_size

C_tot = zeros(Float32, L÷2+1)
database = zeros(Float32, n_quenches, L÷2+1)

function KZ_run(fn, n)
    df = readdlm(fn, ' ')

    ϕ = zeros(ComplexF32, id_size, L÷2+1)

    Threads.@threads for k in 1:L÷2+1
        ϕ[:,k] = df[:,2k-1].^2 .+ df[:,2k].^2
    end

    database[id_size*(n-1)+1:id_size*n,:] .= ϕ
end

function collect_files()
    for id in 1:id_max
        KZ_run("fft_L_$L"*"_id_$id.dat", id)
    end
end

collect_files()

C_tot = sum(database, dims=1)[1,:]/n_quenches
Cerr = std(database, dims=1)/sqrt(n_quenches)

output_file = open("data_KZ_$L.jl","w")

write(output_file, "C_32 = ")
show(output_file, C_tot)
write(output_file, "\n\n")

write(output_file, "Cerr_32 = ")
show(output_file, Cerr[1,:])

close(output_file)
