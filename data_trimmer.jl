cd("/share/tmschaef/jkott/modelB/KZ/cumulants")

using DelimitedFiles
using Printf

id = parse(Int, ARGS[1])
max_run = 832

Threads.@threads for series in 1:16
    for run in 1:max_run
    	df = readdlm("sum_L_32_id_$(id)_series_$(series)_run_$run.dat", ' ')

    	open("trim/sum_L_32_id_$(id)_series_$(series)_run_$run.dat","w") do io
    	    for row in 1:10:size(df, 1)
    	        Printf.@printf(io, "%i %f %f\n", df[row,1], df[row,2], df[row,3])
    	    end
    	end
    end
end
