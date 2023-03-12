cd(@__DIR__)

using Interpolations
using DelimitedFiles
using StaticArrays
using LinearAlgebra
using LaTeXStrings
using PlotThemes
using Plots
using Optim

include("data/data_c2_32.jl")

interp_linear1 = LinearInterpolation(t_1, c2_1)
err_linear1 = LinearInterpolation(t_1, c2_1_err)

interp_linear2 = LinearInterpolation(t_2, c2_2)
err_linear2 = LinearInterpolation(t_2, c2_2_err)

interp_linear3 = LinearInterpolation(t_3, c2_3)
err_linear3 = LinearInterpolation(t_3, c2_3_err)


xs = collect(-40:0.1:40)

function minimize(x)
    return sum(
        .+ abs.(interp_linear1.(xs) .- interp_linear2.(xs*2^x[1])/2^x[2])
        .+ abs.(interp_linear1.(xs) .- interp_linear3.(xs*4^x[1])/4^x[2])
    )
end

initial_x=[0.7,0.3]
res=optimize(minimize, initial_x)

Optim.summary(res)
param=Optim.minimizer(res)
println(param)

function scatter_style(xl,yl)
    scatter!(
        	ylabel=yl, xlabel=xl,
            grid = :off,
            box = :on,
            foreground_color_legend = nothing,
            fontfamily = "serif-roman",
            font="CMU Serif",
            xtickfontsize = 10,
            ytickfontsize = 10,
            xguidefontsize = 10,
            yguidefontsize = 10,
            thickness_scaling=1.5,
            legendfontsize=10,
            markersize=0.7,
            legend=:topleft,
        )
end

plotlims = (-50,100)
xplot_1 = collect(plotlims[1]:0.1:plotlims[2])
xplot_2 = collect(plotlims[1]*2^param[1]:0.1:plotlims[2]*2^param[1])
xplot_3 = collect(plotlims[1]*4^param[1]:0.1:plotlims[2]*4^param[1])
yscale(i) = interp_linear1(0) * 2^((i-1)*param[2])

plot(xplot_1, interp_linear1.(xplot_1)/yscale(1), fillalpha=0.1, ribbon=err_linear1.(xplot_1)/yscale(1), label=L"\hat \tau_Q=1")
plot!(xplot_2/2^param[1], interp_linear2.(xplot_2)/yscale(2), fillalpha=0.1, ribbon=err_linear2.(xplot_2)/yscale(2), label=L"\hat\tau_Q=2")
plot!(xplot_3/4^param[1], interp_linear3.(xplot_3)/yscale(3), fillalpha=0.1, ribbon=err_linear3.(xplot_3)/yscale(3), label=L"\hat\tau_Q=4")
scatter_style(L"\bar t/\hat\tau_Q^{p_1}", L"\hat c_2(\bar t,\tau_Q)/\hat \tau_Q^{p_2}")
savefig("plots/cumulant_fit.pdf")
