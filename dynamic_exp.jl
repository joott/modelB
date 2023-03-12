cd(@__DIR__)

using LsqFit
using Plots
using LaTeXStrings

include("data/data_8.jl")
include("data/data_12.jl")
include("data/data_16.jl")
include("data/data_24.jl")

L = [8, 12, 16, 24]
t = [t_8, t_12, t_16, t_24]
C = [C_8, C_12, C_16, C_24]
C_err = [Cerr_8, Cerr_12, Cerr_16, Cerr_24]
τ = zeros(length(L))
τ_err = zeros(length(L))

for i in 1:length(L)
    C_err[i] = C_err[i] / C[i][1]
    C[i] = C[i] / C[i][1]
end

m(t,p) = exp.(p[1]*t)
p0 = [-0.1]

for i in 1:length(L)
    fit = curve_fit(m, t[i], C[i], p0)
    τ[i] = -inv(fit.param[1])
    τ_err[i] = stderror(fit)[1]
end

fit_fn(t,p) = p[2] .+ p[1] * t

z_fit = curve_fit(fit_fn, log.(L), log.(τ), [4.0, 0.0])
(z, c) = z_fit.param
z_err = stderror(z_fit)[1]

c_err = exp(stderror(z_fit)[2])
println(c*(1-c_err))

scatter(log.(L), log.(τ), legend=:topleft, label="")
plot!(log.(L[[1,end]]), c.+z.*log.(L[[1,end]]),
    legend=nothing,
    foreground_color_legend=nothing,
    grid=:off,
    box=:on,
    fontfamily = "serif-roman",
    xtickfontsize = 10,
    ytickfontsize = 10,
    xguidefontsize = 10,
    yguidefontsize = 10,
    thickness_scaling=1.5,
    legendfontsize=5,
    markersize=1)
xlabel!(L"\ln\, L")
ylabel!(L"\ln\, \tau")
savefig("plots/z_fit_B.pdf")

scatter(xlabel=L"t/L^z",
    ylabel=L"G(t,|k|=2\pi/L)/\chi",
    foreground_color_legend=nothing,
    grid=:off,
    box=:on,
    fontfamily = "serif-roman",
    xtickfontsize = 10,
    ytickfontsize = 10,
    xguidefontsize = 10,
    yguidefontsize = 10,
    thickness_scaling=1.5,
    legendfontsize=10,
    xticks = [0.0, 0.0025, 0.005, 0.0075, 0.01],
    xlims = [0, 0.012]
)

for i in eachindex(L)
    plot!(t[i]/L[i]^z, C[i], ribbon = C_err[i], label=L"L=%$(L[i])", fillalpha = 0.2)
end

savefig("plots/Ck1.pdf")
