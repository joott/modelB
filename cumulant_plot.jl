cd(@__DIR__)

using Plots
using ColorSchemes
using Statistics
using LaTeXStrings
using DelimitedFiles
using LsqFit
using Interpolations
using Optim

L = 32
const λ = 4.0e0
const Γ = 1.0e0
const T = 1.0e0
const z = 3.906e0

const Δt = 0.04e0/Γ
const Rate = Float64(sqrt(2.0*Δt*Γ))

# KZ protocol variables
const m²c, m²0, m²e = -2.28587, -2.0e0, -3.0e0
m_a, m_b = begin
    τ_R = 2 * 10^-3 * L^z
    τ_Q = τ_R

    m²c/τ_Q, m²0
end

# In units of time
t_c = (m²c - m_b) / m_a
t_e = (m²e - m_b) / m_a
##

function time_axis(m, i)
    t = (m .- m_b) / (m_a / i)
    t_ce = t_c * i
    t = t / t_ce
    t
end

function scatter_style(xl,yl)
    scatter(
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
            markersize=1,
            legend=:topleft
        )
end

include("data/data_c2_32.jl")

p = palette(:Set1_5)

df = readdlm("data/sum_1.dat", ' ')
m_1 = df[:,2]
df = readdlm("data/sum_2.dat", ' ')
m_2 = df[:,2]
df = readdlm("data/sum_3.dat", ' ')
m_3 = df[:,2]

scatter_style(L"t/t_c", L"c_2(t)")
plot!(time_axis(m_1, 1), c2_1, fillalpha=0.1, ribbon=c2_1_err, label=L"\hat\tau_Q=1")
plot!(time_axis(m_2, 2), c2_2, fillalpha=0.1, ribbon=c2_2_err, label=L"\hat\tau_Q=2")
plot!(time_axis(m_3, 4), c2_3, fillalpha=0.1, ribbon=c2_3_err, label=L"\hat\tau_Q=4")

savefig("plots/c2_t.pdf")
