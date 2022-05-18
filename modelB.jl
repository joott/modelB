using Plots
using LaTeXStrings

L = 8
v = [0 0 0; 1 0 0; 0 1 0; 1 1 0]
colors = [:black, :red, :green, :blue]

function update(ϕ, x1, x2, m)
    if x1[1]==8
        plot!([x1[1], 9].+0.5, [x1[2], x2[2]].+0.5, lc=colors[m])
        plot!([x2[1], 0].+0.5, [x1[2], x2[2]].+0.5, lc=colors[m])
    else
        plot!([x1[1], x2[1]].+0.5, [x1[2], x2[2]].+0.5, lc=colors[m])
    end
end

function sweep(ϕ)
    plots = []
    μ = [1,0,0]
    for m in 1:4
        for k in 1:L   
            p = plot()
            for j in 1:L
                for i in 1:L÷4
                    x1 = [4(i-1)+2(j-1), j+k-2, k-1] + v[m,:]
                    x2 = x1 + μ
                    update(ϕ, x1.%L .+1, x2.%L .+1, m)
                end
            end
            push!(plots, p)
        end
    end
    plots
end

ϕ = zeros(Float64, L, L, L)
result = sweep(ϕ)
anim = @animate for n in 1:length(result)
    plot(result[n], legend=false)
    title!(latexstring("z=", (n-1)%L+1))
    xaxis!("x", (1,9))
    yaxis!("y", (1,9))
end

gif(anim, "sweeps.gif", fps=1.5)