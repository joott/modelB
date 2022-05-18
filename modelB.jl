using Plots
using LaTeXStrings

L = 8
v = [0 0 0; 1 0 0; 0 1 0; 1 1 0]
colors = [:black, :red, :green, :blue]

function update(ϕ, x1, x2, m)
    if x1[3]==L
        plot!([x1[3], L+1].+0.5, [x1[1], x2[1]].+0.5, lc=colors[m])
        plot!([x2[3], 0].+0.5, [x1[1], x2[1]].+0.5, lc=colors[m])
    else
        plot!([x1[3], x2[3]].+0.5, [x1[1], x2[1]].+0.5, lc=colors[m])
    end
end

function sweep(ϕ)
    plots = []
    T(i,j,k) = [4(i-1)+2(j-1), j+k-2, k-1]
    μ = [1 0 0; 0 1 0; 0 0 1]
    for n in 2:2
        for m in 1:4
            for k in 1:L   
                p = plot()
                for j in 1:L
                    for i in 1:L÷4
                        x0 = T(i,j,k)
                        idx = ((3-n)%3+1, (4-n)%3+1, (5-n)%3+1)
                        v0 = v[m,:]
                        x1 = [x0[idx[1]], x0[idx[2]], x0[idx[3]]] + [v0[idx[1]], v0[idx[2]], v0[idx[3]]]
                        x2 = x1 + μ[n+1,:]
                        update(ϕ, x1.%L .+1, x2.%L .+1, m)
                    end
                end
                push!(plots, p)
            end
        end
    end
    plots
end

ϕ = zeros(Float64, L, L, L)
result = sweep(ϕ)
anim = @animate for n in 1:length(result)
    plot(result[n], legend=false)
    title!(latexstring("y=", (n-1)%L+1))
    xaxis!("z", (1,L+1))
    yaxis!("x", (1,L+1))
end

gif(anim, "sweeps3.gif", fps=1.5)