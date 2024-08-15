using CairoMakie, LinearAlgebra, GeometryBasics

θs = LinRange(0, π, 128)
ϕs = LinRange(π/2, 5π/2, 128)

sphere(θ,ϕ) = [sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ)]

sphere_points = stack(sphere(θ,ϕ) for θ in θs, ϕ in ϕs)

#rs = stack(sphere(θ,ϕ) for θ in θs, ϕ in ϕs)
z = 0.1
r = √(1-z^2)
rs = stack([cos(ϕ), sin(ϕ),z] for ϕ in ϕs)

function f(r, λ)
    denominator = 1+r[3] + λ^2*(1-r[3])
    x = 2*λ*r[1]/denominator
    y = 2*λ*r[2]/denominator
    z = (1+r[3] - λ^2*(1-r[3]))/ denominator
    [x, y, z]
end

function generate_disc_points(n)
    points = Matrix{Float64}(undef, 2, n)
    for i in 1:n
        r = sqrt(rand())  # Random radius
        θ = 2 * π * rand()  # Random angle
        points[1,i] = r * cos(θ)
        points[2,i] = r * sin(θ)
    end
    points
end
##
grid = LinRange(-1,1,5)
evenly_spaced_points = stack(x for x ∈ Iterators.product(grid,grid) if norm(x) <= 1)
evenly_spaced_points
scatter(evenly_spaced_points)

z = -.5
R = √(1-z^2)
rs = stack((R*r[1],R*r[2],z) for r ∈ eachslice(evenly_spaced_points, dims=2))
##
λs = LinRange(0, 1, 512)
trs = stack(mapslices(r->f(r, λ), rs, dims=1) for λ in λs)

"""fig = Figure(size=(600,600))
ax = Axis(fig[1, 1], aspect = DataAspect())
xlims!(ax, (-1.1, 1.1))
ylims!(ax, (-1.1, 1.1))
scatter!(ax, trs[1,:], trs[2,:])
fig"""

trs

fig = Figure()
ax = Axis3(fig[1, 1], aspect = (1, 1, 1))
xlims!(ax, (-1, 1))
ylims!(ax, (-1, 1))
zlims!(ax, (-1, 1))
for X ∈ eachslice(trs, dims=2)
    lines!(ax, X[1,:], X[2,:], X[3,:])
end
mesh!(ax, Sphere(Point3f(0), 1), alpha=0.1)
fig