using Pkg
Pkg.activate(".")

do_instantiate = true

for a in ARGS
    if a ∈ ["-ni", "--no-instantiate"]
        global do_instantiate = false
    end
end

if do_instantiate
    @info "Instantiating environment"
    Pkg.instantiate()
end

using Pluto
Pluto.run(notebook="./fft-tutorial.jl")
