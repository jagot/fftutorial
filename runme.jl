using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Pluto
Pluto.run(notebook="./fft-tutorial.jl")
