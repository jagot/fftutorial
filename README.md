# FFT tutorial

Instructions:
1. Install [Julia](https://julialang.org/), e.g. via
   [`juliaup`](https://github.com/JuliaLang/juliaup).

2. Check out this repository.
   ```sh
   > git clone https://github.com/jagot/fftutorial.git
   ```

2. Run Julia in the root directory of the repository and enter the
   following commands:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()

   using Pluto
   Pluto.run(notebook="./fft-tutorial.jl")
   ```
