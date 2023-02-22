# FFT tutorial

## Quick instructions
1. Install [Julia](https://julialang.org/), e.g. via
   [`juliaup`](https://github.com/JuliaLang/juliaup).

2. Check out this repository.
   ```sh
   > git clone https://github.com/jagot/fftutorial.git
   ```

2. In the root directory of the repository, run
   ```sh
   > julia runme.jl
   ```

## Detailed instructions for Windows

1. Open the terminal, e.g. by pressing `Win+R` and type `cmd`
2. Using
   [`winget`](https://learn.microsoft.com/en-us/windows/package-manager/winget/),
   install the following tools:
   1. [Git](https://git-scm.com/download/win):
   ```
   winget install --id Git.Git -e --source winget
   ```
   2. [Julia](https://julialang.org/):
   ```
   winget install julia -s msstore
   ```
   3. (Optional) [Windows Terminal](https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701):
   ```
   winget install -e --id Microsoft.WindowsTerminal
   ```
3. Close the terminal (we need to reload the `PATH` environment
   variable)
4. Open a new terminal (either `cmd` or the newly installed Windows
   Terminal).
5. In a directory of your choosing, clone this repository:
   1. Browse to this directory:
   ```
   cd \the\directory\you\chose
   ```
   2. Check out this repository:
   ```
   git clone https://github.com/jagot/fftutorial.git
   ```
6. Change into the newly created directory and start the tutorial
   notebook:
   ```
   cd fftutorial
   julia runme.jl
   ```
7. Wait
8. Wait some more (the first time you run this command, some
   plotting-related packages will be downloaded and compiled, and this
   takes some time).
9. A notebook should appear in your web browser. Read the theory and
   marvel at the examples.
10. To close the notebook server, press `Ctrl+C` in the command window,
   and `Ctrl+D` to exit Julia.
