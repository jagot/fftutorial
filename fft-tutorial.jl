### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# ╔═╡ cf7caf47-9729-48b9-98fe-dc0ad5c75488
begin
    using Pkg
    Pkg.activate(".")

    using PyPlot
    using Jagot.plotting

    using LinearAlgebra
    using FFTW
end

# ╔═╡ 67a7ac20-98c1-11ed-26c3-bbea299eec72
md"""# FFTutorial

Drs. Chen & Stefanos
"""

# ╔═╡ 0489010d-31c6-4d41-8a6c-0d6581183aa9
md"""
- [X] Inner product
- [X] Complete set of orthogonal functions
- [ ] Schwarz class
- [X] Conjugate variables
- [ ] DFT, DST, DCT
- [-] Parseval’s theorem
- [ ] Apodizing window functions
- [ ] Background removal
- [-] Fourier differentiation ([notes by Steven G. Johnson](https://math.mit.edu/~stevenj/fft-deriv.pdf))
- [ ] Frequency axis
- [ ] Nyquist/Shannon, folding
    - [ ] Higher sampling frequency => higher max frequency
    - [ ] More samples => denser frequency grid
    - [ ] Zero padding does not give more info, but easier on the eyes
- [ ] Zeta transform
- [ ] Phase
"""

# ╔═╡ 23b0b46e-41c3-487e-a852-c2056cbb3afd
md""" # Theory

## Fourier transform

The _Fourier transform_ transforms between _conjugate variables_,
examples of which are

- time and frequency,
- position and momentum,
- rotation angle and angular momentum,
- etc.

In one dimension, it is given by
```math
\hat{f}(\omega) =
\frac{1}{\sqrt{2\pi}}
\int_{-\infty}^{\infty}
\mathrm{d}t
\mathrm{e}^{-\mathrm{i}\omega t}
f(t),
\tag{FFT}
```
and the inverse transform by
```math
f(t) =
\frac{1}{\sqrt{2\pi}}
\int_{-\infty}^{\infty}
\mathrm{d}\omega
\mathrm{e}^{\mathrm{i}\omega t}
\hat{f}(\omega).
\tag{IFFT}
```
A mathematician would say that ``f(t)`` and ``\hat{f}(\omega)`` are
different functions (hence the hat ``\hat{f}``), whereas a physicist
would say they are just two different representations of the same
element of a vector space. To see this, we call the vector
``|f\rangle``. Its time representation is ``\langle t|f\rangle``, and
hence we find its frequency representation as
```math
\langle \omega|f\rangle =
\int\mathrm{d}t
\langle \omega|t\rangle
\langle t|f\rangle,
```
where we have inserted a complete set
```math
\int\mathrm{d}t
|t\rangle\langle t| =
\hat{\mathbb{1}}.
```
From this, we surmise that
```math
\langle \omega|t\rangle \equiv
\frac{\mathrm{e}^{-\mathrm{i}\omega t}}{\sqrt{2\pi}},
```
which turns out to be a complete set of orthogonal functions on the
interval ``[-\infty,\infty]``:
```math
\langle \omega | \omega'\rangle =
\int\mathrm{d}t
\langle \omega | t\rangle
\langle t | \omega'\rangle =
\frac{1}{2\pi}
\int\mathrm{d}t
\mathrm{e}^{-\mathrm{i}(\omega-\omega') t} =
\delta(\omega-\omega').
```

This notation makes it obvious that we compute the _inner product_ of
``|\omega\rangle`` and ``|f\rangle``, by computing the projection of
``\langle t|f\rangle`` onto the basis functions ``\rangle
t|\omega\rangle`` (remember that in linear algebra, when we calculate
the projection of a vector ``\mathbf{b}`` on a vector ``\mathbf{a}``,
we have to take the conjugate transpose of ``\mathbf{a}``:
``\mathbf{a}^H\mathbf{b}``).

## Parseval's theorem

_Parseval's theorem_ states that the time integral of ``f(t)`` and the
frequency integral of ``\hat{f}(\omega)`` have to agree:
```math
\int_{-\infty}^\infty
\mathrm{d}t
|f(t)|^2 =
\frac{1}{2\pi}
\int_{-\infty}^\infty
\mathrm{d}\omega
|\hat{f}(\omega)|^2
```
This implies that the Fourier transform is _unitary_, i.e. it does not
change the norm. If we think in term of quantum mechanics, this is a
trivial statement: the norm of the wavefunction has to be the same in
e.g. the position and momentum representations:
```math
\begin{aligned}
|f|^2 =
\langle f|f\rangle
&=
\int\mathrm{d}t
\langle f|t\rangle
\langle t|f\rangle =
\int\mathrm{d}t
|f(t)|^2\\
&=
\int\mathrm{d}\omega
\langle f|\omega\rangle
\langle \omega|f\rangle =
\int\mathrm{d}\omega
|f(\omega)|^2\\
\end{aligned}
```
TODO: What happened to the ``(2\pi)^{-1}``? Here we have assumed that
```math
\int\mathrm{d}\omega
|\omega\rangle\langle \omega| =
\hat{\mathbb{1}},
```
which may have the wrong normalization.

Another way to view the Parseval's theorem is to think of it as the
Hilbert-space analogue of Pythagoras' theorem:
``|\mathbf{a}+\mathbf{b}+...|^2=|a|^2+|b|^2+...``, where the Euclidean
vectors ``\mathbf{a},\mathbf{b},...`` are all mutually orthogonal.

This will become important later, when we work with FFT.


## Differentiation

We can very easily figure out what the Fourier transform of a time
derivative of a function is:
```math
\partial_t^n
f(t) =
\partial_t^n
\langle t | f \rangle =
\int\mathrm{d}\omega
\partial_t^n
\langle t | \omega \rangle
\langle \omega | f \rangle =
(\mathrm{i}\omega)^n
\int\mathrm{d}\omega
\langle t | \omega \rangle
\langle \omega | f \rangle =
(\mathrm{i}\omega)^n
f(t)
```
```math
\implies
\langle \omega|
\partial_t^n|
f\rangle =
(\mathrm{i}\omega)^n
\langle \omega|f\rangle
```
(slight abuse of notation). One can also show that the Fourier
transform of a time integral (anti-derivative) of a function is
equivalently given by
```math
\implies
\langle \omega|
\int\mathrm{d}t^n|
f\rangle =
\frac{1}{(\mathrm{i}\omega)^n}
\langle \omega|f\rangle
```
(severe abuse of notation).

A good way to think about this is that the Fourier transform and
differentiation/integration are _linear operations_, which in
particular means that they commute and are distributive, and we can
perform them in any order, i.e. Fourier transform the
differentiated/integrated function, or differentiate/integrate the
Fourier modes after transforming; the latter corresponds to
multiplication/division by ``(\mathrm{i}\omega)^n``.

## Special functions

### Schwarz class


## Discrete Fourier transform

The _Discrete Fourier Transform_ is defined as:
```math
\tag{DFT}
\begin{aligned}
X_{k}&=\sum_{n=0}^{N-1}
x_{n}\cdot
\mathrm{e}^{-{\frac {\mathrm{i}2\pi }{N}}kn}\\
&=\sum_{n=0}^{N-1}
x_{n}\cdot
\left[\cos \left({\frac {2\pi }{N}}kn\right)-
\mathrm{i}\cdot \sin \left({\frac {2\pi }{N}}kn\right)
\right].
\end{aligned}
```
If we only compute the real part, we get the _Discrete Cosine Transform_:
```math
\tag{DCT}
X_{k}=
\sum_{n=0}^{N-1}
x_{n}\cdot
\cos \left({\frac {2\pi }{N}}kn\right),
```
and similarly, if we only compute the imaginary part, we get the _Discrete Sine Transform_:
```math
\tag{DST}
X_{k}=
\sum_{n=0}^{N-1}
x_{n}\cdot
\sin \left({\frac {2\pi }{N}}kn\right).
```
The DCT and DST are useful in signal processing and the solution of partial differential equations.

- Basis function ``\mathrm{e}^{-\mathrm{i}\frac{2\pi k}{N}n}`` instead
  of ``\mathrm{e}^{-\mathrm{i}\omega t}``.
  - Continuous time variable ``t`` replaced by sample index ``n``,
  - Continuous frequency variable ``\omega`` replaced by ``\frac{2\pi k}{N}``.


- Exact reconstructibility
- No Dirac ``\delta(\omega-\omega')``, instead Kronecker
  ``\delta_{kk'}``.

### Fast Fourier Transform

The _Fast Fourier Transform_ (FFT) is a DFT that is optimized for the
case when the number of samples ``N`` is a power-of-two,
i.e. ``N=2^q``, where ``q`` is an integer. In this case, we can
compute the transform using a divide-and-conquer approach where we
compute _all_ ``N`` Fourier modes ``X_k`` simultanously from two
smaller problems of size ``2^{q-1}``, which in turn are formed from
four even smaller problems of size ``2^{q-2}`` and so on, until we
have reached the primitive ``2\times2`` problem. This approach has
``\mathcal{O}(N\log N)`` complexity, instead of the
``\mathcal{O}(N^2)`` complexity of the brute-force evaluation of the
DFT definition.

_The_ standard implementation of FFT is [FFTW](https://www.fftw.org/)
(Fastest Fourier Transform in the West), written by Steven G. Johnson
_et al._

"""

# ╔═╡ d3cc841e-9f41-437c-a8a9-3e9826824b0a


# ╔═╡ c0dfab29-a96c-40bb-85cb-776ccd661f68
md""" # Sampling

"""

# ╔═╡ e76ee3d3-eef6-487b-82d8-a0b615115562
md""" # Apodizing window functions & background removal

"""

# ╔═╡ 0c029b61-c3ef-48aa-b3d2-46e4dc071c6d


# ╔═╡ Cell order:
# ╟─67a7ac20-98c1-11ed-26c3-bbea299eec72
# ╟─0489010d-31c6-4d41-8a6c-0d6581183aa9
# ╠═cf7caf47-9729-48b9-98fe-dc0ad5c75488
# ╟─23b0b46e-41c3-487e-a852-c2056cbb3afd
# ╠═d3cc841e-9f41-437c-a8a9-3e9826824b0a
# ╟─c0dfab29-a96c-40bb-85cb-776ccd661f68
# ╟─e76ee3d3-eef6-487b-82d8-a0b615115562
# ╠═0c029b61-c3ef-48aa-b3d2-46e4dc071c6d
