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

### Side note on accuracy of differential operator approximations

Using the Fourier transform to evaluate the derivative of a function
sampled on an equidistant grid is provably the best we can achieve,
_provided the function fulfills periodic boundary conditions_. Many
time propagators for the Schrödinger equation or Maxwell's equation
work by transforming to ``k``-space using the FFT, where the Laplacian
is diagonal and amounts to multiplication by ``-k^2``, and then
transforming back to real space, where the potential terms are
diagonal (local). Such methods are called _spectral_, and are
associated with highest possible accuracy and being very efficient,
but _only for periodic functions_. For other boundary conditions,
there are also fast transforms available, but no as successful as the
FFT. For these (very important) cases, we are usually better off
staying in real space and approximating our differential operators
using finite-difference matrices of limited bandwidth or
finite-elements (the latter typically have spectral accuracy of the
differential operators within each element).

## Special functions

### Schwarz class

```math
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left[
-\frac{(x-\mu)^2}{2\sigma^2}
\right]
```


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

We define our test function, which is a normalized Gaussian multiplying two frequencies:

```math
f(t,\sigma) =
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(
-\frac{t^2}{2\sigma^2}
\right)
(1.2\sin2\pi t
+\sin3\pi t)
```
"""

# ╔═╡ d8e9c2eb-8e7f-427c-b785-f7b504145dc7
f(t, σ) = 1/(√(2π)*σ) * exp(-t^2/(2σ^2)) * (1.2sin(2π*t) + sin(4.5π*t))

# ╔═╡ a5e79680-8d5a-4d36-b08d-3d32eaf3b751
md"Here is our standard time grid:"

# ╔═╡ ab40d1ac-aff4-4161-907d-aa5688857977
t = range(-15, stop=15, length=150)

# ╔═╡ b9404c8f-0a45-42c3-addc-825981d10814
md"And here is the corresponding frequency grid:"

# ╔═╡ 28e307a3-d169-4371-8a2b-b64ce9d5d6cb
ω = 2π*fftshift(fftfreq(length(t), 1/step(t)))

# ╔═╡ 2374f003-1b73-4d04-8196-713983aa497e
y = f.(t, 2)

# ╔═╡ 7c0a4378-42ed-4a43-9d35-dc075f99200a
Y = fftshift(fft(y))/length(t)

# ╔═╡ abfaf313-e881-48f5-b4b9-eed62505a5d0
md"## Oversampling"

# ╔═╡ 6bcf1e9a-bb40-4735-81c0-2420fc9040c0
md"We first investigate what happens if we change the sampling frequency:"

# ╔═╡ a30a41e5-2608-404f-bc5f-95cb7e7c2dea
tfine = range(-15, stop=15, length=300)

# ╔═╡ 3ce118c0-aa98-4920-91c4-cec860ae14a2
ωfine = 2π*fftshift(fftfreq(length(tfine), 1/step(tfine)))

# ╔═╡ dfb7032e-23e1-49a3-8d3a-fb6b890e20f0
yfine = f.(tfine, 2)

# ╔═╡ f94e1296-ecce-4789-943c-1bf87ac11349
Yfine = fftshift(fft(yfine))/length(tfine)

# ╔═╡ 6f2ffbe2-290e-43e4-9c1c-b3c5183bfa8f
cfigure("function fine sampling") do
    plot(t, y)
    plot(tfine, yfine, "--")
    xlabel(L"t")
    ylabel(L"y(t)")
end

# ╔═╡ 64bbd860-42c9-4dda-a162-e63120ae83c9
cfigure("Fourier transform fine sampling") do
    csubplot(211,nox=true) do
        plot(ω, abs2.(Y))
        plot(ωfine, abs2.(Yfine), "--")
    end
    csubplot(212) do
        semilogy(ω, abs2.(Y))
        semilogy(ωfine, abs2.(Yfine), "--")
        xlabel(L"$\omega$ [rad]")
    end
end

# ╔═╡ f7bf8468-5f1f-4857-86c9-1ab9581a36a4
md"""We see that the spectra agree, but the finer grid allows us to
resolve higher frequencies."""

# ╔═╡ 818e4a62-1882-4e03-9496-407d799dfae6
md"## Undersampling (folding)"

# ╔═╡ bf8c2287-4dab-4883-bb28-f2355e1d6bbf
tcoarse = range(-15, stop=15, length=75)

# ╔═╡ 42ba83b0-8162-4516-bf35-288a54bf4d7c
ωcoarse = 2π*fftshift(fftfreq(length(tcoarse), 1/step(tcoarse)))

# ╔═╡ 6bcf7aa6-d3f8-48eb-83d4-8dca7f50cf03
ycoarse = f.(tcoarse, 2)

# ╔═╡ 9bf8e6e9-efd4-4f2c-9953-bda19d7b8494
Ycoarse = fftshift(fft(ycoarse))/length(tcoarse)

# ╔═╡ 8eab9d56-8c6c-4fea-ad7c-560f3c9a364c
cfigure("function coarse sampling") do
    plot(t, y)
    plot(tcoarse, ycoarse, "--")
    xlabel(L"t")
    ylabel(L"y(t)")
end

# ╔═╡ 4da5b68f-498c-4075-916e-4ddd8e2a48c0
cfigure("Fourier transform coarse sampling") do
    csubplot(211,nox=true) do
        plot(ω, abs2.(Y))
        plot(ωcoarse, abs2.(Ycoarse), "--")
    end
    csubplot(212) do
        semilogy(ω, abs2.(Y))
        semilogy(ωcoarse, abs2.(Ycoarse), "--")
        xlabel(L"$\omega$ [rad]")
    end
end

# ╔═╡ 5a61a358-18d9-4b1f-b8c9-f3fc9a115f6e
md"""
When we do not have enough samples (we do not fulfill the
Shannon–Nyquist sampling theorem), the higher frequency peak appears
at a lower frequency (it "folds" down).
"""

# ╔═╡ 998b21b2-8802-4937-8983-4e7980dd142d
md"## Longer pulse"

# ╔═╡ f1064a13-eef8-4bf0-b747-135f0dc8f635
y2 = f.(t, 5)

# ╔═╡ 105ee14a-40c6-41ef-b908-a2184c3a6109
cfigure("function") do
    plot(t, y)
    plot(t, y2, "--")
    xlabel(L"t")
    ylabel(L"y(t)")
end

# ╔═╡ 2ccd66c9-166a-486b-b546-205ccb84135d
Y2 = fftshift(fft(y2))/length(t)

# ╔═╡ 200db744-1c52-404d-ab28-b7b4ff470d2c
cfigure("Fourier transform") do
    plot(ω, abs2.(Y))
    plot(ω, abs2.(Y2), "--")
end

# ╔═╡ de98ab8f-f2d4-4877-94e1-5a87dbee8485
md"""The dashed spectrum is sharper, since the pulse is longer, and we
have more time measure the actual frequency content."""

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
# ╠═d8e9c2eb-8e7f-427c-b785-f7b504145dc7
# ╟─a5e79680-8d5a-4d36-b08d-3d32eaf3b751
# ╠═ab40d1ac-aff4-4161-907d-aa5688857977
# ╟─b9404c8f-0a45-42c3-addc-825981d10814
# ╠═28e307a3-d169-4371-8a2b-b64ce9d5d6cb
# ╠═2374f003-1b73-4d04-8196-713983aa497e
# ╠═7c0a4378-42ed-4a43-9d35-dc075f99200a
# ╟─abfaf313-e881-48f5-b4b9-eed62505a5d0
# ╟─6bcf1e9a-bb40-4735-81c0-2420fc9040c0
# ╠═a30a41e5-2608-404f-bc5f-95cb7e7c2dea
# ╠═3ce118c0-aa98-4920-91c4-cec860ae14a2
# ╠═dfb7032e-23e1-49a3-8d3a-fb6b890e20f0
# ╠═f94e1296-ecce-4789-943c-1bf87ac11349
# ╟─6f2ffbe2-290e-43e4-9c1c-b3c5183bfa8f
# ╟─64bbd860-42c9-4dda-a162-e63120ae83c9
# ╟─f7bf8468-5f1f-4857-86c9-1ab9581a36a4
# ╟─818e4a62-1882-4e03-9496-407d799dfae6
# ╠═bf8c2287-4dab-4883-bb28-f2355e1d6bbf
# ╠═42ba83b0-8162-4516-bf35-288a54bf4d7c
# ╠═6bcf7aa6-d3f8-48eb-83d4-8dca7f50cf03
# ╠═9bf8e6e9-efd4-4f2c-9953-bda19d7b8494
# ╟─8eab9d56-8c6c-4fea-ad7c-560f3c9a364c
# ╟─4da5b68f-498c-4075-916e-4ddd8e2a48c0
# ╟─5a61a358-18d9-4b1f-b8c9-f3fc9a115f6e
# ╟─998b21b2-8802-4937-8983-4e7980dd142d
# ╠═f1064a13-eef8-4bf0-b747-135f0dc8f635
# ╠═105ee14a-40c6-41ef-b908-a2184c3a6109
# ╠═2ccd66c9-166a-486b-b546-205ccb84135d
# ╠═200db744-1c52-404d-ab28-b7b4ff470d2c
# ╟─de98ab8f-f2d4-4877-94e1-5a87dbee8485
# ╟─e76ee3d3-eef6-487b-82d8-a0b615115562
# ╠═0c029b61-c3ef-48aa-b3d2-46e4dc071c6d
