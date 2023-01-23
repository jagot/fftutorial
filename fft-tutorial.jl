### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ cf7caf47-9729-48b9-98fe-dc0ad5c75488
begin
    using Pkg
    Pkg.activate(dirname(@__FILE__))

    using PlutoUI
    import PlutoUI: combine

    using PyPlot
    PyPlot.svg(true)
    using Jagot.plotting

    using LinearAlgebra
    using FFTW
    import DSP: hanning, hamming

    function multi_input(heading, inputs...)
        combine() do Child
            inputs = [md" $(label): $(Child(name, input))"
                      for (name,input,label) in inputs]
            md"""### $(heading)
            $(inputs)
            """
        end
    end
    nothing
end

# ╔═╡ 67a7ac20-98c1-11ed-26c3-bbea299eec72
md"""# FFTutorial

Drs. Chen & Stefanos

The avid reader can focus on the sections marked with a star (⋆) on
the first read-through, other readers can safely skip these.

An excellent reference is

- Oran Brigham, E. (1988). The fast Fourier transform and its
  applications. London, England: Prentice-Hall. ISBN: 0-13-307547-8

"""

# ╔═╡ 0489010d-31c6-4d41-8a6c-0d6581183aa9
md"""
- [X] Inner product
- [X] Complete set of orthogonal functions
- [X] Schwarz class
- [X] Conjugate variables
- [X] DFT, DST, DCT
- [-] Parseval’s theorem
- [X] Apodizing window functions
- [ ] Background removal
- [X] Fourier differentiation ([notes by Steven G. Johnson](https://math.mit.edu/~stevenj/fft-deriv.pdf))
- [X] Frequency axis
- [-] Nyquist/Shannon, folding
    - [X] Higher sampling frequency => higher max frequency
    - [X] More samples => denser frequency grid
    - [ ] Zero padding does not give more info, but easier on the eyes
      [see Fig. 9.2 of Oran Brigham (1988)]
- [ ] Zeta transform
- [ ] Phase
"""

# ╔═╡ 23b0b46e-41c3-487e-a852-c2056cbb3afd
md""" # Theory

## Fourier transform

The _Fourier transform_ (FT) transforms between _conjugate variables_,
examples of which are

- time and frequency,
- position and momentum,
- rotation angle and angular momentum,
- etc.

In one dimension, it is given by
```math
\hat{f}(\omega) =
a_1
\int_{-\infty}^{\infty}
\mathrm{d}t
\mathrm{e}^{-\mathrm{i}\omega t}
f(t),
\tag{FFT}
```
and the inverse transform by
```math
f(t) =
a_2
\int_{-\infty}^{\infty}
\mathrm{d}\omega
\mathrm{e}^{\mathrm{i}\omega t}
\hat{f}(\omega),
\tag{IFFT}
```
where
```math
\omega ≝ 2\pi f,
```
and the normalization has to fulfill
```math
a_1a_2=\frac{1}{2\pi}.
```
Which choice we make decides our convention, the standard transforms
we list in tables, and _Parseval's theorem_ discussed below. Here, we
choose the symmetric ``a_1=a_2=(2\pi)^{-1/2}``, and define our
transforms in terms of ``\omega`` instead of ``f`` (which is also a
common choice). The energy is then given by ``E=\hbar\omega``, instead
of ``E = hf`` [``\hbar ≝ h/(2\pi)``, so both relations are naturally
always true]. See section 2.4 of Oran Brigham (1988) for more information.

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
We note that these functions are also a complete set in frequency
space:
```math
\int\mathrm{d}\omega
|\omega\rangle\langle \omega| =
\hat{\mathbb{1}}.
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
2\pi
a_1^2
\int_{-\infty}^\infty
\mathrm{d}\omega
|\hat{f}(\omega)|^2.
```
With our choice ``a_1=(2\pi)^{-1/2}``, this simplifies to
```math
\int_{-\infty}^\infty
\mathrm{d}t
|f(t)|^2 =
\int_{-\infty}^\infty
\mathrm{d}\omega
|\hat{f}(\omega)|^2
```
This implies that the FT is _unitary_, i.e. it does not
change the norm. If we think in term of quantum mechanics, this is a
trivial statement: the norm of the wavefunction has to be the same in
e.g. the position and momentum representations:
```math
\begin{aligned}
|f|^2 =
\langle f|f\rangle
&=
\begin{cases}
\displaystyle
\int\mathrm{d}t
\langle f|t\rangle
\langle t|f\rangle =
\int\mathrm{d}t
|f(t)|^2 &\\
\displaystyle
\int\mathrm{d}\omega
\langle f|\omega\rangle
\langle \omega|f\rangle =
\int\mathrm{d}\omega
|f(\omega)|^2&
\end{cases}
\end{aligned}
```

Another way to view the Parseval's theorem is to think of it as the
Hilbert-space analogue of Pythagoras' theorem:
``|\mathbf{a}+\mathbf{b}+...|^2=|a|^2+|b|^2+...``, where the Euclidean
vectors ``\mathbf{a},\mathbf{b},...`` are all mutually orthogonal.

This will become important later, when we work with FFT.

## Differentiation

We can very easily figure out what the FT of a time
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

A good way to think about this is that the FT and
differentiation/integration are _linear operations_, which in
particular means that they commute and are distributive, and we can
perform them in any order, i.e. FT the
differentiated/integrated function, or differentiate/integrate the
Fourier modes after transforming; the latter corresponds to
multiplication/division by ``(\mathrm{i}\omega)^n``.

### ⋆ Side note on accuracy of differential operator approximations

Using the FT to evaluate the derivative of a function
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

## Known transform pairs

To debug the use of a numeric FT, it is useful to
compare with functions whose analytic FTs are known to
us. However, functions like ``\sin(2\pi t)`` and ``\exp(\mathrm{i}2\pi
t)`` have distributional FTs, which are not numerically
representable. Instead we turn to the

### Schwarz space

which consists of compact functions whose FTs also lie
in the Schwarz space. The most famous example is the Gaussian:

```math
\exp(-\alpha t^2)
\leftrightarrow
\frac{1}{\sqrt{2\alpha}}
\exp\left(
-\frac{\omega^2}{4\alpha}
\right)
```
From this, we find the FT of a normalized, shifted Gaussian as
```math
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left[
-\frac{(t-\mu)^2}{2\sigma^2} +
\mathrm{i}\omega_0 t
\right]
\leftrightarrow
\frac{1}{\sqrt{2\pi}}
\exp\left[
-\frac{\sigma^2(\omega-\omega_0)^2}{2}
-\mathrm{i}\mu(\omega-\omega_0)
\right]
```

where we have used

- the shift theorem: ``f(t-a) \leftrightarrow
  \mathrm{e}^{-\mathrm{i}a\omega}\hat{f}(\omega)``, and

- the scaling theorem: ``f(at) \leftrightarrow
  \frac{1}{|a|}\hat{f}(\frac{\omega}{a})``.

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
DFT definition. Note that we do not necessarily only can apply the FFT
to power-of-two problems only, since any number ``N`` can be
decomposed into power-of-two numbers, with some loss in performance.

_The_ standard implementation of FFT is [FFTW](https://www.fftw.org/)
(Fastest Fourier Transform in the West), written by Steven G. Johnson
_et al._
"""

# ╔═╡ d3cc841e-9f41-437c-a8a9-3e9826824b0a
md"""#### FFT grid

The FFT evaluates the spectrum ``X_k`` at the following frequencies:

```math
\frac{f_s}{n}\times
\begin{cases}
[0,1,...,\lfloor\frac{n}{2}\rfloor-1,-\lfloor\frac{n}{2}\rfloor,-\lfloor\frac{n}{2}\rfloor+1,...,-1],
& n \textrm{ even}, \\
[0,1,...,\lfloor\frac{n-1}{2}\rfloor,-\lfloor\frac{n-1}{2}\rfloor,-\lfloor\frac{n-1}{2}\rfloor+1,...,-1],
& n \textrm{ odd},
\end{cases}
```
where
```math
f_s ≝ \frac{1}{\delta t}
```
is the sampling frequency, given by the inverse of the time step ``\delta t``.
"""

# ╔═╡ f19181e4-a11c-4df8-9eaa-a64bd8e6afa5
fftfreq(4, 1)

# ╔═╡ 6d886a64-c00b-4113-9a89-e28f2e10633c
fftfreq(5, 2)

# ╔═╡ 64368015-f25e-4f98-99ee-83820864bcb5
md"""
We can then use `fftshift` to get the frequencies in the "right" order:
"""

# ╔═╡ a4b57014-5c65-4301-bbc5-6e2ded20e647
fftshift(fftfreq(4,1))

# ╔═╡ 41e7fae4-124a-48a6-9e20-89f926b9633f
md"""
We also have to multiply by ``2\pi`` to get angular frequencies:
"""

# ╔═╡ 339f8799-4232-4d87-85ce-8518d838a7b6
2π*fftshift(fftfreq(4,1))

# ╔═╡ 6904a9e8-2506-479c-8373-6c8a8282957a
md""" Therefore, we define the following helper function, that
performs the `fftshift` and multiplication automatically (we can even
provide a specialized version of the function that accepts uniform
grid and deduces the correct function parameters therefrom): """

# ╔═╡ 8a288f71-304f-4b26-9128-c8671ee063f5
begin
    fftω(args...) = 2π*fftshift(fftfreq(args...))
    fftω(t::AbstractRange) = fftω(length(t), 1/step(t))
end

# ╔═╡ b3d09140-4efd-45ef-a64b-974f7a7d57bb
md"""
#### Normalization (Parseval's theorem)

The FFT will the compute the DFT according to
```math
X_k =
\sum_{n=1}^{N}
\exp\left[
-\mathrm{i}\frac{2\pi
(n-1)(k-1)}{N}
\right] x_n,
\tag{FFT}
```
whereas the inverse FFT will compute
```math
X_k =
\frac{1}{N}
\sum_{n=1}^{N}
\exp\left[
+\mathrm{i}\frac{2\pi (n-1)(k-1)}
{N}
\right] x_n,
\tag{IFFT}
```
which we note is unsymmetrically normalized. The point is that the
FFT–IFFT transform pair should be overall unitary, but if we only need
the FFT transform, we can avoid a potentially unnecessary
division. However, if we are interested in the spectral amplitudes, we
have to perform the normalization ourselves.

To illustrate this, we first compare the FFT of a Gaussian with the
analytically known FT given above.

TODO: Derive normalization used below.

⋆ The reason for the seemingly odd choices of constants is the first
rule of numeric debugging: never choose 0 or 1 as a test value, since
we may then inadvertenly miss bugs because the function has nice
values there (example ``x=x^2=x^3=...`` at ``x=0`` _and_ ``x=1``,
which tells us nothing about the behaviour of the function).

"""

# ╔═╡ 10d23a53-9645-4c6b-98ec-9ed9ae28eeb7
t1 = range(-51, stop=51, step=0.31)

# ╔═╡ 400d17df-8133-4155-9bec-aceab27fa9e8
ω1 = fftω(t1)

# ╔═╡ 596abc03-335b-426a-8f2c-127be43a6ff9
A1,μ1,σ1,a1 = 1.45,4.65,1.76,4.0

# ╔═╡ ca9a70ff-5059-4332-bad0-c921550c7e45
y1 = A1/(√(2π)*σ1)*exp.(-(t1 .- μ1).^2/(2σ1^2)) .* exp.(im*a1*t1)

# ╔═╡ f5d076c3-63c2-4502-97f5-d0c6abf2a36f
Y1 = fftshift(fft(y1))*(t1[end]-t1[1])/(√(2π)*length(t1))

# ╔═╡ ea6fa10e-4c1f-4347-8115-3e781a7e7252
Y1exact = A1/√(2π)*exp.(-σ1^2*(ω1 .- a1).^2/2 .- im*(μ1-t1[1])*(ω1 .- a1))

# ╔═╡ d3a060e5-d742-4d47-94e4-bf305ca24011
cfigure("Normalization", figsize=(8,10)) do
    csubplot(211) do
        plot(t1, y1)
        xlabel(L"t")
        ylabel(L"y(t)")
        axes_labels_opposite(:x)
    end
    csubplot(4,2,(3,1),nox=true) do
        plot(t1, real(Y1), label=L"\Re\{Y(\omega)\}")
        plot(t1, real(Y1exact), label=L"$\Re\{Y(\omega)\}$ exact")
        legend()
    end
    csubplot(4,2,(4,1)) do
        plot(t1, imag(Y1), label=L"\Im\{Y(\omega)\}")
        plot(t1, imag(Y1exact), label=L"$\Im\{Y(\omega)\}$ exact")
        legend()
        xlabel(L"\omega")
    end
    csubplot(224) do
        plot(t1, abs2.(Y1), label=L"|Y(\omega)|^2")
        plot(t1, abs2.(Y1exact), label=L"$|Y(\omega)|^2$ exact")
        legend()
        xlabel(L"\omega")
        axes_labels_opposite(:y)
    end
end

# ╔═╡ e9635e7c-6eac-4cdb-81d3-f551cb3af9d9
md"""
Equipped with this knowledge, we can define a helper function that
returns the correctly normalized FFT:

```math
\operatorname{NFFT}(y;t) ≝ \operatorname{FFT}(y)\frac{t_N-t_1}{N\sqrt{2\pi}}
```
"""

# ╔═╡ 5d86a89a-dc2f-4a6f-9462-12d78e4a75f4
nfft(y, t) = fftshift(fft(y))*(t[end]-t[1])/(length(t)*√(2π))

# ╔═╡ c0dfab29-a96c-40bb-85cb-776ccd661f68
md""" # Sampling

We define our test function, which is a normalized Gaussian multiplying two frequencies:

```math
f(t,\sigma) =
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(
-\frac{t^2}{2\sigma^2}
\right)
(a\sin2\pi t
+\sin b\pi t)
```
"""

# ╔═╡ d8e9c2eb-8e7f-427c-b785-f7b504145dc7
f(t, σ; a=1.2, b=4.5) = 1/(√(2π)*σ) * exp(-t^2/(2σ^2)) * (a*sin(2π*t) + sin(b*π*t))

# ╔═╡ 178f1fc5-d6d8-473b-90dc-6a2ff0ec24e4
# ╠═╡ disabled = true
#=╠═╡
# f(t, σ) = 1/(√(2π)*σ) * exp(-t^2/(2σ^2)) * sin(2π*t)
  ╠═╡ =#

# ╔═╡ f6feb286-b126-4855-80ce-c0e7248c2f0e

md"""We also define its first and second derivatives, which we will
need later:

```math
\begin{aligned}
f'(t,\sigma)
&=
-\frac{t}{\sigma^2}
f(t,\sigma)+
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(
-\frac{t^2}{2\sigma^2}
\right)
\pi(2a\cos2\pi t
+b\cos b\pi t), \\
f''(t,\sigma)
&=
-\frac{1}{\sigma^2}[
f(t,\sigma)+
tf'(t,\sigma)] -
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(
-\frac{t^2}{2 \sigma^2}
\right) \times \\
&\qquad
\left[
\frac{t}{\sigma^2}
\pi(2 a \cos{2 \pi t} + b \cos{b \pi t}) +
\pi^2(4 a \sin{2 \pi t} + b^2 \sin{b \pi t})
\right]
\end{aligned}
```
"""

# ╔═╡ bdba5492-bea8-4268-a1ea-8b57f15fe27e
f′(t, σ; a=1.2, b=4.5) = -t/σ^2*f(t, σ; a=a, b=b) + 1/(√(2π)*σ) * exp(-t^2/(2σ^2)) * π*(2a*cos(2π*t) + b*cos(b*π*t))

# ╔═╡ 1932ded5-3117-4912-9bf7-6510e15f6063
# ╠═╡ disabled = true
#=╠═╡
# f′(t, σ) = -t/σ^2*f(t, σ) + 1/(√(2π)*σ) * exp(-t^2/(2σ^2)) * 2π*cos(2π*t)
  ╠═╡ =#

# ╔═╡ 7f74e7ae-961f-46b9-9704-8a3900e4e907
f′′(t, σ; a=1.2, b=4.5) = -1/σ^2*(f(t, σ; a=a, b=b) + t*f′(t, σ; a=a, b=b)) -
    1/(√(2π)*σ) * exp(-t^2/(2σ^2)) *
    (t/σ^2*π*(2a*cos(2π*t) + b*cos(b*π*t)) +
    π^2*(4a*sin(2π*t) + b^2*sin(b*π*t)))

# ╔═╡ 6ada28ff-5d51-4199-9955-833d131c4633
# ╠═╡ disabled = true
#=╠═╡
# f′′(t, σ) = -1/σ^2*(f(t, σ) + t*f′(t, σ)) -
#     1/(√(2π)*σ) * exp(-t^2/(2σ^2)) *
#     (t/σ^2*2π*cos(2π*t) +
#     4π^2*sin(2π*t))
  ╠═╡ =#

# ╔═╡ a5e79680-8d5a-4d36-b08d-3d32eaf3b751
md"Here is our standard time grid:"

# ╔═╡ ab40d1ac-aff4-4161-907d-aa5688857977
t = range(-15, stop=15, length=150)

# ╔═╡ b9404c8f-0a45-42c3-addc-825981d10814
md"And here is the corresponding frequency grid:"

# ╔═╡ 28e307a3-d169-4371-8a2b-b64ce9d5d6cb
ω = fftω(t)

# ╔═╡ 2374f003-1b73-4d04-8196-713983aa497e
y = f.(t, 2)

# ╔═╡ 7c0a4378-42ed-4a43-9d35-dc075f99200a
Y = nfft(y, t)

# ╔═╡ abfaf313-e881-48f5-b4b9-eed62505a5d0
md"## Oversampling"

# ╔═╡ 6bcf1e9a-bb40-4735-81c0-2420fc9040c0
md"We first investigate what happens if we change the sampling frequency:"

# ╔═╡ a30a41e5-2608-404f-bc5f-95cb7e7c2dea
tfine = range(-15, stop=15, length=300)

# ╔═╡ 3ce118c0-aa98-4920-91c4-cec860ae14a2
ωfine = fftω(tfine)

# ╔═╡ dfb7032e-23e1-49a3-8d3a-fb6b890e20f0
yfine = f.(tfine, 2)

# ╔═╡ f94e1296-ecce-4789-943c-1bf87ac11349
Yfine = nfft(yfine, tfine)

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
md"## Undersampling (aliasing, \"folding\")"

# ╔═╡ bf8c2287-4dab-4883-bb28-f2355e1d6bbf
tcoarse = range(-15, stop=15, length=75)

# ╔═╡ 42ba83b0-8162-4516-bf35-288a54bf4d7c
ωcoarse = fftω(tcoarse)

# ╔═╡ 6bcf7aa6-d3f8-48eb-83d4-8dca7f50cf03
ycoarse = f.(tcoarse, 2)

# ╔═╡ 9bf8e6e9-efd4-4f2c-9953-bda19d7b8494
Ycoarse = nfft(ycoarse, tcoarse)

# ╔═╡ 8eab9d56-8c6c-4fea-ad7c-560f3c9a364c
cfigure("function coarse sampling") do
    plot(t, y)
    plot(tcoarse, ycoarse)
    xlabel(L"t")
    ylabel(L"y(t)")
end

# ╔═╡ 4da5b68f-498c-4075-916e-4ddd8e2a48c0
cfigure("Fourier transform coarse sampling") do
    csubplot(211,nox=true) do
        plot(ω, abs2.(Y))
        plot(ωcoarse, abs2.(Ycoarse))
    end
    csubplot(212) do
        semilogy(ω, abs2.(Y))
        semilogy(ωcoarse, abs2.(Ycoarse))
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
    plot(t, y2)
    xlabel(L"t")
    ylabel(L"y(t)")
end

# ╔═╡ 2ccd66c9-166a-486b-b546-205ccb84135d
Y2 = nfft(y2, t)

# ╔═╡ 200db744-1c52-404d-ab28-b7b4ff470d2c
cfigure("Fourier transform") do
    plot(ω, abs2.(Y))
    plot(ω, abs2.(Y2))
end

# ╔═╡ de98ab8f-f2d4-4877-94e1-5a87dbee8485
md"""The dashed spectrum is sharper, since the pulse is longer, and we
have more time measure the actual frequency content."""

# ╔═╡ 1fa261ac-0356-4786-89a5-32ba4fae48a1
md"""## Custom sampling

With these sliders, we can easily see what happens when we change the different parameters
"""

# ╔═╡ 2ba55a25-eae5-491c-a5f3-c28a252f5805
@bind sampling_parameters multi_input("Sampling parameters",
                                      (:time_step, Slider(range(0.001, stop=0.5, length=101), default=0.2, show_value=true),
                                       "Time step"),
                                      (:tmax, Slider(range(5, stop=70, length=101), default=15, show_value=true),
                                       L"t_{\mathrm{max}}"),
                                      (:pulse_length, Slider(range(0.5, stop=10, length=101), default=2.0, show_value=true),
                                       "Pulse length"))

# ╔═╡ a28c85d4-c8c2-4f03-b043-9ad448eee9e4
tcustom = range(-sampling_parameters.tmax, stop=sampling_parameters.tmax,
                step=sampling_parameters.time_step)

# ╔═╡ 79b19bdd-ad18-4efe-b7b9-a45fb50ef168
ωcustom = fftω(tcustom)

# ╔═╡ cc344770-f189-4693-8dc3-b1ced5ebc219
ycustom = f.(tcustom, sampling_parameters.pulse_length)

# ╔═╡ cbb3a102-11f1-4b8f-ab0c-7e51bd5e8031
Ycustom = nfft(ycustom, tcustom)

# ╔═╡ 833df8f8-157a-4c73-826e-cb73c0b2030f
cfigure("function custom sampling", figsize=(8,10)) do

    csubplot(211) do
        plot(t, y)
        plot(tcustom, ycustom)
        xlabel(L"t")
        ylabel(L"y(t)")
        ylim(-0.45, 0.45)
        axes_labels_opposite(:x)
    end
    csubplot(212) do
        plot(ω, abs2.(Y))
        plot(ωcustom, abs2.(Ycustom))
        xlabel(L"$\omega$ [rad]")
        ylabel(L"|Y(\omega)|^2")
    end
end

# ╔═╡ e76ee3d3-eef6-487b-82d8-a0b615115562
md""" # Apodizing window functions & background removal

Up to now, we only considered functions with compact support,
i.e. they were zero (or very close to) at the boundaries. We will now
investigate what happens if we FFT functions which are non-zero at the
boundaries, and what we can do to solve the arising difficulties.

[Wikipedia entry on window functions](https://en.wikipedia.org/wiki/Window_function)

"""

# ╔═╡ 339809db-595c-49bf-9ac9-e2de6b629149
twind = range(0, stop=15, length=501)

# ╔═╡ 5a20af58-e487-47df-a815-2613829e8b96
ωwind = fftω(twind)

# ╔═╡ c06cce88-2963-4ca3-92ea-97e947040ef7
k = 4.0

# ╔═╡ df729903-1de0-4f52-a8ec-db9b8406ceb1
N = 1/√(twind[end]-twind[1])

# ╔═╡ 0c029b61-c3ef-48aa-b3d2-46e4dc071c6d
yrect = N*cos.(2π*k*twind)

# ╔═╡ ddef623b-a2dc-4a6a-abf1-a252aa7fe5ff
Yrect = nfft(yrect, twind)

# ╔═╡ 2eebadbc-49e1-4078-a728-9342e1ee190c
hannwind = hanning(length(twind))

# ╔═╡ 8c6e2637-46a3-435b-9e0d-0665e59930db
hammingwind = hamming(length(twind))

# ╔═╡ 30d9bfba-c8c9-4aaa-8bee-0a85c8d261db
md"""The [Hann function](https://en.wikipedia.org/wiki/Hann_function)
(often misnomed Hanning window, in confusion with the similar [Hamming
window](https://en.wikipedia.org/wiki/List_of_window_functions#Hann_and_Hamming_windows))
smoothly truncates the signal at the edges of the domain: """

# ╔═╡ 240d0868-26f6-4af6-9073-d0c764006077
cfigure("windows") do
    plot(twind, hannwind, label="Hann function")
    plot(twind, hammingwind, label="Hamming window")
    xlabel(L"t")
    ylabel(L"w_0(t)")
    legend()
end

# ╔═╡ b8196d13-8c79-410a-b954-a624cd1865fb
yhann = hannwind .* yrect

# ╔═╡ 1eb05b54-290a-4ed7-857c-05e39875c6d6
Yhann = nfft(yhann, twind)

# ╔═╡ 69c70e3e-dad2-4667-9fa9-baa1372db4cb
yhamming = hammingwind .* yrect

# ╔═╡ d66d32d5-92fc-4ec0-b510-45168eef17f0
Yhamming = nfft(yhamming, twind)

# ╔═╡ 6547d5c7-98b8-4449-9f50-f94b9a49ed3d
cfigure("rectangular window") do
    csubplot(211) do
        plot(twind, yrect)
        plot(twind, yhann)
        plot(twind, yhamming, "--")
        xlabel(L"t")
        ylabel(L"y(t)")
        axes_labels_opposite(:x)
    end
    csubplot(212) do
        semilogy(ωwind, abs2.(Yrect), label="Rectangular window")
        semilogy(ωwind, abs2.(Yhann), label="Hann window")
        semilogy(ωwind, abs2.(Yhamming), "--", label="Hamming window")
        axvline(2π*k, linestyle="--", color="black", linewidth=1.0)
        xlabel(L"$\omega$ [rad]")
        ylabel(L"|Y(\omega)|^2")
        ylim(1e-8, 10)
        legend()
    end
end

# ╔═╡ eed0d9b5-c5b0-4889-8c72-546200034b96
md"""The peaks are much sharper, when we use the Hann window, but it
is not norm-conserving. This is contrast to the rectangular window
(i.e. doing nothing), which _is_ norm-conserving. There are many
different window functions designed to optimize one property or the
other (there is no free lunch). The Hamming window gives yet sharper
peaks, at the cost of a higher noise floor."""

# ╔═╡ 63e6671f-8107-4764-9869-52a74d149f7a
md"# Background removal"

# ╔═╡ 265ca8dd-2996-41a7-81ba-b44d570f4627
md"""# FFT Differentiation

The algorithms given below are copied verbatim from

- Johnson, Steven G. (2011). [Notes on FFT-based
  differentiation.](https://math.mit.edu/~stevenj/fft-deriv.pdf)

Note that we _do not_ add our own normalization as we did in the
helper function `nfft`; as mentioned earlier, the FFT–IFFT constitute
a unitary transform pair, and any algorithm that performs first an FFT
and subsequently an IFFT will automatically have the right norm for
the end result.
"""

# ╔═╡ f087417a-2ef1-47b2-b84f-a392940b4fbe
md"""## Algorithm 1

Compute the sampled first derivative ``y_n' ≈ y'(nL/N)`` from samples ``y_n = y(nL/N)``.

1. Given ``y_n`` for ``0≤n<N``, use an FFT to compute ``Y_k`` for ``0≤k<N``.

2. Multiply ``Y_k`` by ``2π\mathrm{i} k`` for ``k < N/2``, by
   ``2π\mathrm{i} (k − N )`` for ``k > N/2``, and by zero for ``k =
   N/2`` (if ``N`` is even), to obtain ``Y_k'``.

3. Compute ``y_n'`` from ``Y_k'`` via an inverse FFT.
"""

# ╔═╡ 2c867bc5-9a39-4557-a15f-8aee6e77a0ff
function fft_derivative(y, fs=1; apodization::Function=N->true)
    # This implements Algorithm 1 by Johnson (2011).
    N = length(y)
    ω = 2π*fftfreq(length(y), fs)
    Y = fft(y .* apodization(N))
    Y′ = im*ω .* Y
    if iseven(N)
        Y′[N÷2] = 0
    end
    ifft(Y′)
end

# ╔═╡ 570ea7f6-6723-47b8-a6af-99df7cf9059f
tderiv = range(-10, stop=10, length=1001)

# ╔═╡ df940576-fd46-47d0-adfd-e6622822b5b1
yderiv = f.(tderiv, 2)

# ╔═╡ 095883b3-b21a-4cba-ba3a-96f0f0beb5ac
y′deriv = fft_derivative(yderiv, 1/step(tderiv))

# ╔═╡ 892be44e-e10d-4304-9942-39a9c643d04a
md"""We expect the derivative of a real function to be real as well,
we check this by computing the norm of the imaginary part:"""

# ╔═╡ c4e0a0c1-e5f6-4991-b37f-df8115e15968
norm(imag(y′deriv))

# ╔═╡ 5c7a235f-95e0-48fe-87b0-9adad3422add
y′deriv_exact = f′.(tderiv, 2)

# ╔═╡ 50044cbf-2ec3-4ffc-8803-e3177eb69b7c
cfigure("fft differentation", figsize=(8,10)) do
    csubplot(311, nox=true) do
        plot(tderiv, yderiv)
    end
    csubplot(312, nox=true) do
        plot(tderiv, real(y′deriv), label="Derivative via FFT")
        plot(tderiv, y′deriv_exact, "--", label="Exact derivative")
        legend()
    end
    csubplot(313) do
        let err = abs.(y′deriv-y′deriv_exact)
            plot(tderiv, err, label="FFT derivative error")
            maximum(err) < 1e-2 && yscale("log")
        end
        xlabel(L"t")
        legend()
    end
end

# ╔═╡ 6f8bdbfa-ff30-4df8-a400-fcb69f434003

md"""The errors are very small, but increase towards the edges, since
even though our signal is very close to zero, it is not exactly
so. Furthermore, the exact derivative assumes a non-periodic function,
whereas FFT/IFFT actually _requires_ the function to be periodic, and
enforces the periodicity ``t_{\textrm{max}}-t_{\textrm{min}}``.

We thus investigate the influence of apodizing (windowing) our signal,
before the FFT:
"""

# ╔═╡ eb375d72-66f9-4201-b601-12ccf82b239e
y′deriv_hann = fft_derivative(yderiv, 1/step(tderiv), apodization=hanning)

# ╔═╡ 5c5b6c6e-622c-4dd7-b20f-49e2e2ea620b
cfigure("fft windowed differentation", figsize=(8,10)) do
    csubplot(311, nox=true) do
        plot(tderiv, yderiv)
    end
    csubplot(312, nox=true) do
        plot(tderiv, real(y′deriv_hann), label="Derivative via FFT + Hann")
        plot(tderiv, y′deriv_exact, "--", label="Exact derivative")
        legend()
    end
    csubplot(313) do
        let err = abs.(y′deriv_hann-y′deriv_exact)
            plot(tderiv, err, label="FFT derivative + Hann error")
            maximum(err) < 1e-2 && yscale("log")
        end
        xlabel(L"t")
        legend()
    end
end

# ╔═╡ d0a81caa-1973-4e80-8978-6f39e4234a57
md"""This illustrates that windowing is not necessarily always the
correct thing to do."""

# ╔═╡ 43459ba8-8e3d-4851-9bbd-30f29e7788ad
md"""## Algorithm 2

Compute the sampled second derivative ``y'' ≈ y''(nL/N)`` from samples ``y_n = y(nL/N)``.

1. Given ``y_n`` for ``0\le n<N``, use an FFT to compute ``Y_k`` for
   ``0\le k<N``.

2. Multiply ``Y_k`` by ``−[\frac{2π}{L}k]^2`` for ``k≤N/2`` and by
   ``−[\frac{2π}{L}(k−N)]^2`` for ``k>N/2`` to obtain ``Y_k''``.

3. Compute ``y_n''`` from ``Y_k''`` via an inverse FFT.
"""

# ╔═╡ a169f024-3c89-464d-b9bf-f2b965cdb72c
function fft_second_derivative(y, fs=1; apodization::Function=N->true)
    # This implements Algorithm 2 by Johnson (2011).
    N = length(y)
    ω² = (2π*fftfreq(length(y), fs)).^2
    Y = fft(y .* apodization(N))
    Y′′ = -ω² .* Y
    ifft(Y′′)
end

# ╔═╡ 2b128426-045d-4fb0-8cc3-25dd9ae02da4
y′′deriv = fft_second_derivative(yderiv, 1/step(tderiv))

# ╔═╡ 8c290ee4-2b0e-48a4-898c-d4444f246455
md"""Again, we expect the second derivative of a real function to be
real as well:"""

# ╔═╡ 5a58d0e6-5525-49c4-8bd2-4310c247ddbd
norm(imag(y′′deriv))

# ╔═╡ da9115b5-85e1-4227-bd5a-b84355497552
y′′deriv_exact = f′′.(tderiv, 2)

# ╔═╡ efde8cf9-f369-49a0-838c-af33d6d0102d
cfigure("fft second differentation", figsize=(8,10)) do
    csubplot(311, nox=true) do
        plot(tderiv, yderiv)
    end
    csubplot(312, nox=true) do
        plot(tderiv, real(y′′deriv), label="Second derivative via FFT")
        plot(tderiv, y′′deriv_exact, "--", label="Exact second derivative")
        legend()
    end
    csubplot(313) do
        let err = abs.(y′′deriv-y′′deriv_exact)
            plot(tderiv, err, label="FFT second derivative error")
            maximum(err) < 1e-2 && yscale("log")
        end
        xlabel(L"t")
        legend()
    end
end

# ╔═╡ Cell order:
# ╟─67a7ac20-98c1-11ed-26c3-bbea299eec72
# ╟─0489010d-31c6-4d41-8a6c-0d6581183aa9
# ╟─cf7caf47-9729-48b9-98fe-dc0ad5c75488
# ╟─23b0b46e-41c3-487e-a852-c2056cbb3afd
# ╟─d3cc841e-9f41-437c-a8a9-3e9826824b0a
# ╠═f19181e4-a11c-4df8-9eaa-a64bd8e6afa5
# ╠═6d886a64-c00b-4113-9a89-e28f2e10633c
# ╟─64368015-f25e-4f98-99ee-83820864bcb5
# ╠═a4b57014-5c65-4301-bbc5-6e2ded20e647
# ╟─41e7fae4-124a-48a6-9e20-89f926b9633f
# ╠═339f8799-4232-4d87-85ce-8518d838a7b6
# ╟─6904a9e8-2506-479c-8373-6c8a8282957a
# ╠═8a288f71-304f-4b26-9128-c8671ee063f5
# ╟─b3d09140-4efd-45ef-a64b-974f7a7d57bb
# ╠═10d23a53-9645-4c6b-98ec-9ed9ae28eeb7
# ╠═400d17df-8133-4155-9bec-aceab27fa9e8
# ╠═596abc03-335b-426a-8f2c-127be43a6ff9
# ╠═ca9a70ff-5059-4332-bad0-c921550c7e45
# ╠═f5d076c3-63c2-4502-97f5-d0c6abf2a36f
# ╠═ea6fa10e-4c1f-4347-8115-3e781a7e7252
# ╟─d3a060e5-d742-4d47-94e4-bf305ca24011
# ╟─e9635e7c-6eac-4cdb-81d3-f551cb3af9d9
# ╠═5d86a89a-dc2f-4a6f-9462-12d78e4a75f4
# ╟─c0dfab29-a96c-40bb-85cb-776ccd661f68
# ╠═d8e9c2eb-8e7f-427c-b785-f7b504145dc7
# ╟─178f1fc5-d6d8-473b-90dc-6a2ff0ec24e4
# ╟─f6feb286-b126-4855-80ce-c0e7248c2f0e
# ╠═bdba5492-bea8-4268-a1ea-8b57f15fe27e
# ╟─1932ded5-3117-4912-9bf7-6510e15f6063
# ╠═7f74e7ae-961f-46b9-9704-8a3900e4e907
# ╟─6ada28ff-5d51-4199-9955-833d131c4633
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
# ╟─1fa261ac-0356-4786-89a5-32ba4fae48a1
# ╟─2ba55a25-eae5-491c-a5f3-c28a252f5805
# ╟─833df8f8-157a-4c73-826e-cb73c0b2030f
# ╠═a28c85d4-c8c2-4f03-b043-9ad448eee9e4
# ╠═79b19bdd-ad18-4efe-b7b9-a45fb50ef168
# ╠═cc344770-f189-4693-8dc3-b1ced5ebc219
# ╠═cbb3a102-11f1-4b8f-ab0c-7e51bd5e8031
# ╟─e76ee3d3-eef6-487b-82d8-a0b615115562
# ╠═339809db-595c-49bf-9ac9-e2de6b629149
# ╠═5a20af58-e487-47df-a815-2613829e8b96
# ╠═c06cce88-2963-4ca3-92ea-97e947040ef7
# ╠═df729903-1de0-4f52-a8ec-db9b8406ceb1
# ╠═0c029b61-c3ef-48aa-b3d2-46e4dc071c6d
# ╠═ddef623b-a2dc-4a6a-abf1-a252aa7fe5ff
# ╠═2eebadbc-49e1-4078-a728-9342e1ee190c
# ╠═8c6e2637-46a3-435b-9e0d-0665e59930db
# ╟─30d9bfba-c8c9-4aaa-8bee-0a85c8d261db
# ╟─240d0868-26f6-4af6-9073-d0c764006077
# ╠═b8196d13-8c79-410a-b954-a624cd1865fb
# ╠═1eb05b54-290a-4ed7-857c-05e39875c6d6
# ╠═69c70e3e-dad2-4667-9fa9-baa1372db4cb
# ╠═d66d32d5-92fc-4ec0-b510-45168eef17f0
# ╟─6547d5c7-98b8-4449-9f50-f94b9a49ed3d
# ╟─eed0d9b5-c5b0-4889-8c72-546200034b96
# ╟─63e6671f-8107-4764-9869-52a74d149f7a
# ╟─265ca8dd-2996-41a7-81ba-b44d570f4627
# ╟─f087417a-2ef1-47b2-b84f-a392940b4fbe
# ╠═2c867bc5-9a39-4557-a15f-8aee6e77a0ff
# ╠═570ea7f6-6723-47b8-a6af-99df7cf9059f
# ╠═df940576-fd46-47d0-adfd-e6622822b5b1
# ╠═095883b3-b21a-4cba-ba3a-96f0f0beb5ac
# ╟─892be44e-e10d-4304-9942-39a9c643d04a
# ╠═c4e0a0c1-e5f6-4991-b37f-df8115e15968
# ╠═5c7a235f-95e0-48fe-87b0-9adad3422add
# ╟─50044cbf-2ec3-4ffc-8803-e3177eb69b7c
# ╟─6f8bdbfa-ff30-4df8-a400-fcb69f434003
# ╠═eb375d72-66f9-4201-b601-12ccf82b239e
# ╟─5c5b6c6e-622c-4dd7-b20f-49e2e2ea620b
# ╟─d0a81caa-1973-4e80-8978-6f39e4234a57
# ╟─43459ba8-8e3d-4851-9bbd-30f29e7788ad
# ╠═a169f024-3c89-464d-b9bf-f2b965cdb72c
# ╠═2b128426-045d-4fb0-8cc3-25dd9ae02da4
# ╠═8c290ee4-2b0e-48a4-898c-d4444f246455
# ╠═5a58d0e6-5525-49c4-8bd2-4310c247ddbd
# ╠═da9115b5-85e1-4227-bd5a-b84355497552
# ╟─efde8cf9-f369-49a0-838c-af33d6d0102d
