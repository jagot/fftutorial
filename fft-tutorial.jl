### A Pluto.jl notebook ###
# v0.19.22

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
    using PlutoUI
    import PlutoUI: combine

    using Plots
    using LaTeXStrings

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
- [X] Background removal
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
\tag{FT}
```
and the inverse transform by
```math
f(t) =
a_2
\int_{-\infty}^{\infty}
\mathrm{d}\omega
\mathrm{e}^{\mathrm{i}\omega t}
\hat{f}(\omega),
\tag{IFT}
```
where
```math
\omega ≝ 2\pi\nu,
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
``\langle t|f\rangle`` onto the basis functions ``\langle
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

The inverse transform is very similar:
```math
\tag{IDFT}
x_n=
\frac{1}{N}
\sum_{n=0}^{N-1}
X_{k}\cdot
\mathrm{e}^{{\frac {\mathrm{i}2\pi }{N}}kn}.
```
Note that in contrast to the continuous Fourier transform, the
DFT/IDFT pair is almost universally defined with the normalization
``N^{-1}`` attached to the inverse transform. This avoids a
potentially unnecessary division, if only the DFT is needed, and the
normalization is unimportant.

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
let
    p1 = plot(t1, [real(y1) imag(y1)],
              label=[L"\Re" L"\Im"],
              xlabel=L"t", ylabel=L"y_1(t)",
              xmirror=true, lw=2)

    p2 = plot(ω1, [real(Y1) real(Y1exact)],
              label=[L"\Re\{Y(\omega)\}" L"$\Re\{Y(\omega)\}$ exact"],
              leg=:topleft, xaxis=false, lw=2)
    p3 = plot(ω1, [imag(Y1) imag(Y1exact)],
              label=[L"\Im\{Y(\omega)\}" L"$\Im\{Y(\omega)\}$ exact"],
              xlabel=L"\omega", leg=:topleft, lw=2)
    p4 = plot(ω1, [abs2.(Y1) abs2.(Y1exact)],
              label=[L"|Y(\omega)|^2" L"$|Y(\omega)|^2$ exact"],
              xlabel=L"\omega",
              leg=:topleft, ymirror=true, lw=2)

    plot(p1, p2, p3, p4, layout=@layout([a; [[b;c] d]]), size=(700,900))
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

# ╔═╡ 7f74e7ae-961f-46b9-9704-8a3900e4e907
f′′(t, σ; a=1.2, b=4.5) = -1/σ^2*(f(t, σ; a=a, b=b) + t*f′(t, σ; a=a, b=b)) -
    1/(√(2π)*σ) * exp(-t^2/(2σ^2)) *
    (t/σ^2*π*(2a*cos(2π*t) + b*cos(b*π*t)) +
    π^2*(4a*sin(2π*t) + b^2*sin(b*π*t)))

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
let
    p1 = plot(t, y, xlabel=L"t", ylabel=L"y(t)", lw=2, label="Normal sampling")
    plot!(p1, tfine, yfine, lw=2, ls=:dash, label="Fine sampling")
end

# ╔═╡ 64bbd860-42c9-4dda-a162-e63120ae83c9
let
    p1 = plot(ω, abs2.(Y), lw=2, label="Normal sampling", xaxis=false)
    plot!(p1, ωfine, abs2.(Yfine), ls=:dash, lw=2, label="Fine sampling")

    p2 = plot(ω, abs2.(Y), lw=2, yaxis=(:log10, [1e-40, :auto]), legend=false, xlabel=L"$\omega$ [rad]")
    plot!(p2, ωfine, abs2.(Yfine), ls=:dash, lw=2)

    plot(p1,p2, layout=@layout([a;b]), size=(700,800))
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
let
    p1 = plot(t, y, label="Normal sampling", xlabel=L"t", ylabel=L"y(t)", lw=2)
    plot!(p1, tcoarse, ycoarse, label="Coarse sampling", lw=2)
end

# ╔═╡ 4da5b68f-498c-4075-916e-4ddd8e2a48c0
let
    p1 = plot(ω, abs2.(Y), lw=2, label="Normal sampling", xaxis=false)
    plot!(p1, ωcoarse, abs2.(Ycoarse), ls=:dash, lw=2, label="Coarse sampling")

    p2 = plot(ω, abs2.(Y), lw=2, yaxis=(:log10, [1e-35, :auto]), legend=false, xlabel=L"$\omega$ [rad]")
    plot!(p2, ωcoarse, abs2.(Ycoarse), ls=:dash, lw=2)

    plot(p1,p2, layout=@layout([a;b]), size=(700,800))
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
let
    p = plot(t, y, xlabel=L"t", ylabel=L"y(t)", label="Normal pulse", lw=2)
    plot!(p, t, y2, label="Longer pulse", lw=2)
end

# ╔═╡ 2ccd66c9-166a-486b-b546-205ccb84135d
Y2 = nfft(y2, t)

# ╔═╡ 200db744-1c52-404d-ab28-b7b4ff470d2c
let
    p = plot(ω, abs2.(Y), xlabel=L"\omega", ylabel=L"|Y(\omega)|^2", label="Normal pulse", lw=2)
    plot!(p, ω, abs2.(Y2), label="Longer pulse", lw=2)
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
let
    p1 = plot(t, y, label="Normal sampling", xlabel=L"t", ylabel=L"y(t)", lw=2)
    plot!(p1, tcustom, ycustom, label="Custom sampling", lw=2)

    p2 = plot(ω, abs2.(Y), lw=2, label="Normal sampling", xlabel=L"\omega", ylabel=L"|Y(\omega)|^2")
    plot!(p2, ωcustom, abs2.(Ycustom), lw=2, label="Custom sampling")

    plot(p1,p2, layout=@layout([a;b]), size=(700,800))
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
let
    p = plot(twind, hannwind, label="Hann function", xlabel=L"t", ylabel=L"w_0(t)", lw=2)
    plot!(p, twind, hammingwind, label="Hamming window", lw=2)
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
let
    p1 = plot(twind, yrect, label="Rect window", xlabel=L"t", xmirror=true, ylabel=L"y(t)", lw=2)
    plot!(p1, twind, yhann, label="Hann window", lw=2)
    plot!(p1, twind, yhamming, label="Hamming window", ls=:dash, lw=2)

    p2 = plot(ωwind, abs2.(Yrect), label="Rect window", xlabel=L"$\omega$ [rad]", ylabel=L"|Y(\omega)|^2",
              yaxis=(:log10, (1e-8, 10)), lw=2)
    plot!(p2, ωwind, abs2.(Yhann), label="Hann window", lw=2)
    plot!(p2, ωwind, abs2.(Yhamming), ls=:dash, label="Hamming window", lw=2)
    vline!(p2, [2π*k], ls=:dash, color="black", label=L"\omega=2\pi\times4")

    plot(p1, p2, layout=@layout([a; b]), size=(700,700))
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

# ╔═╡ ff6cb77c-c4fb-4e92-9260-37b5fe41043a
md"""
Sometimes, when our periodic signal sits on top of a constant or
slowly varying background, windowing the samples prior to performing
the FFT is not enough. We may, with varying degrees of success,
improve the computation of the spectrum by first attempting to remove
the background signal.
"""

# ╔═╡ c2e6e2e6-acf6-4258-b121-c5fec5764e08
# Signal on top of linear and cubic background
yb = 0.1t + 1e-3t.^3 + sin.(2.3π*t)

# ╔═╡ 6e6ac76e-10da-4997-8ed2-6001be559e68
# Estimate background
yslope = (yb[end]-yb[1])/(2*(t[end] - t[1]))*t

# ╔═╡ 381faf86-1382-4038-a5ae-98acae6d61e0
hw = hanning(length(t))

# ╔═╡ 1507cec8-0ab5-46ca-8f14-fb16a9f1b909
ybh = yb .* hw

# ╔═╡ d4a01db1-60e7-42c4-8f47-2dd7ff3c0010
ybmb = yb - yslope

# ╔═╡ cc00bd19-59cc-43e7-9bf6-e8660e2ea9d3
Yb = nfft(yb, t)

# ╔═╡ 3e06327d-422c-4777-8fcc-f4ecaf989e6c
Ybh = nfft(ybh, t)

# ╔═╡ 986e5c86-2655-4c8c-8eec-0e82fc563254
Ybmb = nfft((yb-yslope) .* hw, t)

# ╔═╡ 9f1a15c8-7120-4120-952f-8c9f1f821603
Yslope = nfft(yslope, t)

# ╔═╡ 33b1201c-4e28-4611-b106-f2898448c70f
let
    p1 = plot(t, yb, lw=2, label="Signal with background", xmirror=true, xaxis=L"t", yaxis=L"y(t)")
    plot!(p1, t, ybmb, lw=2, label="Signal with background removed")
    plot!(p1, t, ybh, lw=2, ls=:dash, label="Signal windowed only")
    plot!(p1, t, yslope, lw=2, ls=:dot, label="Linear slope")

    p2 = plot(ω, abs2.(Yb), lw=2, label="Signal with background", ylabel=L"|Y(\omega)|^2",
              xaxis=false)
    plot!(p2, ω, abs2.(Ybmb), lw=2, label="Background removed and windowed")
    plot!(p2, ω, abs2.(Ybh), lw=2, ls=:dash, label="Windowed only")
    plot!(p2, ω, abs2.(Yslope), lw=2, ls=:dot, label="Linear slope")

    p3 = plot(ω, abs2.(Yb), lw=2, label="Signal with background",
              xlabel=L"\omega",
              ylabel=L"|Y(\omega)|^2",
              yaxis=(:log10, (1e-8, 1e3)),
              legend=false)
    plot!(p3, ω, abs2.(Ybmb), lw=2, label="Background removed and windowed")
    plot!(p3, ω, abs2.(Ybh), lw=2, ls=:dash, label="Windowed only")
    plot!(p3, ω, abs2.(Yslope), lw=2, ls=:dot, label="Linear slope")

    plot(p1, p2, p3, layout=@layout([a;b;c]), size=(700,1000))
end


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
let
    p1 = plot(tderiv, yderiv, legend=false, xaxis=false, ylabel=L"y(t)", lw=2)

    p2 = plot(tderiv, real(y′deriv), label="Derivative via FFT", xaxis=false, ylabel=L"y'(t)", lw=2)
    plot!(p2, tderiv, y′deriv_exact, label="Exact derivative", ls=:dash, lw=2)

    p3 = let err = abs.(y′deriv-y′deriv_exact)
        plot(tderiv, err, label="FFT derivative error", lw=2,
             xlabel=L"t",
             yaxis = maximum(err) < 1e-2 ? :log10 : :identity)
    end

    plot(p1, p2, p3, layout=@layout([a;b;c]), size=(700,700))
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
let
    p1 = plot(tderiv, yderiv, legend=false, xaxis=false, ylabel=L"y(t)", lw=2)

    p2 = plot(tderiv, real(y′deriv), label="Derivative via FFT", xaxis=false, ylabel=L"y'(t)", lw=2)
    plot!(p2, tderiv, real(y′deriv_hann), label="Derivative via FFT + Hann", lw=2)
    plot!(p2, tderiv, y′deriv_exact, label="Exact derivative", ls=:dash, lw=2)

    p3 = let err = abs.(y′deriv-y′deriv_exact),
        err_hann = abs.(y′deriv_hann-y′deriv_exact)
        p3 = plot(tderiv, err, label="FFT derivative error", lw=2,
                  xlabel=L"t",
                  yaxis = maximum(err) < 1e-2 ? :log10 : :identity,
                  legend=:bottomright)
        plot!(p3, tderiv, err_hann, label="FFT derivative + Hann error", lw=2)
    end

    plot(p1, p2, p3, layout=@layout([a;b;c]), size=(700,700))
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
let
    p1 = plot(tderiv, yderiv, legend=false, xaxis=false, ylabel=L"y(t)", lw=2)

    p2 = plot(tderiv, real(y′′deriv), label="Second derivative via FFT", xaxis=false, ylabel=L"y''(t)", lw=2)
    plot!(p2, tderiv, y′′deriv_exact, label="Exact second derivative", ls=:dash, lw=2)

    p3 = let err = abs.(y′′deriv-y′′deriv_exact)
        plot(tderiv, err, label="FFT second derivative error", lw=2,
             xlabel=L"t",
             yaxis = maximum(err) < 1e-2 ? :log10 : :identity,
             legend=:bottomright)
    end

    plot(p1, p2, p3, layout=@layout([a;b;c]), size=(700,700))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
DSP = "~0.7.8"
FFTW = "~1.5.0"
LaTeXStrings = "~1.3.0"
Plots = "~1.38.5"
PlutoUI = "~0.7.49"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "8392c5a3ebc84fca5b3bce9396d862f42405b584"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "844b061c104c408b24537482469400af6075aae4"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "da8b06f89fce9996443010ef92572b193f8dca1f"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.8"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "9e23bd6bb3eb4300cb567bdf63e2c14e5d2ffdbc"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.5"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "aa23c9f9b7c0ba6baeabe966ea1c7d2c7487ef90"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.5+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "37e4657cd56b11abe3d10cd4a1ec5fbdb4180263"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.7.4"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "2422f47b34d4b127720a18f86fa7b1aa2e141f29"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.18"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "680e733c3a0a9cea9e935c8c2184aea6a63fa0b5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.21"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "6503b77492fd7fcb9379bf73cd31035670e3c509"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "18f84637e00b72ba6769034a4b50d79ee40c84a9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.5"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "8ac949bd0ebc46a44afb1fdca1094554a84b086e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "a14a99e430e42a105c898fcc7f212334bc7be887"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

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
# ╟─f6feb286-b126-4855-80ce-c0e7248c2f0e
# ╠═bdba5492-bea8-4268-a1ea-8b57f15fe27e
# ╠═7f74e7ae-961f-46b9-9704-8a3900e4e907
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
# ╟─105ee14a-40c6-41ef-b908-a2184c3a6109
# ╠═2ccd66c9-166a-486b-b546-205ccb84135d
# ╟─200db744-1c52-404d-ab28-b7b4ff470d2c
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
# ╟─ff6cb77c-c4fb-4e92-9260-37b5fe41043a
# ╠═c2e6e2e6-acf6-4258-b121-c5fec5764e08
# ╠═6e6ac76e-10da-4997-8ed2-6001be559e68
# ╠═381faf86-1382-4038-a5ae-98acae6d61e0
# ╠═1507cec8-0ab5-46ca-8f14-fb16a9f1b909
# ╠═d4a01db1-60e7-42c4-8f47-2dd7ff3c0010
# ╠═cc00bd19-59cc-43e7-9bf6-e8660e2ea9d3
# ╠═3e06327d-422c-4777-8fcc-f4ecaf989e6c
# ╠═986e5c86-2655-4c8c-8eec-0e82fc563254
# ╠═9f1a15c8-7120-4120-952f-8c9f1f821603
# ╟─33b1201c-4e28-4611-b106-f2898448c70f
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
# ╟─8c290ee4-2b0e-48a4-898c-d4444f246455
# ╠═5a58d0e6-5525-49c4-8bd2-4310c247ddbd
# ╠═da9115b5-85e1-4227-bd5a-b84355497552
# ╟─efde8cf9-f369-49a0-838c-af33d6d0102d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
