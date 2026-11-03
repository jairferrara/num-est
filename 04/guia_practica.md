# Guía de la práctica 04: del espacio de Fourier al cielo del CMB

- **Parte I:** armónicos esféricos con `scipy.special`.
- **Parte II:** CAMB, el código que da el espectro angular $C_\ell$.

> Entorno: conda `cosmo` (SciPy 1.17.1, CAMB 2.0.4). Todos los ejemplos se ejecutaron en ese entorno.

---
---

# Parte I — Armónicos esféricos (`scipy.special`)

## I.1 Definición

$$
Y_\ell^m(\vartheta,\varphi)=\sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m)!}{(\ell+m)!}}\;P_\ell^m(\cos\vartheta)\,e^{im\varphi},
\qquad \ell\ge0,\quad -\ell\le m\le\ell
$$

- $\vartheta\in[0,\pi]$ es el ángulo **polar** (colatitud) y $\varphi\in[0,2\pi)$ el ángulo **azimutal** (longitud).
- Incluye la fase de Condon–Shortley $(-1)^m$.
- Son ortonormales: $\displaystyle\int Y_\ell^m\,Y_{\ell'}^{m'*}\,d\Omega=\delta_{\ell\ell'}\delta_{mm'}$, con $d\Omega=\sin\vartheta\,d\vartheta\,d\varphi$.
- Cumplen la relación de conjugación $Y_\ell^{-m}=(-1)^m\,\big(Y_\ell^m\big)^*$.

## I.2 Funciones disponibles (SciPy ≥ 1.15)

| Función | Firma | Salida |
|---|---|---|
| `sph_harm_y` | `(n, m, theta, phi, *, diff_n=0)` | `complex ndarray`: $Y_n^m(\theta,\phi)$ |
| `sph_harm_y_all` | `(n, m, theta, phi, *, diff_n=0)` | `complex ndarray` de forma `(n+1, 2m+1, *theta.shape)` con todos los $Y_\ell^{m'}$ para $\ell\le n$ y $\lvert m'\rvert\le m$ |
| `sph_legendre_p` | `(n, m, theta, *, diff_n=0)` | Parte polar normalizada, $Y_n^m(\theta,0)$ (real) |

- **`n`** es el grado $\ell$ (entero, $\ge0$) y **`m`** el orden $m$ (entero).
- **`theta`** es el ángulo **polar** $\in[0,\pi]$ y **`phi`** el **azimutal** $\in[0,2\pi]$, ambos en radianes.
- Si $\lvert m\rvert>\ell$, la función devuelve 0.
- Todos los argumentos admiten *broadcasting* de NumPy: puedes pasar mallas 2D.
- En `sph_harm_y_all`, el índice `[l, m]` acepta $m$ negativo con la indexación negativa de Python: `Y[2, -1]` es $Y_2^{-1}$.

> ⚠️ **Convenciones de ángulos.**
> 1. El PDF de la práctica llama $\theta$ al ángulo **azimutal** y $\phi$ al **polar**, al revés que SciPy.
>    Con la notación del PDF, la llamada es `sph_harm_y(l, m, phi_pdf, theta_pdf)`.
> 2. La función antigua `scipy.special.sph_harm(m, n, theta, phi)` (eliminada en SciPy 1.17) invertía **también** el orden:
>    tomaba `m` antes que `n`, y `theta` era el azimutal. Si copias código viejo, revisa el orden de los argumentos.

## I.3 Malla angular y evaluación

```python
import numpy as np
from scipy.special import sph_harm_y

n_pol, n_az = 181, 361
pol = np.linspace(0, np.pi, n_pol)          # ángulo polar  ϑ ∈ [0, π]
az  = np.linspace(0, 2*np.pi, n_az)         # ángulo azimutal φ ∈ [0, 2π]
AZ, POL = np.meshgrid(az, pol)              # forma (n_pol, n_az)

Y = sph_harm_y(2, 1, POL, AZ)               # Y_2^1 en toda la malla (complejo)
print(Y.shape, Y.dtype)                     # (181, 361) complex128
print(sph_harm_y(0, 0, 0.3, 1.2))           # 1/(2√π) ≈ 0.2821
```

## I.4 Visualizar modos individuales

Como $Y_\ell^m$ es complejo, se grafica su parte real. El primer ejemplo usa la proyección rectangular $(\varphi,\vartheta)$:

```python
import matplotlib.pyplot as plt

modos = [(1, 0), (2, 0), (2, 1), (5, 2)]
fig, axs = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
for ax, (l, m) in zip(axs.flat, modos):
    im = ax.pcolormesh(AZ, POL, sph_harm_y(l, m, POL, AZ).real,
                       cmap='RdBu_r', shading='auto')
    ax.set_title(rf'$\mathrm{{Re}}\,Y_{{{l}}}^{{{m}}}$', fontsize=16)
    ax.set_xlabel(r'$\varphi$ (azimutal)'); ax.set_ylabel(r'$\vartheta$ (polar)')
    ax.invert_yaxis()                       # polo norte (ϑ=0) arriba
    fig.colorbar(im, ax=ax)
plt.show()
```

Para la proyección Mollweide, la habitual en mapas del CMB, matplotlib espera longitud $\in[-\pi,\pi]$ y latitud $\in[-\pi/2,\pi/2]$:

```python
lon = AZ - np.pi                            # φ → longitud
lat = np.pi/2 - POL                         # ϑ → latitud

fig = plt.figure(figsize=(8, 4.5))
ax = fig.add_subplot(projection='mollweide')
im = ax.pcolormesh(lon, lat, sph_harm_y(5, 2, POL, AZ).real, cmap='RdBu_r', shading='auto')
ax.set_xticklabels([]); ax.grid(alpha=0.3)
fig.colorbar(im, orientation='horizontal', pad=0.05, label=r'$\mathrm{Re}\,Y_5^2$')
plt.show()
```

## I.5 Comprobar la ortonormalidad numéricamente

Este ejemplo es útil para verificar que estás usando bien la convención de ángulos:

```python
dpol, daz = pol[1] - pol[0], az[1] - az[0]
dOmega = np.sin(POL) * dpol * daz           # elemento de ángulo sólido

def producto(l1, m1, l2, m2):
    return np.sum(sph_harm_y(l1, m1, POL, AZ) *
                  np.conj(sph_harm_y(l2, m2, POL, AZ)) * dOmega)

print(abs(producto(3, 2, 3, 2)))   # ≈ 1
print(abs(producto(3, 2, 4, 2)))   # ≈ 0
```

## I.6 Síntesis de un campo a partir de coeficientes $a_{\ell m}$

La expansión $f(\vartheta,\varphi)=\sum_{\ell}\sum_{m=-\ell}^{\ell}a_{\ell m}Y_\ell^m(\vartheta,\varphi)$ se implementa directamente con un doble lazo:

```python
lmax = 20
alm = {(l, m): 0j for l in range(1, lmax + 1) for m in range(-l, l + 1)}
alm[(3, 0)] = 1.0                           # ejemplo: un solo modo

f = np.zeros_like(POL, dtype=complex)
for (l, m), a in alm.items():
    f += a * sph_harm_y(l, m, POL, AZ)

print(np.abs(f.imag).max())                 # si los a_lm cumplen la condición de realidad, ≈ 0
```

Con `sph_harm_y_all` se evalúan todos los modos de una vez, a cambio de más memoria (forma `(lmax+1, 2·lmax+1, n_pol, n_az)`):

```python
from scipy.special import sph_harm_y_all
Yall = sph_harm_y_all(lmax, lmax, POL, AZ)
print(Yall.shape)                            # (21, 41, 181, 361)
print(np.allclose(Yall[5, -2], sph_harm_y(5, -2, POL, AZ)))   # True
```

> Cuántos coeficientes $a_{\ell m}$ generar, con qué varianza y cómo imponer que el campo sea real es justamente lo que pide resolver la práctica (secciones 0.4 y siguientes del PDF).

## I.7 Escala angular de un multipolo

Un modo de multipolo $\ell$ varía en escalas angulares de $\theta\sim\pi/\ell$ rad $\approx 180^\circ/\ell$. Por ejemplo, $\ell\approx200$ corresponde a $\sim1^\circ$.

---
---

# Parte II — CAMB

CAMB (*Code for Anisotropies in the Microwave Background*) calcula espectros angulares del CMB ($C_\ell$), espectros de potencia de materia $P(k)$ y cantidades de fondo cosmológico.

> Versión de referencia: **CAMB 2.0.4**, en el entorno conda `cosmo`. Todos los ejemplos se ejecutaron con esa versión.

---

## 0. Instalación e importación

```bash
conda activate cosmo          # o bien: pip install camb
```

```python
import numpy as np
import camb
from camb import model, initialpower
print(camb.__version__)
```

---

## 1. Flujo general

```
CAMBparams  ──(configurar)──►  camb.get_results(pars)  ──►  CAMBdata  ──►  C_ℓ, P(k), distancias, σ8 …
```

1. **Crear** un objeto `CAMBparams` (constructor o `camb.set_params`).
2. **Configurar** la cosmología, el espectro primordial y la precisión.
3. **Resolver** con `camb.get_results(pars)` (o con `get_background` si solo necesitas el fondo).
4. **Extraer** los resultados con los métodos de `CAMBdata`.

---

## 2. Constructores de parámetros

### 2.1 `camb.CAMBparams()`, configuración paso a paso

```python
pars = camb.CAMBparams()                       # constructor vacío (valores por defecto)
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0)
```

### 2.2 `camb.set_params(...)`, todo en una sola llamada

```python
camb.set_params(cp=None, verbose=False, **params)  ->  CAMBparams
```

Reparte los argumentos con nombre (*kwargs*) entre los *setters* correspondientes (`set_cosmology`, `InitPower.set_params`, `set_for_lmax`, `set_dark_energy`, …):

```python
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, As=2e-9, ns=0.965,
                       tau=0.06, lmax=2500, WantTransfer=True)
```

También acepta atributos directos de `CAMBparams` (por ejemplo `WantTransfer`, `WantTensors` o `DoLensing`).

---

## 3. Métodos de configuración de `CAMBparams`

### 3.1 `set_cosmology`, parámetros de fondo

```python
pars.set_cosmology(*, H0=None, ombh2=0.022, omch2=0.12, omk=0.0,
                   cosmomc_theta=None, thetastar=None,
                   neutrino_hierarchy='degenerate', num_massive_neutrinos=1,
                   mnu=0.06, nnu=3.044, YHe=None, TCMB=2.7255,
                   tau=None, zrei=None, Alens=1.0, ...)
```

| Argumento | Significado |
|---|---|
| `H0` | Constante de Hubble [km/s/Mpc] |
| `ombh2`, `omch2` | $\Omega_b h^2$, $\Omega_c h^2$ |
| `omk` | Curvatura $\Omega_k$ |
| `cosmomc_theta` | $\theta_{MC}$ (p. ej. `0.0104092`); sustituye a `H0` (pasar `H0=None`) |
| `mnu`, `nnu` | Suma de masas de neutrinos [eV], $N_{\rm eff}$ |
| `tau` / `zrei` | Profundidad óptica de reionización o el redshift de reionización |
| `Alens` | Amplitud del lensing (1 = física) |

Todos los argumentos son de solo palabra clave (después de `*`).

### 3.2 `InitPower.set_params`, espectro primordial

```python
pars.InitPower.set_params(As=2e-09, ns=0.96, nrun=0, r=0.0, nt=None,
                          pivot_scalar=0.05, pivot_tensor=0.05)
```

Corresponde a $P_{\mathcal R}(k) = A_s (k/k_*)^{n_s-1}$. Si `r > 0`, activa también `pars.WantTensors = True`.

### 3.3 `set_for_lmax`, rango en ℓ y precisión

```python
pars.set_for_lmax(lmax, max_eta_k=None, lens_potential_accuracy=None, ...)
```

- `lmax`: el $\ell$ máximo que quieres usar. Internamente calcula un poco más para el lensing.
- `lens_potential_accuracy`: 0 es rápido; 1 o más es más preciso para el lensing.

### 3.4 `set_matter_power`, $P(k)$

```python
pars.set_matter_power(redshifts=(0.0,), kmax=1.2, k_per_logint=None,
                      nonlinear=None, ...)
```

Activa el cálculo de las funciones de transferencia (`WantTransfer=True`). Los `redshifts` se ordenan internamente de mayor a menor.

### 3.5 `set_dark_energy`, ecuación de estado

```python
pars.set_dark_energy(w=-1.0, cs2=1.0, wa=0, dark_energy_model='fluid')
# para wa != 0 usa dark_energy_model='ppf'
```

### 3.6 Atributos útiles

```python
pars.WantTensors = True                   # modos tensoriales (necesario si r > 0)
pars.NonLinear = model.NonLinear_both     # NonLinear_none | _pk | _lens | _both
pars.DoLensing = True
pars.Accuracy.AccuracyBoost = 1           # >1 para mayor precisión
print(pars)                               # imprime todos los parámetros
```

### 3.7 Tabla completa de argumentos de `camb.set_params`

`camb.set_params(**params)` reparte cada argumento con nombre entre los *setters* siguientes, en este orden. Cada fila indica el argumento, su símbolo, sus unidades y su valor por defecto. Puedes usar cualquiera de estos nombres directamente en `set_params`, o en el método correspondiente de `CAMBparams`.

**a) Precisión — `pars.set_accuracy`**

| Argumento | Símbolo | Unidades | Defecto | Descripción |
|---|---|---|---|---|
| `AccuracyBoost` | — | adim. | `1.0` | Aumenta la precisión global (pasos en tiempo y en $k$) |
| `lSampleBoost` | — | adim. | `1.0` | Densidad del muestreo en $\ell$ |
| `lAccuracyBoost` | — | adim. | `1.0` | Truncamiento de la jerarquía de Boltzmann en $\ell$ |
| `DoLateRadTruncation` | — | bool | `True` | Aproximación de radiación a tiempos tardíos |
| `min_l_logl_sampling` | — | adim. | `None` | $\ell$ a partir del cual el muestreo es logarítmico |

**b) Clases de modelos — `pars.set_classes`**

| Argumento | Defecto | Opciones típicas |
|---|---|---|
| `dark_energy_model` | `'fluid'` | `'fluid'`, `'ppf'`, `'DarkEnergyPPF'`, `'AxionEffectiveFluid'`, … |
| `initial_power_model` | `InitialPowerLaw` | `'InitialPowerLaw'`, `'SplinedInitialPower'` |
| `non_linear_model` | `Halofit` | `'Halofit'`, `'SecondOrderPK'` |
| `recombination_model` | `Recfast` | `'Recfast'` (CosmoRec y HyRec requieren compilar CAMB con esas librerías) |
| `reionization_model` | `TanhReionization` | `'TanhReionization'`, `'ExpReionization'` |

**c) Recombinación — `pars.Recomb.set_params`**

| Argumento | Defecto | Descripción |
|---|---|---|
| `recfast_approx_model` | `None` | Variante de RECFAST que se usa |

**d) Energía oscura — `pars.DarkEnergy.set_params`**

| Argumento | Símbolo | Unidades | Defecto | Descripción |
|---|---|---|---|---|
| `w` | $w_0$ | adim. | `-1.0` | Ecuación de estado hoy: $w(a)=w_0+w_a(1-a)$ |
| `wa` | $w_a$ | adim. | `0` | Evolución de $w$ (con $w_a\neq0$, usa `'ppf'`) |
| `cs2` | $c_s^2$ | $c^2$ | `1.0` | Velocidad del sonido en el sistema en reposo del fluido |
| `use_tabulated_w` | — | bool | `False` | Usa una tabla $w(a)$ |
| `wde_a_array`, `wde_w_array` | $a$, $w(a)$ | adim. | `None` | Tabla de $w(a)$ |

**e) Reionización — `pars.Reion.set_extra_params`**

| Argumento | Símbolo | Unidades | Defecto | Descripción |
|---|---|---|---|---|
| `deltazrei` | $\Delta z_{\rm re}$ | adim. | `0.5` | Ancho de la transición de reionización (tanh) |
| `max_zrei` | $z_{\rm re}^{\max}$ | adim. | `None` | Redshift máximo de reionización |

**f) Cosmología de fondo — `pars.set_cosmology`**

| Argumento | Símbolo | Unidades | Defecto | Descripción |
|---|---|---|---|---|
| `H0` | $H_0$ | km s⁻¹ Mpc⁻¹ | `None` | Constante de Hubble ($h=H_0/100$) |
| `ombh2` | $\Omega_b h^2$ | adim. | `0.022` | Densidad física de bariones |
| `omch2` | $\Omega_c h^2$ | adim. | `0.12` | Densidad física de materia oscura fría |
| `omk` | $\Omega_K$ | adim. | `0.0` | Curvatura |
| `cosmomc_theta` | $\theta_{\rm MC}$ | rad | `None` | Tamaño angular aproximado del horizonte de sonido (~0.0104); sustituye a `H0` |
| `thetastar` | $\theta_*$ | rad | `None` | Tamaño angular exacto del horizonte de sonido; sustituye a `H0` |
| `neutrino_hierarchy` | — | str | `'degenerate'` | `'degenerate'`, `'normal'` o `'inverted'` |
| `num_massive_neutrinos` | $N_{\nu,m}$ | adim. | `1` | Número de neutrinos masivos |
| `mnu` | $\sum m_\nu$ | eV | `0.06` | Suma de masas de los neutrinos |
| `omnuh2_active` | $\Omega_\nu h^2$ | adim. | `None` | Densidad de neutrinos activos; alternativa a `mnu` |
| `nnu` | $N_{\rm eff}$ | adim. | `3.044` | Número efectivo de especies relativistas |
| `YHe` | $Y_P$ | adim. | `None` | Fracción de masa del helio; con `None` se obtiene de BBN |
| `meffsterile` | $m_{\rm eff}^{\rm st}$ | eV | `0.0` | Masa efectiva de un neutrino estéril |
| `standard_neutrino_neff` | $N_{\rm eff}^{\rm std}$ | adim. | `3.044` | $N_{\rm eff}$ del modelo estándar |
| `TCMB` | $T_{\rm CMB}$ | K | `2.7255` | Temperatura actual del CMB |
| `tau` | $\tau$ | adim. | `None` | Profundidad óptica de reionización |
| `zrei` | $z_{\rm re}$ | adim. | `None` | Redshift de reionización; alternativa a `tau` |
| `Alens` | $A_L$ | adim. | `1.0` | Escala del potencial de lensing |
| `bbn_predictor` | — | str / objeto | `None` | Tabla de BBN para calcular $Y_P$ |
| `theta_H0_range` | — | km s⁻¹ Mpc⁻¹ | `(10, 100)` | Rango de búsqueda de $H_0$ cuando se fija $\theta$ |
| `setter_H0` | — | función | `None` | Función propia para fijar $H_0$ |

**g) Espectro de materia — `pars.set_matter_power`**

| Argumento | Símbolo | Unidades | Defecto | Descripción |
|---|---|---|---|---|
| `redshifts` | $z_i$ | adim. | `(0.0,)` | Redshifts en los que se calcula $P(k)$ (máx. 150) |
| `kmax` | $k_{\max}$ | Mpc⁻¹ | `1.2` | $k$ máximo; **sin** factor $h$ |
| `k_per_logint` | — | adim. | `None` | Puntos en $k$ por intervalo logarítmico |
| `nonlinear` | — | bool | `None` | Activa el $P(k)$ no lineal (Halofit/HMcode) |
| `accurate_massive_neutrino_transfers` | — | bool | `False` | Transferencias precisas para neutrinos masivos |
| `silent` | — | bool | `False` | Suprime los avisos |

**h) Rango en ℓ — `pars.set_for_lmax`**

| Argumento | Símbolo | Unidades | Defecto | Descripción |
|---|---|---|---|---|
| `lmax` | $\ell_{\max}$ | adim. | — | $\ell$ máximo de salida |
| `max_eta_k` | $(k\eta_0)_{\max}$ | adim. | `None` | $k\,\eta_0$ máximo de integración; con `None`, `k_eta_fac·lmax` |
| `lens_potential_accuracy` | — | adim. | `None` | 0 = rápido; 1 o más = lensing preciso (CMB-S4: 4 u 8) |
| `lens_output_margin` | $\Delta\ell$ | adim. | `200` | Margen extra de $\ell$ para el lensing |
| `k_eta_fac` | — | adim. | `2.5` | Factor $k\eta_0/\ell_{\max}$ |
| `lens_k_eta_reference` | — | adim. | `18000.0` | $k\eta_0$ de referencia para el lensing |
| `nonlinear` | — | bool | `None` | Lensing no lineal |

**i) Espectro primordial — `pars.InitPower.set_params`** (ley de potencias)

| Argumento | Símbolo | Unidades | Defecto | Descripción |
|---|---|---|---|---|
| `As` | $A_s$ | adim. | `2e-9` | Amplitud escalar en $k_*$ |
| `ns` | $n_s$ | adim. | `0.96` | Índice espectral escalar |
| `nrun` | $\alpha_s=dn_s/d\ln k$ | adim. | `0` | *Running* del índice escalar |
| `nrunrun` | $\beta_s=d\alpha_s/d\ln k$ | adim. | `0.0` | *Running* del *running* |
| `r` | $r$ | adim. | `0.0` | Razón tensor/escalar en `pivot_tensor` |
| `nt` | $n_t$ | adim. | `None` | Índice tensorial; con `None` usa la relación de consistencia $n_t=-r/8$ |
| `ntrun` | $\alpha_t$ | adim. | `0.0` | *Running* del índice tensorial |
| `pivot_scalar` | $k_*$ | Mpc⁻¹ | `0.05` | Escala pivote escalar |
| `pivot_tensor` | $k_{*,t}$ | Mpc⁻¹ | `0.05` | Escala pivote tensorial |
| `parameterization` | — | str | `'tensor_param_rpivot'` | Forma de especificar la amplitud tensorial |

$$P_{\mathcal R}(k)=A_s\left(\frac{k}{k_*}\right)^{n_s-1+\frac{1}{2}\alpha_s\ln(k/k_*)+\frac{1}{6}\beta_s\ln^2(k/k_*)},\qquad P_t(k)=r\,A_s\left(\frac{k}{k_{*,t}}\right)^{n_t+\dots}$$

**j) No linealidad — `pars.NonLinearModel.set_params`** (Halofit/HMcode)

| Argumento | Símbolo | Unidades | Defecto | Descripción |
|---|---|---|---|---|
| `halofit_version` | — | str | `'mead2020'` | `'original'`, `'takahashi'`, `'mead'`, `'mead2020'`, `'mead2020_feedback'`, … |
| `HMCode_A_baryon` | $A_{\rm bary}$ | adim. | `3.13` | Amplitud de la concentración del halo (HMcode-2016) |
| `HMCode_eta_baryon` | $\eta_{\rm bary}$ | adim. | `0.603` | Parámetro de *bloating* del halo (HMcode-2016) |
| `HMCode_logT_AGN` | $\log_{10}(T_{\rm AGN}/{\rm K})$ | adim. | `7.8` | Retroalimentación de AGN (`mead2020_feedback`) |
| `HMCode_wiggle_max_fnu` | — | adim. | `0.01` | Fracción máxima de neutrinos para el tratamiento de las BAO |

**k) Atributos directos de `CAMBparams`** (también aceptados por `set_params`, p. ej. `WantTransfer=True`; para atributos anidados: `**{'Accuracy.AccuracyBoost': 2}`)

| Atributo | Tipo | Defecto | Descripción |
|---|---|---|---|
| `WantCls` | bool | `True` | Calcula los $C_\ell$ del CMB |
| `WantScalars` / `WantTensors` / `WantVectors` | bool | `True` / `False` / `False` | Modos que se calculan |
| `WantTransfer` | bool | `False` | Calcula las funciones de transferencia y $P(k)$ |
| `WantDerivedParameters` | bool | `True` | Calcula $z_*$, $r_{\rm drag}$, edad, … |
| `DoLensing` | bool | `True` | Aplica el lensing a los $C_\ell$ |
| `Want_CMB`, `Want_CMB_lensing` | bool | `True` | Calcula los $C_\ell$ del CMB y del potencial de lensing |
| `NonLinear` | enum | `NonLinear_none` | `model.NonLinear_none`, `_pk`, `_lens` o `_both` |
| `min_l` | int | `2` | $\ell$ mínimo |
| `max_l`, `max_l_tensor` | int | — | $\ell$ máximos internos (los fija `set_for_lmax`) |
| `OutputNormalization` | int | `1` | Normalización de salida |
| `Transfer.high_precision` | bool | `False` | Transferencias de alta precisión |
| `Transfer.kmax` | float [Mpc⁻¹] | — | $k_{\max}$ de las transferencias |
| `Accuracy.AccuracyBoost`, `Accuracy.lAccuracyBoost`, … | float | `1` | Precisión (véase a) |
| `SourceTerms.limber_windows` | bool | `True` | Aproximación de Limber para ventanas |
| `Do21cm`, `Evolve_delta_xe`, … | bool | `False` | Física adicional (21 cm, perturbaciones de $x_e$) |

Atributos derivados, de solo lectura: `h`, `omegab`, `omegac`, `omegam`, `omeganu`, `N_eff`.

```python
# Ejemplo: usar varias categorías a la vez y ver los setters equivalentes
pars = camb.set_params(H0=67.4, ombh2=0.0224, omch2=0.120, mnu=0.06, tau=0.054,
                       As=2.1e-9, ns=0.965, r=0.01, w=-0.9, deltazrei=0.5,
                       lmax=2500, lens_potential_accuracy=1, AccuracyBoost=1.2,
                       halofit_version='mead2020', WantTransfer=True, verbose=True)
print(pars.h, pars.omegam, pars.N_eff, pars.WantTensors)
```

---

## 4. Resolver: funciones de nivel superior

| Función | Firma | Devuelve | Úsala para |
|---|---|---|---|
| `camb.get_results` | `(params)` | `CAMBdata` | Todo ($C_\ell$, $P(k)$, fondo) |
| `camb.get_background` | `(params, no_thermo=False)` | `CAMBdata` | Solo distancias y $H(z)$ (muy rápido) |
| `camb.get_transfer_functions` | `(params, only_time_sources=False)` | `CAMBdata` | Funciones de transferencia; los $C_\ell$ se calculan después |
| `camb.get_matter_power_interpolator` | `(params, zmin=0, zmax=10, kmax=10, nonlinear=True, hubble_units=True, k_hunit=True, ...)` | Interpolador `PK` | $P(k,z)$ en cualquier punto |

```python
results = camb.get_results(pars)
```

---

## 5. Salidas del CMB (métodos de `CAMBdata`)

### 5.1 Todos los espectros a la vez

```python
results.get_cmb_power_spectra(params=None, lmax=None,
        spectra=('total','unlensed_scalar','unlensed_total',
                 'lensed_scalar','tensor','lens_potential'),
        CMB_unit=None, raw_cl=False)  ->  dict[str, ndarray]
```

**Formato de salida:** cada entrada es un `ndarray` de forma `(lmax+1, 4)` con columnas **TT, EE, BB, TE**. El índice de la fila es $\ell$, así que las filas 0 y 1 valen cero.

- `CMB_unit='muK'` da $\mu K^2$; con `None` el resultado es adimensional ($\Delta T/T$).
- `raw_cl=False` devuelve $D_\ell = \ell(\ell+1)C_\ell/2\pi$; con `raw_cl=True` devuelve $C_\ell$.
- `'lens_potential'` tiene las columnas $\phi\phi$, $T\phi$ y $E\phi$, con la normalización $[\ell(\ell+1)]^2 C_\ell^{\phi\phi}/2\pi$.

### 5.2 Métodos individuales (misma firma `(lmax=None, CMB_unit=None, raw_cl=False)`)

```python
results.get_total_cls(...)              # lensado + tensores
results.get_lensed_scalar_cls(...)
results.get_unlensed_scalar_cls(...)
results.get_lens_potential_cls(...)
```

### 5.3 Ejemplo: espectro TT

```python
import numpy as np, camb
import matplotlib.pyplot as plt

pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, As=2e-9, ns=0.965,
                       tau=0.06, lmax=2000)
results = camb.get_results(pars)
powers  = results.get_cmb_power_spectra(pars, CMB_unit='muK')

Dl = powers['total']                    # (lmax+1, 4): TT, EE, BB, TE
ell = np.arange(Dl.shape[0])
plt.plot(ell[2:], Dl[2:, 0])
plt.xlabel(r'$\ell$'); plt.ylabel(r'$D_\ell^{TT}\ [\mu K^2]$')
plt.show()

# C_ell "crudo" (sin ℓ(ℓ+1)/2π), útil para generar mapas gaussianos
Cl = results.get_total_cls(lmax=2000, CMB_unit='muK', raw_cl=True)[:, 0]
np.savetxt('Cl_TT.txt', np.c_[ell[:2001], Cl], header='l  C_l (muK^2)')
```

---

## 6. Espectro de potencia de materia

### 6.1 En una malla de $k$

```python
results.get_matter_power_spectrum(minkh=1e-4, maxkh=1.0, npoints=100,
                                  var1=None, var2=None)
    ->  (kh, z, pk)       # kh: (npoints,)  z: (nz,)  pk: (nz, npoints)
```

Las unidades son $k$ en $h\,\mathrm{Mpc}^{-1}$ y $P$ en $(\mathrm{Mpc}/h)^3$.

```python
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, As=2e-9, ns=0.965)
pars.set_matter_power(redshifts=[0., 1.], kmax=2.0)
pars.NonLinear = camb.model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)
print(z, pk.shape)          # [0.0, 1.0] (2, 200)
print(results.get_sigma8()) # σ8 en cada z (ordenado de z alto a z bajo)
print(results.get_sigma8_0())  # σ8 a z=0
```

### 6.2 Interpolador $P(k,z)$

```python
PK = camb.get_matter_power_interpolator(pars, zmax=2, kmax=10, nonlinear=False)
PK.P(0.5, 0.1)                       # P(z=0.5, k=0.1 h/Mpc)
PK.P(np.array([0, 1]), np.logspace(-3, 0, 50))   # salida con forma (2, 50)
```

---

## 7. Cosmología de fondo

Usa `get_background`, que es mucho más rápido que `get_results` cuando solo necesitas el fondo.

| Método | Entrada | Salida (unidades) |
|---|---|---|
| `hubble_parameter(z)` | escalar o array | $H(z)$ [km/s/Mpc] |
| `comoving_radial_distance(z)` | escalar o array | $\chi(z)$ [Mpc] |
| `angular_diameter_distance(z)` | escalar o array | $D_A(z)$ [Mpc] |
| `luminosity_distance(z)` | escalar o array | $D_L(z)$ [Mpc] |
| `redshift_at_comoving_radial_distance(chi)` | Mpc | $z$ |
| `conformal_time(z)` / `physical_time(z)` | $z$ | $\eta$ [Mpc] / $t$ [Gyr] |
| `get_Omega(var, z=0)` | `'baryon'`, `'cdm'`, `'de'`, `'photon'`, `'neutrino'`, `'nu'`, `'K'` | $\Omega_X(z)$ |
| `get_derived_params()` | — | `dict` (`age`, `zstar`, `rstar`, `thetastar`, `zdrag`, `rdrag`, …) |

```python
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122)
bg = camb.get_background(pars)
z = np.array([0.5, 1.0, 2.0])
print(bg.hubble_parameter(z))
print(bg.angular_diameter_distance(z))
print(bg.get_Omega('de', z=0))

res = camb.get_results(pars)            # get_derived_params necesita la termodinámica
d = res.get_derived_params()
print(d['age'], d['zstar'], d['rdrag'])
```

---

## 8. Variar un parámetro sin rehacer todo

Para recalcular los $C_\ell$ cambiando solo el espectro primordial, reutiliza las funciones de transferencia:

```python
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, As=2e-9, ns=0.965, lmax=2000)
data = camb.get_transfer_functions(pars)      # la parte costosa, se hace una vez

for ns in [0.94, 0.965, 0.99]:
    pars.InitPower.set_params(As=2e-9, ns=ns)
    data.power_spectra_from_transfer(pars.InitPower)
    Dl = data.get_cmb_power_spectra(CMB_unit='muK')['total'][:, 0]
    print(ns, Dl[220])
```

Si el lensing no lineal está activo (es el valor por defecto), CAMB avisa que no recalcula esa corrección; para comparaciones precisas usa `pars.NonLinear = model.NonLinear_none`.

Si cambias parámetros de fondo (`H0`, `omch2`, …), en cambio, debes volver a llamar a `camb.get_results`.

---

## 9. Tensores ($r$) y modos B

```python
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, As=2e-9, ns=0.965,
                       r=0.1, WantTensors=True, lmax=1000)
results = camb.get_results(pars)
p = results.get_cmb_power_spectra(pars, CMB_unit='muK')
print(p['tensor'][80, 2], p['lensed_scalar'][80, 2])   # BB primordial vs. BB por lensing en ℓ=80
```

---

## 10. Errores y detalles comunes

- **Índice = ℓ:** `Dl[ell, col]`, sin desplazamiento. Ignora las filas 0 y 1.
- **$D_\ell$ frente a $C_\ell$:** para simular mapas gaussianos necesitas `raw_cl=True`.
- **Unidades:** sin `CMB_unit='muK'`, los valores son de orden $10^{-10}$.
- **`H0` y `cosmomc_theta`** se excluyen mutuamente: fija uno y deja el otro en `None`.
- **Tensores:** `r > 0` sin `WantTensors=True` no produce BB primordial.
- **$P(k)$:** llama a `set_matter_power` *antes* de `get_results`.
- **Redshifts:** CAMB devuelve `z` y `get_sigma8()` en el orden con que almacena los redshifts internamente; revisa siempre el array `z` que te devuelve.
