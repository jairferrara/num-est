# Plan del proyecto: covarianza de los estimadores de ξ(r) en un modelo de dilución de materia oscura

**Objetivo:** comparar la covarianza de ξ̂(r) obtenida con distintos estimadores (Peebles–Hauser, Davis–Peebles, Hamilton y Landy–Szalay), en ΛCDM y con dilución.

**Alcance:** solo materia, orden lineal y espacio real. Solo se usa D₊. No entran D_Ψ, D_𝓜, velocidades, RSD, términos no lineales ni la covarianza analítica.

**Base teórica:** A. Avilés & J. Ferrara, *Dark Matter Dilution* (borrador).

**Fechas:** reunión de avance el 25 de noviembre, presentación final el 2 de diciembre y entrega el 4 de diciembre.

---

## 0. Idea general

```
toy model g(z) ──► crecimiento D₊ ──► P_dil(k) ──► campos gaussianos ──► estimadores ξ̂(r) ──► covarianza
                        ▲                ▲                  ▲__________________│
                        └──── CLASS ─────┘                     N realizaciones
                       H(z), Ω_m(z)    P_L(k)
```

- **Continuidad con fuente:** ρ̇ + 3Hρ + (1/a)∇·(ρ**v**) = S. Restando el fondo:
  δ̇ + (1/a)∇·[(1+δ)**v**] = [S − (1+δ)S̄]/ρ̄.
  - S = Γρ (tasa de interacción, como en energía oscura interactuante): el lado derecho es 0. Solo cambia el fondo.
  - S = −gHρ̄δ con S̄ = 0 (toy model): el lado derecho es −gHδ. El fondo ΛCDM queda intacto y se diluyen solo las perturbaciones.
- **Toy model:** g(z) = (g₀/2)[1 − tanh((z − z_t)/Δz)], con g₀ = 0.2, z_t = 2, Δz = 0.2 y Ω_m0 = 0.3.
- **Crecimiento lineal:**
  D̈₊ + (2+g)HḊ₊ − [ (3/2)Ω_m H² − g(2H² + Ḣ) − ġH ] D₊ = 0.
- **Espectro:** P_dil(k,z) = [D₊(z)/D₊^ΛCDM(z)]² P_L(k,z). La dilución solo cambia la amplitud: la forma y el BAO quedan iguales.
- **Por qué hacen falta realizaciones:** con P_dil se conoce el ξ teórico, pero no la dispersión de cada estimador. Esa dispersión viene de la varianza cósmica de cada caja, de los aleatorios finitos y de la discretización, y solo aparece al medir en muchas realizaciones.

---

## 1. Parámetros

```python
# cosmología (ΛCDM plano, igual para ambos modelos)
Om0, Ob0, h, ns, sigma8 = 0.3, 0.045, 0.7, 0.965, 0.8

# toy model
g0_list = [0.0, 0.2]          # 0 = ΛCDM
zt, dz  = 2.0, 0.2
z_obs   = 0.5

# caja
L, Nmesh = 1000.0, 128        # Mpc/h, malla 128³
r_edges  = arange(20, 155, 5) # Mpc/h (incluye el BAO ~ 105)

# realizaciones y catálogos
N       = 20 000–50 000       # realizaciones por modelo, en Slurm (ver sección 10)
nbar    = 5e-5                # (h/Mpc)³ → N_D ≈ 5e4 puntos por caja
alphas  = [1, 3]              # N_R = α N_D
# la misma semilla s para g0 = 0 y g0 = 0.2 (reduce el ruido en la comparación)
```

---

## 2. Paso 1: crecimiento lineal con CLASS

```python
cosmo = CLASS(Om0, Ob0, h, ns, sigma8)      # fondo y P_L(k, z=0)
H(z), Om(z)                                  # del fondo de CLASS (con radiación y neutrinos)

def crecimiento(g0):
    # integrar la ecuación de D₊ en ln a, desde z = 99 (donde g = 0) hasta z = 0
    # condición inicial: D y f del factor de crecimiento de CLASS en z = 99
    return D₊(z)

D_LCDM = crecimiento(0.0)
D_dil  = crecimiento(0.2)
P_dil  = (D_dil(z_obs) / D_LCDM(z_obs))**2 * P_L(k, z_obs)
```

**Validaciones**
- g₀ = 0 reproduce el D₊ de CLASS (error relativo ≲ 1e-8).
- P_dil/P_ΛCDM ≈ 0.80 en z = 0.5.

---

## 3. Paso 2: campos gaussianos y catálogos

```python
def campo_gaussiano(P, rng):
    w = rng.standard_normal((Nmesh,)*3)             # ruido blanco REAL en la malla: δ(−k) = δ*(k) sale sola
    delta_k = rfftn(w) * sqrt(P(|k|) / V_celda)     # δ(k) = sqrt(P/V_celda) · FFT[w]
    return irfftn(delta_k)                          # δ(x) en la malla

def puntos(delta, rng):
    # densidad positiva (lognormal): 1 + δ → exp(δ − σ²/2)  (el δ gaussiano puede ser < −1)
    # conteo por celda ~ Poisson[nbar · V_celda · exp(δ − σ²/2)]; posiciones uniformes dentro de la celda
    return posiciones                               # catálogo D, periódico

def aleatorios(alpha, nbar):
    return uniformes_en_la_caja(alpha * nbar * V)   # catálogo R (fijo para todas las realizaciones)
```

**Validaciones**
- ⟨|δ(k)|²⟩ promediado en realizaciones reproduce P_dil(k).
- El número medio de puntos es n̄V.

---

## 4. Paso 3: estimadores de ξ(r)

Conteo de pares con `scipy.spatial.cKDTree(..., boxsize=L)` (caja periódica). DD, DR y RR se normalizan por el número de pares.

```python
PH  = DD/RR - 1
DP  = DD/DR - 1
Ham = DD*RR/DR**2 - 1
LS  = (DD - 2*DR + RR)/RR
```

RR y DR dependen de los aleatorios; por eso se varía α.

**Validación:** el promedio de cada estimador sobre las realizaciones reproduce el ξ(r) teórico, ξ(r) = ∫ k²dk/(2π²) P_dil(k) j₀(kr).

---

## 5. Paso 4: covarianza muestral

```python
for g0 in g0_list:
    for est in [PH, DP, Ham, LS]:
        X = xi[g0, est, :, :]                       # (N, n_bins)
        C[g0, est] = cov(X, rowvar=False)           # 1/(N−1) Σ (ξ_s − ξ̄)(ξ_s − ξ̄)ᵀ
        R[g0, est] = C / sqrt(outer(diag(C), diag(C)))
```

Error relativo de cada elemento: ~ sqrt(2/N).

---

## 6. Paso 5: comparaciones

1. **Entre estimadores:** Var[ξ̂_est(r)]/Var[ξ̂_LS(r)] contra r, para α = 1 y α = 3.
2. **Entre modelos:** C_dil/C_ΛCDM (diagonal) y las matrices de correlación lado a lado.
3. **Qué se espera:**
   - ξ_dil ≈ 0.8 ξ_ΛCDM, con la misma forma.
   - La parte de varianza cósmica de C escala como P² ≈ 0.64; el ruido de Poisson (1/n̄) no se diluye, así que la reducción total es menor.
   - Las diferencias entre estimadores vienen del ruido de DR y RR con aleatorios finitos.

---

## 7. Pseudocódigo completo

```python
cosmo = CLASS(cosmología)                       # P_L(k), H(z)
for g0 in [0, 0.2]:
    Dp   = crecimiento(g0, cosmo)
    Pdil = (Dp / Dp_LCDM)**2 * P_L
    for s in 1..N:
        delta = campo_gaussiano(Pdil, L)
        D, R  = puntos(delta), aleatorios()
        DD, DR, RR = contar_pares(D, R)
        for est in [PH, DP, Ham, LS]:
            xi[g0, est, s] = est(DD, DR, RR)
    C[g0, est] = cov(xi[g0, est, :])            # para cada est
comparar(C)
```

---

## 8. Figuras

1. g(z) del toy model.
2. ⟨ξ̂(r)⟩ con barras de error para g₀ = 0 y g₀ = 0.2, contra el modelo.
3. Matrices de correlación por estimador.
4. Var[ξ̂] por estimador, relativa a LS.
5. Ĉ_dil contra Ĉ_ΛCDM.

---

## 9. Calendario

| Fecha | Meta |
|---|---|
| 24–25 nov | Crecimiento y P_dil validados; campos y catálogos |
| **25 nov** | **Reunión de avance** |
| 26–28 nov | Conteo de pares y los cuatro estimadores; N realizaciones |
| 27–29 nov | Datos para extensiones en Slurm (3 nodos en preempt, sección 10) |
| 29 nov–1 dic | Covarianzas, comparaciones y figuras |
| **2 dic** | **Presentación final** |
| 3–4 dic | Reporte y limpieza del notebook |

---

## 10. Datos para extensiones

Por si cambia el enfoque, `slurm/farm.py` genera realizaciones y guarda los conteos de pares crudos (DD, DR y RR en bins de 1 h⁻¹Mpc hasta 150), además de P̂(k) y ξ̂ en la malla del campo gaussiano, del campo lognormal y de los puntos. Con `--independientes` reproduce bit a bit la corrida de N = 50 000. Los datos quedan en `slurm/datos/` y los detalles en `slurm/LEEME.md`.

**Catálogos anidados.** En la corrida de N = 50 000, la semilla común solo comparte el campo gaussiano: el sorteo de Poisson se desfasa entre modelos y su ruido queda prácticamente independiente (correlación de 0.04–0.08 por bin entre modelos). En `farm.py`, por cada semilla hay un solo catálogo madre por caja, con media n̄_madre V_celda max_A e^{δ_A−σ_A²/2} por celda (n̄_madre es el n̄ más alto de esa caja y el máximo corre sobre las amplitudes de `base`). Cada (n̄, A) se queda con cada punto con probabilidad λ/λ_madre, con un mismo uniforme por punto. Cada catálogo sigue siendo Poisson(λ), pero comparten casi todos los puntos entre amplitudes y entre densidades, y con el mismo n̄ y A son idénticos entre configuraciones (`base`, `alpha10` y `aleatorios_nuevos` tienen el mismo catálogo D). Así, las diferencias entre amplitudes, densidades o catálogos aleatorios tienen mucho menos ruido.

**Todo modelo es una amplitud.** A orden lineal y con la forma de P fija, la simulación solo ve A = P_modelo/P_ΛCDM(z = 0.5). Por eso basta una malla en A, y cada (g₀, z_t, Δz, z) se ubica en ella resolviendo D₊:

| Variación | A |
|---|---|
| g₀ = 0.05 … 0.5 | 0.945 … 0.575 |
| z_t = 0.5 … 10 | 0.984 … 0.582 |
| Δz = 0.05 … 2 | 0.799 … 0.817 |
| ΛCDM en z = 0 / z = 1 | 1.673 / 0.626 |

**Configuraciones:**
- base con A ∈ {0.5, 0.6, 0.7, 0.7999, 0.9, 1.0, 1.2, 1.4, 1.7} (N = 20 000);
- con A ∈ {1, 0.7999}: n̄ = 1, 2 y 5 × 10⁻⁴, aleatorios nuevos en cada realización, α = 10 y L = 500.

**Qué permiten calcular:**
- cualquier estimador (incluido el natural), binning o α;
- Ĉ para cualquier modelo interpolando en A, y de ahí ∂C/∂g₀;
- un ajuste efectivo de C(A) con un polinomio en A. **No** es la separación gaussiana C = A²C^PP + A C^Pn + C^nn: los puntos son lognormales (ξ_pts = e^{Aξ_G} − 1, con términos de 3 y 4 puntos de orden A² y A³), la varianza de Poisson de los pares lleva factores (1 + Aξ) y los estimadores no son lineales. Entre A = 1 y A = 0.8 la diagonal de C cambia ~4 %, pero la parte de C que depende de A es mayor, ~10–20 % (de R ≈ 0.965 = 1 − 0.36 f_PP − 0.2 f_PN). Las correcciones de orden ξ (≈ 4 % en el primer bin) al término de Poisson, que es el ~80–90 % de C, entran lineales en A y son de unos pocos por ciento de C, una fracción apreciable de lo que se leería como C^Pn. Para interpretar los coeficientes hay que compararlos con los términos analíticos o incluir términos de orden A³;
- la dependencia con n̄, aleatorios fijos contra nuevos y la covarianza de P̂(k);
- el escalamiento con el volumen (L = 500 contra L = 1000), con una salvedad: con L = 500, k_f = 0.0126 h/Mpc queda debajo del pico de P (k ≈ 0.017), así que faltan modos cerca del máximo. La potencia que falta con k < k_f aporta ~4×10⁻⁴ a ξ, comparable con ξ ≈ 2×10⁻³ del BAO, y ese efecto se mezcla con el de volumen. Además, L500 usa los mismos números del generador acomodados en una malla de 64³, así que su campo no está relacionado con el de L = 1000.

**Qué no cubren:** cambios de forma de P, RSD, no linealidad real, bordes de un survey ni r > 150 h⁻¹Mpc.
