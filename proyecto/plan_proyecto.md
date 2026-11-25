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
N       = 60–100              # realizaciones por modelo (el conteo de pares es lo caro)
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
def campo_gaussiano(P, L, Nmesh, seed):
    w = ruido_blanco_complejo(seed)                 # con δ(−k) = δ*(k)
    delta_k = sqrt(P(|k|) * V) * w
    return irfft(delta_k)                           # δ(x) en la malla

def puntos(delta, nbar, seed):
    # densidad positiva: 1 + δ → exp(δ_G − σ²/2)  (el δ gaussiano puede ser < −1)
    # conteo por celda ~ Poisson[nbar · V_celda · (1 + δ)]; posiciones uniformes dentro de la celda
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
| 29 nov–1 dic | Covarianzas, comparaciones y figuras |
| **2 dic** | **Presentación final** |
| 3–4 dic | Reporte y limpieza del notebook |

---

## 10. Nota sobre el notebook actual

`covarianza.ipynb` hace más de lo que entra en este alcance: covarianza gaussiana analítica, P̂(k), varios n̄, el estimador natural, ∂C/∂g₀ y el ajuste de amplitud. Para la entrega conviene dejar solo los pasos 1–6 de este plan.
