# Plan del proyecto: covarianza de estimadores de dos puntos en modelos de dilución de materia oscura

**Alcance:** solo materia, orden lineal y espacio real. No se incluyen distorsiones en espacio de redshift (RSD), peso de masa 𝓜, velocidades ni términos no lineales.

**Base teórica:** A. Avilés & J. Ferrara, *Dark Matter Dilution* (borrador).

**Covarianza:** formalismo gaussiano. Es el término gaussiano de Steele & Smith (2026) y la fórmula de Wang et al. (2026), restringida a espacio real.

**Fechas:** reunión de avance el 25 de noviembre, presentación final el 2 de diciembre y entrega el 4 de diciembre.

---

## 0. Idea general

```
g(z) ──► D₊(z) ──► P_dil(k) ──► campos gaussianos ──► P̂(k), ξ̂(r) ──► covarianza
                                    (N realizaciones)       muestral vs. gaussiana
```

- La dilución entra **solo** por el factor de crecimiento D₊, de modo que P_dil = [D₊^dil/D₊^ΛCDM]² P_ΛCDM.
- La covarianza gaussiana depende **solo** de P(k), del volumen V y de la densidad de partículas n̄.
- **Qué se espera ver:** en un campo continuo, la dilución solo reescala la covarianza (como P², es decir como D₊⁴). Lo interesante aparece con **partículas discretas**: el ruido de disparo 1/n̄ no se diluye. Eso cambia la relación señal/ruido y el desempeño de los estimadores de ξ.

---

## 1. Parámetros globales

```python
# cosmología de fondo (ΛCDM plano fijo, igual para ambos modelos)
Om0, h, Ob0, ns, As = 0.3, 0.7, 0.045, 0.965, 2.1e-9

# dilución (benchmark del borrador, ec. 3.13)
zt, dz = 2.0, 0.2
g0_list = [0.0, 0.1, 0.2]          # 0 = ΛCDM; 0.1 para validar la derivada; 0.2 = fiducial
z_obs = 0.5                        # redshift de trabajo (también se puede probar z = 0)

# caja y malla
L  = 1000.0                        # Mpc/h
N  = 128                           # malla N³ (256 si da tiempo)
V  = L**3;  Vcell = V / N**3
kf = 2*pi/L;  kNyq = pi*N/L        # ≈ 0.0063 y ≈ 0.40 h/Mpc

# binning
k_edges = arange(kf, 0.30, 2*kf)       # P(k)
r_edges = arange(20, 200 + 5, 5)       # ξ(r), Mpc/h (incluye el BAO ~ 105)

# realizaciones y trazadores
Ns      = 500                      # realizaciones (campo continuo)
Ns_part = 100–200                  # realizaciones con partículas (más caras)
nbar_list = [1e-4, 3e-4, 1e-3]     # (h/Mpc)³ → N_D ≈ 1e5, 3e5, 1e6
alpha_R   = [1, 5, 20]             # N_R/N_D para los estimadores con aleatorios
seeds = range(Ns)                  # la MISMA semilla para todos los g0
```

---

## 2. Paso 1: crecimiento lineal D₊(z)

### Ecuación

La ecuación 3.2 del borrador, con x = ln a y ' = d/dx, queda:

```
D'' + (2 + ε + g) D' − (3/2) Ω_m(a) (1 + Ξ) D = 0

ε  = dlnH/dlna = −(3/2) Ω_m(a)                 (ΛCDM plano, sin radiación)
Ξ  = −(2 / 3Ω_m) [ g (2 + ε) + dg/dlna ]
```

Equivale al sistema de primer orden, más estable numéricamente:

```
δ' = u − g δ
u' = −(2 + ε) u + (3/2) Ω_m δ
```

### Pseudocódigo

```python
def g_of_z(z, g0):
    return 0.5*g0*(1 - tanh((z - zt)/dz))

def dg_dlna(z, g0):                       # dlna = −dz/(1+z)
    dgdz = -0.5*g0/dz / cosh((z - zt)/dz)**2
    return -(1 + z)*dgdz                  # solo hace falta si usas la forma de 2º orden

def E2(a):  return Om0*a**-3 + (1 - Om0)  # H²/H0²
def Om(a):  return Om0*a**-3 / E2(a)

def rhs(x, y, g0):
    a = exp(x); z = 1/a - 1
    delta, u = y
    g   = g_of_z(z, g0)
    eps = -1.5*Om(a)
    return [u - g*delta,
            -(2 + eps)*u + 1.5*Om(a)*delta]

def solve_growth(g0):
    a_ini = 1e-3                          # ahí g ≈ 0: solución de crecimiento de EdS
    y0 = [a_ini, a_ini]                   # δ = a, u = a
    sol = solve_ivp(rhs, [log(a_ini), 0], y0, args=(g0,),
                    dense_output=True, rtol=1e-10, atol=1e-12)
    D = lambda z: sol.sol(log(1/(1+z)))[0]
    f = lambda z: (lambda y: y[1]/y[0] - g_of_z(z, g0))(sol.sol(log(1/(1+z))))
    return D, f

D = {g0: solve_growth(g0)[0] for g0 in g0_list}
ratio = {g0: (D[g0](z_obs) / D[0.0](z_obs))**2 for g0 in g0_list}   # P_dil / P_ΛCDM
```

### Validaciones

- [ ] Con g₀ = 0, D(z)/D(0) coincide con el crecimiento de ΛCDM (o con el de CAMB) a < 0.1 %.
- [ ] Con g₀ = 0, f ≈ Ω_m(z)^0.55.
- [ ] Con g₀ = 0.2, f(z) cae fuertemente después de z ≈ 2 (comparar con la figura 1 del borrador).
- [ ] La forma de primer orden y la de segundo orden (con Ξ) dan el mismo D.

### Figuras

1. g(z).
2. D_dil/D_ΛCDM contra z.
3. f(z) para cada g₀.

---

## 3. Paso 2: espectro de potencias lineal

```python
# CAMB (como en la práctica 04): P_ΛCDM a z = 0
k_lin, P_lin0 = camb_linear_pk(Om0, h, Ob0, ns, As, z=0)   # h/Mpc, (Mpc/h)³

# misma condición inicial en ambos modelos ⇒ solo cambia D₊
def P_model(k, g0, z=z_obs):
    return interp(k, k_lin, P_lin0) * (D[g0](z) / D[0.0](0))**2
```

### Validaciones

- [ ] P_model(k, g₀=0, z=0) es igual a P_lin0.
- [ ] P_model(g₀)/P_model(0) es constante en k y vale `ratio[g0]`.

### Figura

4. P(k) para cada g₀ y el cociente (una línea horizontal).

---

## 4. Paso 3: campos gaussianos

### 4a. Campo continuo

```python
k3d = magnitudes |k| de la malla rfft (kx, ky, kz con fftfreq · 2π N/L)

def gaussian_field(seed, P_of_k):
    rng = default_rng(seed)
    w   = rng.normal(0, 1, (N, N, N))                 # ruido blanco, varianza 1 por celda
    dk  = rfftn(w) * sqrt(P_of_k(k3d) / Vcell)
    dk[0, 0, 0] = 0                                   # sin modo k = 0
    return irfftn(dk, s=(N, N, N))                    # δ(x) real
```

Con la **misma semilla**, δ_dil = √ratio · δ_ΛCDM exactamente. Esto sirve como prueba de sanidad, no como resultado.

**Ruido blanco en espacio real:** con `rfftn` de un campo real, la simetría hermítica queda garantizada automáticamente.

### 4b. Trazadores discretos (muestreo de Poisson)

```python
def poisson_tracers(delta, nbar, seed):
    rng = default_rng(seed + 10**6)
    lam = nbar * Vcell * clip(1 + delta, 0, None)     # registrar la fracción con δ < −1
    n   = rng.poisson(lam)                            # partículas por celda
    pos = posiciones uniformes dentro de cada celda (repetir el índice n veces + rng.uniform)
    return pos % L                                    # catálogo D, periódico
```

- Para trabajar con P(k) y ξ(r) en las escalas de interés basta el campo lineal suavizado a la malla. Hay que reportar la fracción de celdas con δ < −1, que debería ser pequeña con N = 128 a z = 0.5.
- **Alternativa por investigar:** una transformación lognormal, δ_LN = exp(δ_G − σ²/2) − 1, que evita el recorte pero cambia ligeramente P(k).

### Validaciones

- [ ] El P̂(k) promedio de las realizaciones continuas reproduce P_model(k) dentro de σ/√Ns.
- [ ] El número de partículas cumple ⟨N_D⟩ = n̄V.
- [ ] El P̂(k) de los trazadores, después de restar 1/n̄, reproduce P_model(k).

---

## 5. Paso 4: estimadores

### 5a. P(k) con FFT

```python
def measure_Pk(delta_grid):                       # campo continuo, o trazadores pintados con CIC
    dk = rfftn(delta_grid) * Vcell
    W  = ventana_CIC(kx, ky, kz)                   # Π sinc²(k_i L/2N); solo para trazadores
    P3 = abs(dk / W)**2 / V
    peso = 2 en rfft, salvo los planos kz = 0 y kz = Nyquist (peso 1)
    P_b, N_k = promedio_por_bin(P3, k3d, k_edges, peso)   # N_k = número de modos (malla completa)
    return P_b - (1/nbar si son trazadores), N_k
```

### 5b. ξ(r) con FFT (campo continuo, rápido)

```python
def measure_xi_fft(delta):
    dk = rfftn(delta)
    xi3 = irfftn(abs(dk)**2, s=(N,)*3) / N**3     # ⟨δ(x) δ(x+r)⟩ sobre la malla
    return promedio_por_bin(xi3, |r| de la malla (con periodicidad), r_edges)
```

### 5c. ξ(r) por conteo de pares (trazadores; conexión con la práctica 05)

```python
def pair_counts(A, B, r_edges, L):
    tA, tB = cKDTree(A, boxsize=L), cKDTree(B, boxsize=L)
    cum = tA.count_neighbors(tB, r_edges)        # acumulado (cumulativo)
    return diff(cum)                             # restar auto-pares si A es B

def estimators(D, R, r_edges):
    ND, NR = len(D), len(R)
    DD = pair_counts(D, D) / (ND*(ND-1))
    RR = pair_counts(R, R) / (NR*(NR-1))
    DR = pair_counts(D, R) / (ND*NR)
    return {
      "PH":   DD/RR - 1,                     # Peebles–Hauser
      "DP":   DD/DR - 1,                     # Davis–Peebles
      "Ham":  DD*RR/DR**2 - 1,               # Hamilton
      "LS":   (DD - 2*DR + RR)/RR,           # Landy–Szalay
      "nat":  DD/RR_analitico - 1,           # caja periódica: RR = V_shell/V
    }
```

- R son aleatorios uniformes en la caja, con N_R = α·N_D y α ∈ {1, 5, 20}.
- Con N_D ≈ 10⁵ y r ≤ 200 Mpc/h hay unos 10⁸ pares. `count_neighbors` solo los cuenta y es viable. Si es lento, usar n̄ = 10⁻⁴ o un r_max más chico.

### Validaciones

- [ ] El ξ_FFT promedio coincide con ξ_model(r) = ∫k²dk/(2π²) P(k) j₀(kr), incluyendo el BAO.
- [ ] ξ_LS ≈ ξ_nat ≈ ξ_FFT en promedio.
- [ ] Con datos uniformes (δ = 0), todos los estimadores dan ≈ 0 (igual que en la práctica 05).

---

## 6. Paso 5: covarianza muestral

```python
def sample_cov(X):                    # X: (Ns, nb)
    m = X.mean(0)
    C = (X - m).T @ (X - m) / (Ns - 1)
    r = C / sqrt(outer(diag(C), diag(C)))       # matriz de correlación
    return m, C, r

def hartlap(C, Ns):
    nb = C.shape[0]
    return (Ns - nb - 2)/(Ns - 1) * inv(C)      # precisión insesgada (requiere Ns > nb + 2)

# error del estimador: σ(Ĉ_ii)/C_ii ≈ sqrt(2/(Ns − 1))  (~6 % con Ns = 500)
```

Esto se aplica a cada combinación (g₀, n̄, estimador, α).

---

## 7. Paso 6: covarianza gaussiana analítica

### 7a. P(k)

Es el término gaussiano de Steele & Smith (ec. 13, 17) y la fórmula de Wang et al. (ec. 2.3–2.6) en espacio real:

```python
def cov_P_gauss(g0, nbar):
    P_b = promedio_por_bin(P_model(k3d, g0))          # promediado igual que la medición
    return diag( 2/N_k * (P_b + 1/nbar)**2 )          # N_k contado en la malla (paso 5a)
```

### 7b. ξ(r)

**Versión discreta.** Es exacta para ξ_FFT, porque usa los mismos modos que la medición:

```python
# ξ̂_i = (1/V) Σ_{k ∈ malla completa} P̂(k) j̄₀,i(k)
# ⇒ Cov[ξ_i, ξ_j] = (2/V²) Σ_{k ∈ malla completa} j̄₀,i(k) j̄₀,j(k) P̃(k)²
def cov_xi_gauss_discrete(g0, nbar):
    Pt  = P_model(k3d, g0) + (1/nbar si hay trazadores)
    # exacto: J_i(k) = promedio de cos(k·r) sobre los puntos r de la malla en el bin i
    # aproximado: J_i(k) ≈ j̄₀(k; r_lo, r_hi), j₀ promediada en la cáscara
    J   = [J_i(k3d) for cada bin]
    return 2/V**2 * sum_k(J_i * J_j * Pt**2)
```

**Versión continua.** Es la de Grieb et al. 2016; sirve como verificación:

```python
# Cov[ξ_i, ξ_j] = (2/V) ∫ k² dk/(2π²) j̄₀(k r_i) j̄₀(k r_j) [P(k) + 1/n̄]²
```

**Por investigar:** el término de Poisson puro (∝ 1/n̄²) en la covarianza de los estimadores de pares. En la literatura aparece un término diagonal adicional ∝ 1/(n̄² V V_shell,i). Hay que comparar con la covarianza muestral para ver si [P + 1/n̄]² basta.

### Pruebas sugeridas

```python
# 1. diagonal
plot(k, diag(C_sample)/diag(C_gauss))               # ≈ 1 ± sqrt(2/Ns)
# 2. estructura
imshow(r_sample); imshow(r_gauss)                   # ξ: bins muy correlacionados; P: casi diagonal
# 3. prueba χ²
chi2_s = (X_s - m) @ inv(C_gauss) @ (X_s - m)       # histograma vs. χ²(nb)
# 4. escalamiento con la dilución (campo continuo, sin shot noise)
C_sample(g0) / C_sample(0)  ==  ratio[g0]**2        # exactamente D⁴ con la misma semilla
```

---

## 8. Paso 7: efecto de la dilución y de los estimadores

### 8a. Señal/ruido de la dilución

```python
Delta = m_P(g0=0.2) - m_P(g0=0)
SN2   = Delta @ hartlap(C_P(g0=0.2)) @ Delta        # (S/N)² en función de k_max y de n̄
```

**Qué se espera:** con n̄ alto, la S/N es casi independiente de n̄. Con n̄ bajo, el shot noise domina y la dilución, que baja P, hace que las mediciones sean relativamente más ruidosas.

### 8b. Comparación de estimadores de ξ (el núcleo del título)

Para cada estimador (PH, DP, Ham, LS), cada α y cada g₀:

```python
sesgo   = mean_s(ξ_est) - ξ_model                   # contra r
varianza= diag(C_sample_est)
ratio_v = varianza / varianza_LS
```

**Pregunta:** ¿cambia la ventaja de LS sobre PH o Hamilton cuando la señal está diluida, es decir con ξ más chica respecto al shot noise?

### 8c. Dependencia de la covarianza con g₀ (opcional, estilo Steele & Smith)

```python
dC_dg0 = (C(g0=0.2) - C(g0=0.0)) / 0.2              # diferencias finitas
C_lin(0.1) = C(0) + 0.1 * dC_dg0                    # predicción lineal
error = norm(C_lin(0.1) - C(0.1)) / norm(C(0.1))     # validar contra la medición

# verificación analítica (campo continuo): ∂ ln C/∂g0 = 4 ∂ ln D₊/∂g0
```

**Prueba de sesgo por usar la covarianza equivocada.** Se ajusta una amplitud A a los datos diluidos, con el modelo A·P_ΛCDM(k):

```python
for C in [C(g0=0.2), C(g0=0)]:
    Prec  = inv(C)
    A_hat = (P_ΛCDM @ Prec @ m_dil) / (P_ΛCDM @ Prec @ P_ΛCDM)
    sigA  = 1/sqrt(P_ΛCDM @ Prec @ P_ΛCDM)
# comparar A_hat y sigA: ¿subestimar o sobreestimar el error?
```

---

## 9. Organización del notebook

```
proyecto/
  plan_proyecto.md                 ← este archivo
  notebooks/
    01_crecimiento.ipynb           Paso 1–2   (D₊, f, P_model)
    02_campos.ipynb                Paso 3     (campos y trazadores + validaciones)
    03_estimadores.ipynb           Paso 4     (P(k), ξ_FFT, conteo de pares)
    04_covarianza.ipynb            Paso 5–7   (muestral, gaussiana, comparaciones)
  src/
    growth.py      g_of_z, solve_growth
    fields.py      gaussian_field, poisson_tracers
    stats.py       measure_Pk, measure_xi_fft, pair_counts, estimators
    covariance.py  sample_cov, hartlap, cov_P_gauss, cov_xi_gauss_discrete
  data/            guardar P̂ y ξ̂ de cada realización en .npz (no recalcular)
```

**Reproducibilidad:** fijar las semillas; guardar los parámetros en el `.npz`; poner comentarios en cada celda (requisito de la entrega).

---

## 10. Calendario

| Fecha | Meta |
|---|---|
| 19–20 nov | Paso 1 y 2: D₊ validado, P_model, figuras 1–4 |
| 21–22 nov | Paso 3a y 5a: campo continuo, P̂(k) validado |
| 23–24 nov | Paso 6–7a: covarianza muestral vs. gaussiana de P(k), prueba χ² |
| **25 nov** | **Reunión de avance:** crecimiento + P(k) + primera covarianza |
| 26–27 nov | Paso 3b, 5b y 5c: trazadores, ξ_FFT, estimadores de pares |
| 28–29 nov | Paso 7b y 8a–8b: covarianza de ξ, S/N, comparación de estimadores |
| 30 nov–1 dic | Paso 8c (opcional), figuras finales, presentación |
| **2 dic** | **Presentación final** |
| 3–4 dic | Reporte y limpieza del notebook |

---

## 11. Figuras para la presentación final

1. g(z), D_dil/D_ΛCDM y f(z).
2. P_model(k) para cada g₀ (cociente constante).
3. P̂(k) y ξ̂(r) promedio contra el modelo, con barras de error.
4. Diagonal de la covarianza muestral entre la gaussiana, para P y para ξ.
5. Matrices de correlación de P y de ξ (muestral contra analítica).
6. Histograma de χ² contra χ²(n_b).
7. S/N de la dilución en función de k_max y de n̄.
8. Varianza de PH, DP, Ham y LS entre la de LS, contra r, para g₀ = 0 y 0.2.
9. (Opcional) ∂C/∂g₀ y el sesgo en el ajuste de amplitud.

---

## 12. Preguntas abiertas

- ¿Recorte de Poisson o transformación lognormal para los trazadores?
- ¿Qué término de Poisson exacto usar en la covarianza gaussiana de los estimadores de pares?
- ¿Trabajar a z = 0.5 (más lineal) o a z = 0 (efecto de la dilución más grande)?
- Verificar las referencias citadas de memoria (Grieb et al. 2016; Hartlap et al. 2007) antes de ponerlas en el reporte.
