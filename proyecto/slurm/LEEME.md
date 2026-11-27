# Datos de realizaciones (Slurm)

`farm.py` es el único código que corre en los jobs. Usa las mismas funciones que `covarianza_test.ipynb`
(crecimiento con CLASS, campos gaussianos, puntos lognormal + Poisson y conteo de pares con `cKDTree`),
pero guarda **conteos de pares crudos**, para poder armar después cualquier estimador.
Con `--prueba ... --independientes` (el muestreo del notebook) reproduce bit a bit `datos/xi_estimadores_N50000.npz`,
que es la corrida del reporte; las figuras y números del reporte salen de `../covarianza.ipynb`.

## Idea

A orden lineal, en espacio real y con la forma de P fija, la simulación solo depende de la amplitud
A = P_modelo / P_ΛCDM(z = 0.5): δ escala como √A. Por eso se corre una malla en A, en lugar de una
malla en (g₀, z_t, Δz, z). Cada modelo se ubica en ella resolviendo D₊.

| Variación | A |
|---|---|
| g₀ = 0.05 … 0.5 (z_t = 2, Δz = 0.2) | 0.945 … 0.575 |
| z_t = 0.5 … 10 (g₀ = 0.2) | 0.984 … 0.582 |
| Δz = 0.05 … 2 (g₀ = 0.2) | 0.799 … 0.817 |
| ΛCDM en z = 0 / z = 1 | 1.673 / 0.626 |

La amplitud 0.79989… es exactamente g₀ = 0.2 (z_t = 2, Δz = 0.2, z = 0.5).

**Ruido común.** La realización `s` usa el mismo ruido blanco (semilla `s`) para todas las amplitudes y
configuraciones con malla de 128³, así que sus campos gaussianos son proporcionales. L500 es la excepción:
usa los mismos números del generador acomodados en una malla de 64³, así que su campo no tiene relación con
el de L = 1000 y no se puede comparar semilla por semilla.

**Catálogo madre.** Para que el ruido de Poisson también sea común, por cada semilla hay un solo catálogo
madre por caja, con media n̄_madre V_celda max_A e^{δ_A − σ_A²/2} por celda. n̄_madre es el n̄ más alto de esa
caja (5×10⁻⁴ para L = 1000) y el máximo corre sobre las 9 amplitudes de `base`. El catálogo de (n̄, A) se
queda con cada punto con probabilidad n̄ e^{δ_A−σ_A²/2} / λ_madre, con un mismo uniforme por punto. Así:

- cada catálogo es exactamente Poisson(n̄ V_celda e^{δ_A−σ_A²/2}) por celda;
- entre amplitudes y entre densidades comparten casi todos los puntos (los de n̄ menor están contenidos en los
  de n̄ mayor);
- con el mismo n̄ y la misma A son **idénticos** entre configuraciones: para cada semilla, `base`, `alpha10` y
  `aleatorios_nuevos` tienen el mismo catálogo D, y la comparación entre ellas aísla el efecto de R.

Comparado con el muestreo independiente del notebook, en 2048 semillas de L500 la correlación entre
ξ̂(A = 1) y ξ̂(A = 0.8) por bin sube de 0.09 a 0.94, y la dispersión bin a bin de C(0.8)/C(1) baja de 0.042 a
0.016 (equivale a ~7 veces más realizaciones). La prueba pareada no muestra sesgo.

## Configuraciones (en orden de prioridad)

| Configuración | A | n̄ [(h/Mpc)³] | Aleatorios | α | L [Mpc/h] | N |
|---|---|---|---|---|---|---|
| `base` | 0.5, 0.6, 0.7, 0.7999, 0.9, 1.0, 1.2, 1.4, 1.7 | 5e-5 | fijos | 1, 3 | 1000 | 20 000 |
| `nbar1e-4` | 1.0, 0.7999 | 1e-4 | fijos | 1, 3 | 1000 | 20 000 |
| `nbar2e-4` | 1.0, 0.7999 | 2e-4 | fijos | 1, 3 | 1000 | 10 000 |
| `aleatorios_nuevos` | 1.0, 0.7999 | 5e-5 | nuevos en cada realización | 1, 3 | 1000 | 20 000 |
| `alpha10` | 1.0, 0.7999 | 5e-5 | fijos | 10 | 1000 | 20 000 |
| `nbar5e-4` | 1.0, 0.7999 | 5e-4 | fijos | 1 | 1000 | 5000 |
| `L500` | 1.0, 0.7999 | 5e-5 | fijos | 1, 3 | 500 (malla 64³) | 20 000 |

Todas comparten z = 0.5, Ω_m = 0.3, σ₈ = 0.8 y la forma de P_L de CLASS. Los aleatorios fijos se generan con la
secuencia del notebook (`rng(42)` → campo de ejemplo → sus puntos → R), pero solo en `base` coinciden con los
del notebook: en las demás cambian n̄, L, la malla o los α, así que R es otro.

## Archivos

```
datos/                                # no está en git (.gitignore): ~2 GB
  xi_estimadores_N50000.npz           # corrida del reporte: 4 estimadores, bins de 5 Mpc/h (g0 = 0 y 0.2),
                                      # muestreo independiente
  <config>/RR_aleatorios_fijos.npz    # RR de los aleatorios fijos (con una huella para validar el caché)
  <config>/semillas_<s0>-<s1>.npz     # un bloque de realizaciones
  <config>/descartados/               # bloques que no correspondían al código (se apartan, no se borran)
datos_prueba/                         # salida de --prueba y --hasta (nunca se mezcla con datos/)
logs/                                 # salidas de Slurm y la copia de farm.py de cada envío
```

Cada bloque contiene (n = número de semillas del bloque, nA = amplitudes, nα = valores de α):

| Arreglo | Forma | Contenido |
|---|---|---|
| `DD` | (n, nA, 150) int64 | pares DD ordenados, sin normalizar, en bins de 1 Mpc/h de 0 a 150 |
| `DR` | (n, nA, nα, 150) int64 | pares DR |
| `RR` | (nα, 150), o (n, nα, 150) con aleatorios nuevos | pares RR ordenados |
| `ND`, `NR` | (n, nA), (nα,) | número de puntos (con aleatorios nuevos, N_R = α·int(n̄V)) |
| `Pk_gauss`, `xi_malla_gauss` | (n, nA, nk), (n, nA, 26) | P̂(k) y ξ̂ en la malla del δ **gaussiano** (antes de la lognormal) |
| `Pk_lognormal`, `xi_malla_lognormal` | (n, nA, nk), (n, nA, 26) | lo mismo para el campo lognormal e^{δ−σ²/2} − 1, que es el que trazan los puntos |
| `Pk_puntos`, `xi_malla_puntos` | (n, nA, nk), (n, nA, 26) | lo mismo para los conteos por celda de los puntos (sin restar 1/n̄) |
| `formato`, `muestreo`, `semillas`, `amps`, `amps_madre`, `nbar`, `nbar_madre`, `alphas`, `L`, `nmesh`, `aleatorios`, `r_edges`, `r_edges_malla`, `k_edges`, `nmodos` | — | metadatos |

Normalización, igual que en el notebook: DD/(N_D(N_D−1)), DR/(N_D N_R), RR/(N_R(N_R−1)).
Para comparar `Pk_puntos` (menos 1/n̄) con un campo, la referencia es `Pk_lognormal`, no `Pk_gauss`.

## Cargar

`farm.py` también es la biblioteca para leer sus datos (desde `proyecto/`):

```python
import sys
sys.path.insert(0, 'slurm')
from farm import cargar, estimadores, rebinear, xi_teo, espectro

d = cargar('base')                   # solo los bloques válidos (los que el job aceptaría), concatenados
xi = estimadores(d, iA=5, ia=1)      # {'PH','DP','Ham','LS'}: (N, 26) para A = d['amps'][5] = 1 y alpha = 3,
                                     # en bins de 5 Mpc/h desde 20 (ancho y r_min se pueden cambiar)
k, P_L, A_g02 = espectro()           # P_L de CLASS en z = 0.5 y la amplitud de g0 = 0.2
xi_t = xi_teo(r, k, d['amps'][5]*P_L)
```

`cargar` devuelve los arreglos por semilla concatenados (`semillas`, `DD`, `DR`, `ND`, `Pk_*`, `xi_malla_*` y `RR`
con aleatorios nuevos) y los metadatos del primer bloque (`amps`, `alphas`, `NR`, `RR` fijo, `r_edges`, ...).

## Correr

```bash
cd proyecto/slurm
mkdir -p logs && sbatch --array=0-2 farm.sbatch     # 3 nodos en preempt, con --requeue
python farm.py --estado                             # avance por configuración (hechos, en curso)
python farm.py --prueba base 0 8 --procesos 8                     # un bloque suelto, en datos_prueba/
python farm.py --prueba base 0 8 --procesos 8 --independientes    # muestreo del notebook, en datos_prueba/
```

**Código congelado por envío.** `farm.sbatch` copia `farm.py` a `logs/farm_<job>.py` al arrancar la primera tarea
y todas las tareas (y sus reencolados) corren esa copia, así que editar `farm.py` no afecta un job en curso.

**Reparto dinámico.** Cada tarea del array toma el siguiente bloque libre en orden de prioridad, con un
`.lock` de creación exclusiva que renueva cada 2 min. Por eso todas las tareas trabajan primero en `base`,
después en `nbar1e-4`, etc., y una tarea termina (y libera su nodo) cuando ya no quedan bloques libres.
El número de tareas no importa: se pueden enviar más o volver a enviar una sola (`sbatch --array=0 farm.sbatch`).
Un lock sin renovar en 6 min, o de una encarnación anterior de la misma tarea (reencolada por preempt), se
puede tomar; cada tarea recorre la lista hasta que en una pasada ya no toma nada, así que también recoge los
bloques de un job cancelado o de un nodo que falló.

**Validación.** Un bloque solo se salta si existe y sus metadatos (`formato`, `muestreo`, amplitudes, n̄, α,
L, malla, catálogo madre, semillas) corresponden al código actual. Si no, se aparta a `descartados/` y se
recalcula, así que nunca se mezclan bloques de versiones distintas.

**Escritura.** Cada archivo se escribe a un temporal propio (nodo y proceso) y se renombra al terminar.

## Qué se puede calcular y con qué cuidado

- Cualquier estimador (incluido el natural, con RR analítico), cualquier binning de 1 Mpc/h o más, cualquier α disponible.
- Ĉ para cualquier modelo interpolando en A, y su derivada respecto a A (o a g₀, z_t…).
- **Ajuste de C(A):** un polinomio en A es un ajuste efectivo, **no** la separación gaussiana
  A²C^PP + A C^Pn + C^nn. Los puntos son lognormales (ξ_pts = e^{Aξ_G} − 1, con términos de 3 y 4 puntos de
  orden A² y A³), la varianza de Poisson de los pares lleva factores (1 + Aξ) y los estimadores no son
  lineales. Entre A = 1 y A = 0.8 la diagonal de C cambia ~4 %; la parte de C que depende de A es mayor,
  ~10–20 % (de R ≈ 0.965 = 1 − 0.36 f_PP − 0.2 f_PN). Las correcciones de orden ξ (≈ 4 % en el primer bin)
  al término de Poisson, que es el ~80–90 % de C, entran lineales en A y son de unos pocos por ciento de C:
  una fracción apreciable de lo que se leería como C^Pn. Para interpretar los coeficientes hay que
  compararlos con los términos analíticos o incluir términos de orden A³.
- **Escalamiento con el volumen (L500 contra base):** con L = 500, k_f = 0.0126 h/Mpc queda debajo del pico
  de P (k ≈ 0.017), así que faltan modos cerca del máximo. La potencia que falta con k < k_f aporta ~4×10⁻⁴ a
  ξ, comparable con ξ ≈ 2×10⁻³ del BAO, y ese efecto se mezcla con el de volumen.

## Costo

- Medido: 1.38 nodo-s por semilla en `base` (9 amplitudes) en el nodo de cómputo. En total son ~41 horas-nodo
  de trabajo, ~14 h por nodo con 3 nodos.
- En `preempt` se cobra 1 las primeras 2 h de cada job y 0.5 después: ~24 horas-nodo, contra ~41 en `regular`.
- Cobrado a `desi` (usuario, `iris`): 6.3 de 1000 horas-nodo antes de estos envíos.
- Dos envíos se cancelaron antes de terminar: 62107796 (~30 min, para pasar a catálogos anidados) y 62109695
  (~15 min, para compartir el catálogo madre entre configuraciones y repartir el trabajo con locks).
- Costo real: (se completa al terminar)

## Qué no cubre

Cambios de forma de P(k) (otros Ω_m, n_s), RSD y velocidades, no linealidad real (el modelo es lognormal),
bordes de un survey, r > 150 Mpc/h, ni escalas menores que ~15 Mpc/h (celdas de 7.8 Mpc/h).
