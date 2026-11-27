"""Realizaciones con conteos de pares crudos para el proyecto de covarianza con dilución.

Es el código que corre en los jobs de Slurm (farm.sbatch) y también la biblioteca para leer sus datos
(cargar, rebinear, estimadores, xi_teo). Las funciones físicas son las de covarianza_test.ipynb.
El muestreo (catálogo madre por caja), el reparto con locks y el formato de los datos están en LEEME.md.

Uso:
    python farm.py                                # trabajo de una tarea (lo que corre farm.sbatch)
    python farm.py --estado                       # avance por configuración
    python farm.py --prueba base 0 8              # un bloque suelto, en datos_prueba/
    python farm.py --prueba base 0 8 --independientes   # reproduce la corrida N50000 (en datos_prueba/)
"""
import argparse
import functools
import os
import socket
import threading
import time
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import fft
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(AQUI, 'datos')          # se cambia con --salida; con --prueba o --hasta es datos_prueba/
MUESTREO = 'anidado'                         # 'independiente' con --independientes
FORMATO = 'farm-3'                           # versión del formato y del muestreo de los bloques

# cosmología y toy model (igual que el notebook)
Om0, Ob0, h, ns, sigma8 = 0.3, 0.045, 0.7, 0.965, 0.8
zt, dz = 2.0, 0.2
z_obs = 0.5

R_EDGES = np.arange(0, 151, 1.0)           # bins finos de 1 Mpc/h (count_neighbors no acumulativo: mismo costo que bins de 5)
R_EDGES_MALLA = np.arange(20, 155, 5.0)    # bins de xi en la malla (FFT)
CAMPOS = ['gauss', 'lognormal', 'puntos']  # P(k) y xi en la malla de cada uno

# 'g02' es la amplitud exacta de g0 = 0.2 (sale de CLASS, ver amps_de)
A9 = [0.5, 0.6, 0.7, 'g02', 0.9, 1.0, 1.2, 1.4, 1.7]

# catálogo madre de cada caja (L, malla): todas las configuraciones de esa caja salen de él
MADRE = {(1000.0, 128): dict(nbar=5e-4, amps=A9),
         (500.0, 64): dict(nbar=5e-5, amps=A9)}

# configuraciones en orden de prioridad
CONFIGS = [
    dict(nombre='base', amps=A9, nbar=5e-5, alphas=[1, 3],
         aleatorios='fijos', L=1000.0, nmesh=128, N=20000, bloque=1000),
    dict(nombre='nbar1e-4', amps=[1.0, 'g02'], nbar=1e-4, alphas=[1, 3],
         aleatorios='fijos', L=1000.0, nmesh=128, N=20000, bloque=1000),
    dict(nombre='nbar2e-4', amps=[1.0, 'g02'], nbar=2e-4, alphas=[1, 3],
         aleatorios='fijos', L=1000.0, nmesh=128, N=10000, bloque=250),
    dict(nombre='aleatorios_nuevos', amps=[1.0, 'g02'], nbar=5e-5, alphas=[1, 3],
         aleatorios='nuevos', L=1000.0, nmesh=128, N=20000, bloque=2000),
    dict(nombre='alpha10', amps=[1.0, 'g02'], nbar=5e-5, alphas=[10],
         aleatorios='fijos', L=1000.0, nmesh=128, N=20000, bloque=2000),
    dict(nombre='nbar5e-4', amps=[1.0, 'g02'], nbar=5e-4, alphas=[1],
         aleatorios='fijos', L=1000.0, nmesh=128, N=5000, bloque=250),
    dict(nombre='L500', amps=[1.0, 'g02'], nbar=5e-5, alphas=[1, 3],
         aleatorios='fijos', L=500.0, nmesh=64, N=20000, bloque=5000),
]


def madre(cfg):
    return MADRE[cfg['L'], cfg['nmesh']]


for _c in CONFIGS:      # cada configuración tiene que caber en el catálogo madre de su caja
    assert _c['nbar'] <= madre(_c)['nbar'] and set(_c['amps']) <= set(madre(_c)['amps']), _c['nombre']


def config(nombre):
    return next(c for c in CONFIGS if c['nombre'] == nombre)


# ---------------------------------------------------------------- 1. crecimiento y espectro (CLASS)

@functools.cache
def espectro():
    """P_L(k, z_obs) de CLASS y la amplitud A de g0 = 0.2, como en el notebook (se calcula una vez)."""
    from classy import Class

    cosmo = Class()
    cosmo.set({'output': 'mPk', 'h': h, 'omega_b': Ob0*h**2, 'omega_cdm': (Om0 - Ob0)*h**2,
               'n_s': ns, 'sigma8': sigma8, 'P_k_max_h/Mpc': 20.0, 'z_max_pk': 1.0})
    cosmo.compute()

    bg = cosmo.get_background()
    lna, i = np.unique(-np.log1p(bg['z']), return_index=True)
    lnH = CubicSpline(lna, np.log(bg['H [1/Mpc]'][i]))
    Om = CubicSpline(lna, ((bg['(.)rho_cdm'] + bg['(.)rho_b'])/bg['(.)rho_crit'])[i])
    D_class = CubicSpline(lna, bg['gr.fac. D'][i])
    f_class = CubicSpline(lna, bg['gr.fac. f'][i])
    dlnH = lnH.derivative()

    def g(z, g0):
        return g0/2*(1 - np.tanh((z - zt)/dz))

    def dg(z, g0):
        return (1 + z)*g0/(2*dz)*(1 - np.tanh((z - zt)/dz)**2)

    def crecimiento(g0, z_ini=99):
        def rhs(x, y):
            z = np.exp(-x) - 1
            e = dlnH(x)
            D, dD = y
            return [dD, -(2 + e + g(z, g0))*dD + (1.5*Om(x) - g(z, g0)*(2 + e) - dg(z, g0))*D]
        x0 = -np.log1p(z_ini)
        y0 = [D_class(x0), f_class(x0)*D_class(x0)]
        sol = solve_ivp(rhs, [x0, 0], y0, dense_output=True, rtol=1e-10, atol=1e-13)
        return lambda z: sol.sol(-np.log1p(z))[0]

    ratio = (crecimiento(0.2)(z_obs)/crecimiento(0.0)(z_obs))**2
    k = np.logspace(-4, 1, 2000)
    P_L = np.array([cosmo.pk_lin(ki*h, z_obs) for ki in k])*h**3
    return k, P_L, ratio


def amps_de(lista):
    return [espectro()[2] if A == 'g02' else float(A) for A in lista]


def xi_teo(r, k, Pk):
    """xi(r) = int k^2 dk/(2 pi^2) P(k) j0(kr); e^{-k^2} solo para que converja (no cambia r >= 20)."""
    kk = np.linspace(1e-4, 5, 50000)
    Pkk = np.interp(kk, k, Pk)*np.exp(-kk**2)
    j0 = np.sinc(kk[:, None]*r/np.pi)
    return np.trapezoid(kk[:, None]**2*Pkk[:, None]*j0, kk, axis=0)/(2*np.pi**2)


# ---------------------------------------------------------------- 2. configuración activa
# G guarda todo lo de la configuración activa; los procesos del Pool lo heredan al hacer fork.
G = {}


def pares(t1, t2):
    """Pares por bin de R_EDGES (sin el bin r = 0, que solo tiene los pares de un punto consigo)."""
    return t1.count_neighbors(t2, R_EDGES, cumulative=False)[1:]


def arboles(L, alphas, ND0, rng):
    """Catálogos aleatorios uniformes de alpha*ND0 puntos, como árboles periódicos."""
    return {a: cKDTree(rng.uniform(0, L, (a*ND0, 3)), boxsize=L) for a in alphas}


def preparar(cfg):
    k, P_L, _ = espectro()
    L, N, nbar = cfg['L'], cfg['nmesh'], cfg['nbar']
    Vcell = (L/N)**3
    amps, amps_madre = amps_de(cfg['amps']), amps_de(madre(cfg)['amps'])

    kx = 2*np.pi*fft.fftfreq(N, d=L/N)
    kz = 2*np.pi*fft.rfftfreq(N, d=L/N)
    K = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kz[None, None, :]**2)

    def raiz_P(A):
        Pk = np.interp(K, k, A*P_L)
        Pk[0, 0, 0] = 0
        return np.sqrt(Pk/Vcell)

    # P(k) en la malla: bins de ancho kf; cada modo de rfftn pesa 2 salvo en los planos kz = 0 y Nyquist
    kf = 2*np.pi/L
    k_edges = kf*(np.arange(N//2 + 1) + 0.5)
    pesos = np.full(K.shape, 2.0)
    pesos[..., 0] = 1.0
    pesos[..., -1] = 1.0
    kidx = np.digitize(K, k_edges) - 1
    ksel = np.flatnonzero((kidx >= 0) & (kidx < len(k_edges) - 1))
    kidx, kpesos = kidx.ravel()[ksel], pesos.ravel()[ksel]

    # xi en la malla: distancias periódicas
    ix = np.minimum(np.arange(N), N - np.arange(N))*(L/N)
    R3 = np.sqrt(ix[:, None, None]**2 + ix[None, :, None]**2 + ix[None, None, :]**2)
    ridx = np.digitize(R3, R_EDGES_MALLA) - 1
    rsel = np.flatnonzero((ridx >= 0) & (ridx < len(R_EDGES_MALLA) - 1))
    ridx = ridx.ravel()[rsel]

    G.clear()
    G.update(cfg=cfg, L=L, N=N, nbar=nbar, nbar_madre=madre(cfg)['nbar'], Vcell=Vcell, amps=amps,
             raiz_madre=[raiz_P(A) for A in amps_madre],
             en_cfg={amps_madre.index(A): iA for iA, A in enumerate(amps)},   # amplitud de la madre -> de cfg
             k_edges=k_edges, kidx=kidx, ksel=ksel, kpesos=kpesos,
             nmodos=np.bincount(kidx, weights=kpesos, minlength=len(k_edges) - 1),
             ridx=ridx, rsel=rsel, nr=np.bincount(ridx, minlength=len(R_EDGES_MALLA) - 1),
             ND0=int(nbar*L**3))

    if cfg['aleatorios'] == 'fijos':
        # misma secuencia que el notebook: rng(42) -> campo de ejemplo -> sus puntos -> R[alpha]
        # (los aleatorios coinciden con los del notebook solo en base, que tiene su mismo nbar, L, malla y alphas)
        rng = np.random.default_rng(42)
        delta = fft.irfftn(fft.rfftn(rng.standard_normal((N, N, N)))*raiz_P(1.0), s=(N, N, N))
        muestrear(nbar*Vcell*np.exp(delta - delta.var()/2), rng)     # solo avanza el rng, como el notebook
        G['arboles_R'] = arboles(L, cfg['alphas'], G['ND0'], rng)
        G['NR'] = np.array([G['arboles_R'][a].n for a in cfg['alphas']])
        G['RR'] = rr_fijos(cfg)


def coincide(ruta, esperado):
    """True si el npz existe y tiene esos valores en esas claves."""
    try:
        with np.load(ruta) as f:
            return all(k in f and np.array_equal(f[k], v) for k, v in esperado.items())
    except Exception:
        return False


def rr_fijos(cfg):
    """RR de los aleatorios fijos; se guarda una vez por configuración para no repetirlo al reencolar.

    El caché solo se usa si corresponde a estos mismos aleatorios (alphas, N_R, bins y una huella de R).
    """
    ruta = os.path.join(DATOS, cfg['nombre'], 'RR_aleatorios_fijos.npz')
    arb = G['arboles_R']
    esperado = dict(alphas=cfg['alphas'], NR=G['NR'], r_edges=R_EDGES,
                    huella=np.array([arb[a].data[:100].sum() for a in cfg['alphas']]))
    if coincide(ruta, esperado):
        return np.load(ruta)['RR']
    if os.path.exists(ruta):
        print(f'{ruta}: el caché no corresponde a estos aleatorios; se recalcula', flush=True)
    RR = np.array([pares(arb[a], arb[a]) for a in cfg['alphas']])
    guardar(ruta, RR=RR, **esperado)
    return RR


# ---------------------------------------------------------------- 3. una realización

def muestrear(lam, rng):
    """Poisson(lam) por celda y posiciones uniformes dentro de cada celda (igual que el notebook)."""
    N, L = G['N'], G['L']
    n = rng.poisson(lam)
    celda = np.repeat(np.arange(n.size), n.ravel())
    ijk = np.column_stack(np.unravel_index(celda, (N, N, N)))
    return (ijk + rng.uniform(size=ijk.shape))*L/N, celda


def potencia_y_xi(dk):
    """P(k) y xi(r) binneados en la malla a partir de la FFT de delta."""
    N, Vcell, V = G['N'], G['Vcell'], G['L']**3
    p3 = np.abs(dk)**2
    Pk = np.bincount(G['kidx'], weights=G['kpesos']*p3.ravel()[G['ksel']], minlength=len(G['nmodos']))/G['nmodos']*Vcell**2/V
    xi3 = fft.irfftn(p3, s=(N, N, N))/N**3
    xi = np.bincount(G['ridx'], weights=xi3.ravel()[G['rsel']], minlength=len(G['nr']))/G['nr']
    return Pk, xi


def realizacion(s):
    cfg, L, N, Vcell = G['cfg'], G['L'], G['N'], G['Vcell']
    alphas, nA = cfg['alphas'], len(G['amps'])

    rng = np.random.default_rng(s)                  # misma semilla que el notebook
    W = fft.rfftn(rng.standard_normal((N, N, N)))
    estado = rng.bit_generator.state

    if cfg['aleatorios'] == 'nuevos':
        arboles_R = arboles(L, alphas, G['ND0'], np.random.default_rng([s, 7]))
        RR = np.array([pares(arboles_R[a], arboles_R[a]) for a in alphas])
    else:
        arboles_R, RR = G['arboles_R'], None

    nk, nr = len(G['nmodos']), len(G['nr'])
    out = dict(DD=np.empty((nA, len(R_EDGES) - 1), np.int64),
               DR=np.empty((nA, len(alphas), len(R_EDGES) - 1), np.int64),
               ND=np.empty(nA, np.int64),
               **{f'Pk_{c}': np.empty((nA, nk)) for c in CAMPOS},
               **{f'xi_malla_{c}': np.empty((nA, nr)) for c in CAMPOS})
    if RR is not None:
        out['RR'] = RR

    # campos: delta gaussiano y la densidad lognormal e^{delta - s^2/2} (lo que trazan los puntos);
    # se recorren todas las amplitudes de la madre para su máximo, y se guardan las de esta configuración
    e_max, lams = None, [None]*nA
    for j, raiz in enumerate(G['raiz_madre']):
        delta = fft.irfftn(W*raiz, s=(N, N, N))
        e = np.exp(delta - delta.var()/2)
        e_max = e if e_max is None else np.maximum(e_max, e)
        if j in G['en_cfg']:
            iA = G['en_cfg'][j]
            lams[iA] = G['nbar']*Vcell*e
            out['Pk_gauss'][iA], out['xi_malla_gauss'][iA] = potencia_y_xi(fft.rfftn(delta))
            out['Pk_lognormal'][iA], out['xi_malla_lognormal'][iA] = potencia_y_xi(fft.rfftn(e - 1))

    # catálogos de puntos
    if MUESTREO == 'anidado':
        lam_madre = G['nbar_madre']*Vcell*e_max
        pos, celda = muestrear(lam_madre, rng)
        umad = rng.uniform(size=len(celda))*lam_madre.ravel()[celda]
        queda = [umad < lam.ravel()[celda] for lam in lams]      # probabilidad lambda / lambda_madre
        catalogos = [(pos[q], celda[q]) for q in queda]
    else:
        catalogos = []
        for lam in lams:
            rng.bit_generator.state = estado
            catalogos.append(muestrear(lam, rng))

    for iA, (D, celda) in enumerate(catalogos):
        tD = cKDTree(D, boxsize=L)
        out['DD'][iA] = pares(tD, tD)
        for ia, a in enumerate(alphas):
            out['DR'][iA, ia] = pares(tD, arboles_R[a])
        out['ND'][iA] = len(D)
        n = np.bincount(celda, minlength=N**3).reshape(N, N, N)
        dn = n/(len(D)/N**3) - 1
        out['Pk_puntos'][iA], out['xi_malla_puntos'][iA] = potencia_y_xi(fft.rfftn(dn))
    return out


# ---------------------------------------------------------------- 4. bloques, validación y escritura

def guardar(ruta, **arreglos):
    """Escribe a un temporal propio y renombra: un job interrumpido no deja archivos a medias, y dos tareas
    que escriben el mismo archivo a la vez (por ejemplo el caché de RR) no chocan."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    tmp = f'{ruta}.{socket.gethostname()}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arreglos)
    os.replace(tmp, ruta)


def ruta_bloque(cfg, s0, s1):
    return os.path.join(DATOS, cfg['nombre'], f'semillas_{s0:05d}-{s1:05d}.npz')


def metadatos(cfg, s0, s1):
    return dict(formato=FORMATO, muestreo=MUESTREO, semillas=np.arange(s0, s1), nbar=cfg['nbar'],
                alphas=cfg['alphas'], L=cfg['L'], nmesh=cfg['nmesh'], aleatorios=cfg['aleatorios'],
                nbar_madre=madre(cfg)['nbar'], amps=amps_de(cfg['amps']), amps_madre=amps_de(madre(cfg)['amps']))


def bloque_valido(cfg, s0, s1):
    """True si el bloque existe y sus metadatos corresponden a este código y configuración."""
    return coincide(ruta_bloque(cfg, s0, s1), metadatos(cfg, s0, s1))


def apartar(cfg, s0, s1):
    """Mueve un bloque que no corresponde al código actual a descartados/ (no se borra)."""
    ruta = ruta_bloque(cfg, s0, s1)
    destino = os.path.join(os.path.dirname(ruta), 'descartados', f'{os.path.basename(ruta)}.{int(time.time())}')
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    os.replace(ruta, destino)
    print(f'{ruta}: no corresponde a este código; se movió a {destino}', flush=True)


def correr_bloque(cfg, s0, s1, pool):
    t0 = time.time()
    res = pool.map(realizacion, range(s0, s1), chunksize=1)
    datos = {key: np.array([r[key] for r in res]) for key in res[0]}
    if cfg['aleatorios'] == 'fijos':
        datos['RR'], datos['NR'] = G['RR'], G['NR']
    guardar(ruta_bloque(cfg, s0, s1), **datos, **metadatos(cfg, s0, s1),
            r_edges=R_EDGES, r_edges_malla=R_EDGES_MALLA, k_edges=G['k_edges'], nmodos=G['nmodos'])
    dt = time.time() - t0
    print(f"{cfg['nombre']} {s0}-{s1}: {dt:.0f} s ({dt/(s1 - s0):.2f} s/semilla con el nodo)", flush=True)


def bloques(solo=None, hasta=None):
    out = []
    for cfg in CONFIGS:
        if solo and cfg['nombre'] not in solo:
            continue
        N = min(cfg['N'], hasta) if hasta else cfg['N']
        out += [(cfg, s0, min(s0 + cfg['bloque'], N)) for s0 in range(0, N, cfg['bloque'])]
    return out


# ---------------------------------------------------------------- 5. leer los datos (para los notebooks)

def cargar(nombre):
    """Junta los bloques válidos de una configuración (los mismos que el job aceptaría).

    Devuelve un dict con los arreglos por semilla concatenados (semillas, DD, DR, ND, Pk_*, xi_malla_* y
    RR si los aleatorios son nuevos) y los metadatos del primer bloque (amps, alphas, NR, RR fijo, bins...).
    """
    cfg = config(nombre)
    validos = [ruta_bloque(c, s0, s1) for c, s0, s1 in bloques([nombre]) if bloque_valido(c, s0, s1)]
    if not validos:
        raise FileNotFoundError(f'no hay bloques válidos de {nombre} en {DATOS}')
    bloq = [np.load(f) for f in validos]
    por_semilla = ['semillas', 'DD', 'DR', 'ND'] + [f'{p}_{c}' for p in ['Pk', 'xi_malla'] for c in CAMPOS]
    if cfg['aleatorios'] == 'nuevos':
        por_semilla.append('RR')
    d = {k: bloq[0][k] for k in bloq[0].files if k not in por_semilla}
    d.update({k: np.concatenate([b[k] for b in bloq]) for k in por_semilla})
    return d


def rebinear(X, r_edges, ancho=5, r_min=20):
    """Suma bins finos (último eje) en bins de `ancho` Mpc/h desde r_min."""
    i0 = int(np.where(r_edges == r_min)[0][0])
    nb = (len(r_edges) - 1 - i0)//ancho
    return X[..., i0:i0 + nb*ancho].reshape(*X.shape[:-1], nb, ancho).sum(-1)


EST = {'PH':  lambda DD, DR, RR: DD/RR - 1,
       'DP':  lambda DD, DR, RR: DD/DR - 1,
       'Ham': lambda DD, DR, RR: DD*RR/DR**2 - 1,
       'LS':  lambda DD, DR, RR: (DD - 2*DR + RR)/RR}


def estimadores(d, iA, ia, ancho=5, r_min=20):
    """Los cuatro estimadores (N, bins) para la amplitud iA y alpha ia de un dict de cargar()."""
    ND = d['ND'][:, iA][:, None]
    NR = d['NR'][ia] if 'NR' in d else d['alphas'][ia]*int(d['nbar']*d['L']**3)   # aleatorios nuevos
    DD = rebinear(d['DD'][:, iA], d['r_edges'], ancho, r_min)/(ND*(ND - 1))
    DR = rebinear(d['DR'][:, iA, ia], d['r_edges'], ancho, r_min)/(ND*NR)
    RR = rebinear(d['RR'][..., ia, :], d['r_edges'], ancho, r_min)/(NR*(NR - 1))
    return {e: f(DD, DR, RR) for e, f in EST.items()}


# ---------------------------------------------------------------- 6. locks (reparto dinámico)

LATIDO = 120             # s entre renovaciones del lock
ESPERA_LOCK = 3*LATIDO   # s sin renovar para considerar abandonado un lock


def mi_id():
    """Identidad de la tarea; la misma tras un --requeue, distinta entre tareas."""
    return f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'local'))}_" \
           f"{os.environ.get('SLURM_ARRAY_TASK_ID', socket.gethostname())}"


def tomar(ruta):
    """Intenta tomar el bloque; True si ahora es nuestro."""
    lock = Path(ruta + '.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            with open(lock, 'x') as f:
                f.write(f"{mi_id()} {os.environ.get('SLURM_RESTART_COUNT', '0')} {socket.gethostname()} {os.getpid()}\n")
            return True
        except FileExistsError:
            try:
                dueno = lock.read_text().split()[0]
                viejo = time.time() - lock.stat().st_mtime > ESPERA_LOCK
            except (FileNotFoundError, IndexError):
                continue
            if dueno == mi_id() or viejo:          # encarnación anterior de esta tarea, o lock abandonado
                lock.unlink(missing_ok=True)
                continue
            return False
    return False


@contextmanager
def latido(lock):
    """Renueva el lock cada LATIDO segundos mientras se calcula el bloque."""
    fin = threading.Event()

    def latir():
        while not fin.wait(LATIDO):
            try:
                os.utime(lock)
            except FileNotFoundError:
                pass

    hilo = threading.Thread(target=latir, daemon=True)
    hilo.start()
    try:
        yield
    finally:
        fin.set()
        hilo.join()


def estado():
    print(f"{'configuración':20s} {'bloques':>9s} {'semillas':>13s} {'en curso':>9s}")
    for cfg in CONFIGS:
        bs = bloques([cfg['nombre']])
        hechos = [(s0, s1) for _, s0, s1 in bs if bloque_valido(cfg, s0, s1)]
        curso = sum(os.path.exists(ruta_bloque(cfg, s0, s1) + '.lock') for _, s0, s1 in bs)
        n = sum(s1 - s0 for s0, s1 in hechos)
        print(f"{cfg['nombre']:20s} {len(hechos):4d}/{len(bs):<4d} {n:6d}/{cfg['N']:<6d} {curso:9d}")


def main():
    global DATOS, MUESTREO
    p = argparse.ArgumentParser()
    p.add_argument('--procesos', type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument('--estado', action='store_true')
    p.add_argument('--prueba', nargs=3, metavar=('CONFIG', 'S0', 'S1'))
    p.add_argument('--independientes', action='store_true', help='muestreo de Poisson del notebook (solo con --prueba)')
    p.add_argument('--salida', help='carpeta de datos (default: datos/, o datos_prueba/ con --prueba o --hasta)')
    p.add_argument('--solo', nargs='+', choices=[c['nombre'] for c in CONFIGS], help='solo estas configuraciones')
    p.add_argument('--hasta', type=int, help='solo semillas < HASTA (pruebas; escribe en datos_prueba/)')
    args = p.parse_args()

    prueba = args.prueba or args.hasta
    DATOS = os.path.abspath(args.salida or os.path.join(AQUI, 'datos_prueba' if prueba else 'datos'))
    if args.independientes:
        if not args.prueba:
            p.error('--independientes solo se usa con --prueba, para verificar contra la corrida N50000')
        MUESTREO = 'independiente'

    if args.estado:
        estado()
        return

    if args.prueba:
        cfg = config(args.prueba[0])
        preparar(cfg)
        with Pool(args.procesos) as pool:
            correr_bloque(cfg, int(args.prueba[1]), int(args.prueba[2]), pool)
        return

    print(f'tarea {mi_id()} (reinicio {os.environ.get("SLURM_RESTART_COUNT", "0")}), {args.procesos} procesos, '
          f'datos en {DATOS}', flush=True)
    activa, pool = None, None
    # se recorre la lista hasta que en una pasada ya no se toma nada: un bloque con lock de otra tarea
    # se vuelve a intentar en la siguiente pasada (por si su lock queda abandonado)
    tomado = True
    while tomado:
        tomado = False
        for cfg, s0, s1 in bloques(args.solo, args.hasta):
            ruta = ruta_bloque(cfg, s0, s1)
            if bloque_valido(cfg, s0, s1) or not tomar(ruta):
                continue                                # ya está, o lo está calculando otra tarea
            tomado = True
            try:
                if os.path.exists(ruta):                # existe pero no es válido
                    apartar(cfg, s0, s1)
                if cfg is not activa:                   # cambio de configuración: nuevo fork con su estado
                    if pool:
                        pool.close()
                        pool.join()
                    preparar(cfg)
                    pool, activa = Pool(args.procesos), cfg
                with latido(ruta + '.lock'):
                    correr_bloque(cfg, s0, s1, pool)
            finally:
                Path(ruta + '.lock').unlink(missing_ok=True)
    if pool:
        pool.close()
        pool.join()
    print(f'tarea {mi_id()}: no quedan bloques libres', flush=True)


if __name__ == '__main__':
    main()
