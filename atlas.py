"""
Atlas de Distribuciones de Probabilidad — Unidad 3
"Un día en la vida de un servidor"

Ejecutar:
    pip install -r requirements.txt
    streamlit run atlas.py
"""
import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Atlas de Distribuciones", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ════════════════════════════ Paleta y estilo ════════════════════════════
C = dict(disc="#6F96FF", cont="#F09A4E", aprox="#AC8AF2", samp="#3FC5B1",
         ink="#E4EAF2", muted="#93A0B2", grid="#1F2B39", line="#283545", bg="#0E141C", panel="#16202C",
         ok="#4CC27E", warn="#E2B23C", bad="#F07A7A", sim="#5E6D82")
PINS = ["#F07A7A", "#3FC5B1", "#E2B23C", "#AC8AF2"]
FAMS = {"disc": ("Discreta", C["disc"]), "cont": ("Continua", C["cont"]),
        "aprox": ("Aproximación", C["aprox"]), "samp": ("De muestreo", C["samp"]), "none": ("", C["ink"])}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }
.block-container { padding-top: 3.6rem; max-width: 1500px; }
h1, h2, h3 { font-family: "Bricolage Grotesque", "Segoe UI", sans-serif !important; letter-spacing: -.01em; }
.log { font-family: "IBM Plex Mono", monospace; font-size: 13px; background: #16202C; border: 1px solid #283545;
       border-radius: 8px; padding: 8px 12px; color: #93A0B2; margin-bottom: 6px; }
.log b { color: #E4EAF2; font-weight: 500; } .log .lvl { color: var(--fam); font-weight: 500; }
.ptitle { font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; font-size: 44px; line-height: 1.05;
          letter-spacing: -.02em; color: #E4EAF2; margin: 6px 0 4px; }
.badge { font-family: "IBM Plex Mono", monospace; font-size: 12px; letter-spacing: .06em; text-transform: uppercase;
         padding: 3px 9px; border-radius: 999px; border: 1px solid var(--fam); color: var(--fam);
         vertical-align: middle; margin-left: 10px; font-weight: 500; }
.lead { font-size: 16.5px; max-width: 80ch; color: #E4EAF2; margin-bottom: 8px; }
.eyebrow { font-family: "IBM Plex Mono", monospace; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #93A0B2; }
.effect { border: 1px dashed var(--fam); border-radius: 10px; padding: 10px 12px; font-size: 14px; }
.effect .eyebrow { color: var(--fam); } .delta { font-family: "IBM Plex Mono", monospace; font-size: 12px; color: #93A0B2; }
.big { font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; font-size: 40px; color: var(--fam); line-height: 1; }
.calc { font-family: "IBM Plex Mono", monospace; font-size: 13px; color: #93A0B2; }
.calc b { color: #E4EAF2; font-weight: 500; }
.chip { display: inline-block; font-family: "IBM Plex Mono", monospace; font-size: 12.5px; border-radius: 999px;
        padding: 2px 10px; margin: 0 6px 6px 0; border: 1px solid; }
.chip.ok { color: #4CC27E; border-color: #4CC27E; } .chip.no { color: #F07A7A; border-color: #F07A7A; }
.burst i { display: inline-block; width: 18px; height: 18px; border-radius: 4px; margin: 2px; }
.rule { background: #1B2734; border-radius: 10px; padding: 12px 14px; font-size: 15px; }
.slot-hr { font-family: "IBM Plex Mono", monospace; font-size: 13px; color: #93A0B2; }
.slot-tag { font-family: "IBM Plex Mono", monospace; font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em; }
.slot-name { font-family: "Bricolage Grotesque", sans-serif; font-size: 20px; font-weight: 600; color: #E4EAF2; }
.kpi b { font-family: "Bricolage Grotesque", sans-serif; font-size: 34px; display: block; line-height: 1; }
.kpi span { font-size: 12.5px; color: #93A0B2; }
[data-testid="stMetricValue"] { font-family: "IBM Plex Mono", monospace; font-size: 22px; }
section[data-testid="stSidebar"] .stRadio label p { font-family: "IBM Plex Mono", monospace; font-size: 13.5px; }
</style>
""", unsafe_allow_html=True)


def fmt(v, d=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "no existe"
    if v == math.inf:
        return "∞"
    if v != 0 and abs(v) < 1e-4:
        m, e = f"{v:.2e}".split("e")
        return f"{m}·10^{int(e)}"
    s = f"{v:.{d}f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def pct(v):
    return f"{v * 100:.2f} %"


# ════════════════════════════ Fichas de distribuciones ════════════════════════════
def P(k, sym, label, mn, mx, step, v, fam, meaning, effect, maxk=None):
    return dict(k=k, sym=sym, label=label, min=mn, max=mx, step=step, v=v, fam=fam, meaning=meaning, effect=effect, maxk=maxk)


def t_sample(p, n):
    v = p["v"]; x = np.random.normal(size=(n, v + 1))
    return x.mean(1) / (x.std(1, ddof=1) / np.sqrt(v + 1))


def chi_sample(p, n):
    k = p["k"]; x = np.random.normal(size=(n, k + 1))
    return k * x.var(1, ddof=1)


def f_sample(p, n):
    a = np.random.normal(size=(n, p["d1"] + 1)); b = np.random.normal(size=(n, p["d2"] + 1))
    return a.var(1, ddof=1) / b.var(1, ddof=1)


def f_range(p):
    q = stats.f(p["d1"], p["d2"]).ppf(0.985)
    return (0.0, min(max(3.0, q * 1.1), 14.0))


D = {
 "bernoulli": dict(
    name="Bernoulli", fam="disc", disc=True, xlabel="x  (0 = íntegro · 1 = con error)", xdesc="el resultado del paquete (1 = con error)",
    params=[P("p", "p", "prob. de que el paquete llegue con error", 0.01, 0.99, 0.01, 0.2, [0.1, 0.5, 0.8],
              "Probabilidad de “éxito”: el evento que contamos (aquí, que el paquete llegue con error).",
              "Al aumentar p, la barra de X = 1 sube y la de X = 0 baja lo mismo (siempre suman 1). La varianza p(1 − p) es máxima en p = 0.5, el caso de mayor incertidumbre.")],
    dist=lambda p: stats.bernoulli(p["p"]), support=lambda p: (0, 1),
    lead="Un único ensayo con solo dos resultados posibles: éxito (X = 1) con probabilidad p, o fracaso (X = 0) con probabilidad 1 − p.",
    formula=r"P(X=x)=p^{x}(1-p)^{1-x},\qquad x\in\{0,1\}", moments=r"E[X]=p \qquad V[X]=p(1-p)",
    chars=["Discreta: solo toma los valores 0 y 1.", "Es el bloque básico: la Binomial es la suma de n Bernoulli independientes.",
           "“Éxito” es el evento que se cuenta, aunque sea algo malo (un error, una pérdida).", "La varianza nunca supera 0.25."],
    app="Cada paquete que llega se verifica con su checksum: X = 1 si llega con error y X = 0 si llega íntegro. En un enlace con 2 % de error, X ~ Bernoulli(0.02).",
    more=["Un ping recibe respuesta o se pierde por timeout.", "Un test unitario pasa o falla.", "Un intento de inicio de sesión es válido o no."],
    when="Hay un solo intento y solo dos resultados posibles.", careful="Si el ensayo se repite n veces y se cuentan los éxitos, ya no es Bernoulli: es Binomial.",
    prob=("eq", 1, 1)),
 "binomial": dict(
    name="Binomial", fam="disc", disc=True, xlabel="x = paquetes con error", xdesc="la cantidad de paquetes con error",
    params=[P("n", "n", "paquetes enviados (ensayos)", 1, 60, 1, 20, [5, 20, 50], "Número de ensayos independientes (paquetes enviados).",
              "Al aumentar n la distribución se desplaza a la derecha (media np) y se ensancha (varianza np(1 − p)). Con n grande toma forma de campana."),
            P("p", "p", "prob. de error de cada paquete", 0.01, 0.99, 0.01, 0.15, [0.1, 0.5, 0.8], "Probabilidad de éxito en cada ensayo; es la misma en todos.",
              "Con p = 0.5 la forma es simétrica. Con p < 0.5 se concentra a la izquierda con cola a la derecha; con p > 0.5 ocurre lo contrario.")],
    dist=lambda p: stats.binom(p["n"], p["p"]), support=lambda p: (0, p["n"]),
    lead="Cuenta el número de éxitos en n ensayos independientes de Bernoulli, todos con la misma probabilidad de éxito p.",
    formula=r"P(X=x)=\binom{n}{x}p^{x}(1-p)^{n-x},\qquad x=0,1,\dots,n", moments=r"E[X]=np \qquad V[X]=np(1-p)",
    chars=["Discreta, con valores de 0 a n (tiene tope).", "Ensayos independientes y p constante: equivale a muestrear con reemplazo.",
           "Simétrica si p = 0.5; sesgada en otro caso.", "Con n grande se aproxima a la Normal; con n grande y p pequeña, a la Poisson."],
    app="Se envían n = 20 paquetes por un enlace con 15 % de pérdida. X = paquetes perdidos. P(X ≥ 5) es la probabilidad de tener que retransmitir 5 o más.",
    more=["De 50 builds diarios, cuántos fallan si cada uno falla con p = 0.04.", "Bits erróneos en una trama de n bits con tasa de error de bit (BER) p.", "Cuántos de 10 discos de un RAID fallan en el año."],
    when="Hay un número fijo n de intentos independientes, cada uno con éxito o fracaso y la misma p.",
    careful="Si se extrae de un lote pequeño sin devolver, p cambia en cada extracción: usar Hipergeométrica.", prob=("ge", 5, 5)),
 "poisson": dict(
    name="Poisson", fam="disc", disc=True, xlabel="x = peticiones en 1 segundo", xdesc="el número de peticiones en un segundo",
    params=[P("l", "λ", "peticiones promedio por segundo", 0.5, 30.0, 0.5, 4.0, [1.0, 4.0, 10.0],
              "Tasa promedio de eventos por intervalo (aquí, peticiones por segundo). Es a la vez la media y la varianza.",
              "Al aumentar λ la distribución se desplaza a la derecha y se ensancha (media = varianza = λ). Con λ pequeño es muy asimétrica; con λ ≥ 10 ya se parece a una campana.")],
    dist=lambda p: stats.poisson(p["l"]), support=lambda p: (0, int(math.ceil(p["l"] + 6 * math.sqrt(p["l"]) + 4))),
    lead="Cuenta cuántos eventos ocurren en un intervalo fijo (de tiempo o espacio) cuando ocurren de forma independiente y a una tasa promedio constante λ.",
    formula=r"P(X=x)=\frac{e^{-\lambda}\lambda^{x}}{x!},\qquad x=0,1,2,\dots", moments=r"E[X]=\lambda \qquad V[X]=\lambda",
    chars=["Discreta y sin tope superior.", "Media igual a la varianza: es la forma más rápida de reconocerla en datos.",
           "La suma de Poisson independientes es Poisson (λ₁ + λ₂).", "Es el límite de la Binomial cuando n es grande y p pequeña (λ = np)."],
    app="Un servidor recibe en promedio λ = 4 peticiones por segundo. Si solo puede atender 8 por segundo, P(X ≥ 9) es la probabilidad de saturarse en un segundo dado.",
    more=["Correos que llegan al servidor SMTP por minuto.", "Errores que aparecen en el log por hora.", "Defectos por cada 1 000 líneas de código."],
    when="Se cuenta “cuántos eventos” en un intervalo continuo y solo se conoce la tasa promedio.",
    careful="Si la pregunta es “cuánto tiempo pasa hasta el próximo evento”, la distribución es la Exponencial.", prob=("ge", 9, 9)),
 "uniforme": dict(
    name="Uniforme continua", fam="cont", disc=False, xlabel="x = espera antes de reintentar (ms)", xdesc="la espera antes de reintentar (ms)",
    params=[P("a", "a", "espera mínima (ms)", 0, 900, 10, 100, [0, 100, 250], "Límite inferior del intervalo.",
              "Mover a cambia el inicio del intervalo. Si el intervalo se reduce, la altura 1/(b − a) sube, porque el área total siempre es 1."),
            P("b", "b", "espera máxima (ms)", 10, 1000, 10, 400, [300, 500, 800], "Límite superior del intervalo.",
              "Al ampliar b el rectángulo se alarga y baja de altura. La media (a + b)/2 se mueve al centro y la varianza (b − a)²/12 crece.")],
    dist=lambda p: stats.uniform(p["a"], p["b"] - p["a"]),
    support=lambda p: (p["a"] - 0.18 * (p["b"] - p["a"]), p["b"] + 0.18 * (p["b"] - p["a"])),
    lead="Todos los valores del intervalo [a, b] son igualmente probables: la densidad es un rectángulo de altura 1/(b − a).",
    formula=r"f(x)=\frac{1}{b-a},\quad a\le x\le b \qquad F(x)=\frac{x-a}{b-a}", moments=r"E[X]=\frac{a+b}{2}\qquad V[X]=\frac{(b-a)^2}{12}",
    chars=["Continua: P(X = x) = 0; solo tienen probabilidad los intervalos.", "La probabilidad de un subintervalo es su longitud dividida entre (b − a).",
           "Simétrica; no tiene una moda única.", "numpy.random.rand() genera una U(0, 1): es la base de toda simulación."],
    app="Jitter en reintentos: si un servicio cae, cada cliente espera un tiempo aleatorio U(100, 400) ms antes de reintentar. Así no golpean al servidor todos al mismo tiempo.",
    more=["Generadores de números aleatorios: U(0, 1).", "Instante en que llega un paquete dentro de una ventana de 10 ms sin más información.", "Retardo aleatorio de arranque de tareas programadas (cron con jitter)."],
    when="Se sabe que el valor está entre a y b y no hay razón para preferir ninguna zona.",
    careful="Si los valores son enteros (servidor 1…8, un dado), usar la Uniforme discreta.", prob=("le", 100, 200), mode_text="todo [a, b]"),
 "exponencial": dict(
    name="Exponencial", fam="cont", disc=False, xlabel="x = segundos hasta la siguiente petición", xdesc="el tiempo hasta la siguiente petición (s)",
    params=[P("l", "λ", "eventos por segundo (tasa)", 0.1, 5.0, 0.1, 1.0, [0.5, 1.0, 2.0], "Tasa de ocurrencia de eventos. El tiempo medio de espera es 1/λ.",
              "Con λ grande los eventos son frecuentes: la curva arranca más alta y cae más rápido, y la espera media 1/λ se acorta. Con λ pequeño la curva es baja y con cola larga.")],
    dist=lambda p: stats.expon(scale=1 / p["l"]), support=lambda p: (0.0, 6 / p["l"]),
    lead="Modela el tiempo de espera hasta el siguiente evento en un proceso de Poisson: el tiempo entre peticiones, o el tiempo hasta que falla un componente.",
    formula=r"f(x)=\lambda e^{-\lambda x},\quad x\ge 0 \qquad F(x)=1-e^{-\lambda x}", moments=r"E[X]=\frac{1}{\lambda}\qquad V[X]=\frac{1}{\lambda^2}",
    chars=["Continua, solo valores positivos; la moda está en 0.", "Sin memoria: P(X > s + t | X > s) = P(X > t).", "Muy asimétrica a la derecha.", "Confiabilidad: R(t) = P(X > t) = e^(−λt)."],
    app="Si llegan λ = 1 peticiones por segundo, el tiempo entre una petición y la siguiente es Exponencial(1): en promedio 1 s, pero hay ≈ 5 % de probabilidad de que el servidor quede ocioso más de 3 s.",
    more=["Tiempo hasta la falla de un disco con MTBF = 1/λ horas.", "Duración de una sesión o de una llamada en un call center.", "Tiempo entre caídas de un servicio."],
    when="La pregunta es “cuánto tiempo pasa” hasta un evento que ocurre a tasa constante.",
    careful="Si la pregunta es “cuántos eventos en un intervalo”, es Poisson. No sirve para piezas que se desgastan con la edad.", prob=("ge", 3.0, 3.0)),
 "normal": dict(
    name="Normal y Normal estándar", fam="cont", disc=False, critical=True, crit_tail="two", xlabel="x = latencia (ms)", xdesc="la latencia (ms)",
    params=[P("mu", "μ", "latencia media (ms)", 0, 100, 1, 40, [25, 40, 60], "Media: centro de la campana. Coincide con la mediana y la moda.",
              "Cambiar μ desplaza la campana a la izquierda o derecha sin cambiar su forma ni su altura."),
            P("s", "σ", "desviación estándar (ms)", 1.0, 25.0, 0.5, 8.0, [4.0, 8.0, 15.0], "Desviación estándar: mide la dispersión alrededor de μ.",
              "Con σ grande la campana se aplana y se ensancha (más variabilidad); con σ pequeña es alta y angosta. El área total siempre es 1.")],
    dist=lambda p: stats.norm(p["mu"], p["s"]), support=lambda p: (p["mu"] - 4 * p["s"], p["mu"] + 4 * p["s"]),
    lead="Variable continua simétrica en forma de campana alrededor de la media μ. Aparece cuando un valor es la suma de muchos efectos pequeños (Teorema Central del Límite). La Normal estándar Z tiene μ = 0 y σ = 1.",
    formula=r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}} \qquad Z=\frac{X-\mu}{\sigma}\sim N(0,1)",
    moments=r"E[X]=\mu \qquad V[X]=\sigma^2 \qquad P(\mu-\sigma<X<\mu+\sigma)\approx 68.27\%",
    chars=["Simétrica: media = mediana = moda.", "Regla empírica: 68.27 % en μ ± σ, 95.45 % en μ ± 2σ y 99.73 % en μ ± 3σ.",
           "Estandarizar con Z = (X − μ)/σ permite usar una sola tabla para cualquier normal.", "Es el límite de Binomial y Poisson para parámetros grandes."],
    app="La latencia (ping) de srv-01 sigue N(40, 8) ms. Una latencia de 64 ms tiene z = 3: está fuera del 99.73 % habitual y el sistema de monitoreo la marca como anómala.",
    more=["Tiempo de respuesta promedio de una API.", "Error de medición de sensores en IoT.", "Detección de anomalías por puntaje z (|z| > 3)."],
    when="La variable es continua, simétrica y se agrupa alrededor de un valor típico.",
    careful="No sirve para variables muy asimétricas (tiempos de espera: Exponencial) ni para conteos pequeños.", prob=("ge", 56.0, 56.0)),
 "hipergeometrica": dict(
    name="Hipergeométrica", fam="disc", disc=True, xlabel="x = módulos con bug en la muestra", xdesc="el número de módulos con bug en la muestra",
    params=[P("N", "N", "módulos en el release (población)", 2, 100, 1, 50, [15, 50, 100], "Tamaño de la población finita.",
              "Si N crece mucho frente a n, sacar sin reemplazo casi no cambia la proporción y la distribución se parece a la Binomial."),
            P("K", "K", "módulos con bug en el release", 0, 100, 1, 6, [3, 6, 15], "Número de éxitos en la población.",
              "Al aumentar K la distribución se desplaza a la derecha: la media nK/N crece.", maxk="N"),
            P("n", "n", "módulos revisados (muestra)", 1, 100, 1, 10, [5, 10, 25], "Tamaño de la muestra, tomada sin reemplazo.",
              "Una muestra más grande detecta más bugs en promedio. Si n se acerca a N, la varianza baja por el factor (N − n)/(N − 1).", maxk="N")],
    dist=lambda p: stats.hypergeom(p["N"], p["K"], p["n"]), support=lambda p: (0, max(1, min(p["n"], p["K"]))),
    lead="Cuenta los éxitos en una muestra de n elementos tomada sin reemplazo de una población finita de N, donde hay K éxitos.",
    formula=r"P(X=x)=\frac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}}",
    moments=r"E[X]=n\frac{K}{N}\qquad V[X]=n\frac{K}{N}\left(1-\frac{K}{N}\right)\frac{N-n}{N-1}",
    chars=["Discreta, entre max(0, n − N + K) y min(n, K).", "Sin reemplazo: la probabilidad cambia después de cada extracción.",
           "Misma media que la Binomial(n, K/N), pero menor varianza.", "Si n/N ≤ 5 %, se puede aproximar con la Binomial."],
    app="QA revisa n = 10 módulos de un release de N = 50, de los cuales K = 6 tienen bugs. P(X ≥ 1) es la probabilidad de que la revisión encuentre al menos un módulo defectuoso.",
    more=["Auditoría: revisar 20 de 200 registros del log buscando accesos no autorizados.", "Control de calidad de un lote de placas o memorias.", "Muestra de cuentas que incluya cuentas comprometidas."],
    when="Se extrae de una población finita y pequeña, sin devolver los elementos.",
    careful="Si la población es enorme o hay reemplazo, la Binomial da casi el mismo resultado y es más simple.", prob=("ge", 1, 1)),
 "uniforme-discreta": dict(
    name="Uniforme discreta", fam="disc", disc=True, xlabel="x = servidor asignado", xdesc="el número del servidor asignado",
    params=[P("a", "a", "primer servidor", 0, 20, 1, 1, [1, 3, 6], "Valor entero mínimo.",
              "Mover a desplaza el inicio. Con menos valores posibles, cada uno tiene más probabilidad: 1/(b − a + 1)."),
            P("b", "b", "último servidor", 1, 30, 1, 8, [4, 8, 16], "Valor entero máximo.",
              "Al agregar valores (b mayor) cada barra baja, porque la probabilidad se reparte entre más servidores.")],
    dist=lambda p: stats.randint(p["a"], p["b"] + 1), support=lambda p: (p["a"], p["b"]),
    lead="Hay k = b − a + 1 valores enteros posibles y todos tienen la misma probabilidad 1/k.",
    formula=r"P(X=x)=\frac{1}{b-a+1},\qquad x=a,a+1,\dots,b", moments=r"E[X]=\frac{a+b}{2}\qquad V[X]=\frac{(b-a+1)^2-1}{12}",
    chars=["Discreta, valores enteros consecutivos.", "Todas las barras tienen la misma altura.", "Simétrica alrededor de (a + b)/2.", "Es la forma de “elegir al azar” entre opciones equivalentes."],
    app="Balanceo aleatorio: el balanceador asigna cada petición a uno de los servidores 1…8 con probabilidad 1/8. En promedio cada servidor recibe el 12.5 % del tráfico.",
    more=["Elegir un puerto efímero al azar.", "Índice aleatorio para muestrear filas de una tabla.", "Tirar un dado o elegir una réplica al azar."],
    when="Los resultados son enteros y no hay razón para preferir ninguno.",
    careful="Si el valor puede ser cualquier número real del intervalo, es la Uniforme continua.", prob=("le", 1, 3), mode_text="todos"),
 "t": dict(
    name="t de Student", fam="samp", disc=False, critical=True, crit_tail="two", xlabel="t", xdesc="el estadístico T",
    params=[P("v", "ν", "grados de libertad (n − 1)", 1, 60, 1, 9, [1, 4, 30], "Grados de libertad ν = n − 1, donde n es el tamaño de la muestra.",
              "Con pocos grados de libertad las colas son más gruesas: hay más probabilidad de valores extremos porque σ se estima con s. Al aumentar ν la curva se acerca a la Normal estándar; con ν ≥ 30 son casi iguales.")],
    dist=lambda p: stats.t(p["v"]), support=lambda p: (-8.0, 8.0) if p["v"] < 3 else (-5.0, 5.0), sampler=t_sample,
    sim_note="Fábrica de muestras: cada repetición toma n = ν + 1 latencias de una población normal, calcula t = (x̄ − μ)/(s/√n) y lo agrega al histograma.",
    lead="Describe el estadístico t = (x̄ − μ)/(s/√n) cuando la población es normal, σ es desconocida y se estima con s. Se usa sobre todo con muestras pequeñas.",
    formula=r"T=\frac{\bar X-\mu}{S/\sqrt{n}}\sim t_{\nu},\qquad \nu=n-1", moments=r"E[T]=0\ (\nu>1)\qquad V[T]=\frac{\nu}{\nu-2}\ (\nu>2)",
    chars=["Simétrica y centrada en 0, como la Z.", "Colas más gruesas que la Normal; depende de ν.", "Cuando ν → ∞, t → N(0, 1).", "Base de intervalos de confianza y pruebas para la media con σ desconocida."],
    app="A/B testing: se mide el tiempo de respuesta de la nueva versión de la API 10 veces. Con la t (ν = 9) se construye el intervalo de confianza de la media o se prueba si mejoró respecto a la versión anterior.",
    more=["Intervalo de confianza para la latencia media con pocas mediciones.", "Comparar el rendimiento de dos algoritmos con 8 ejecuciones cada uno.", "Prueba t pareada: mismo servidor antes y después de una optimización."],
    when="Se infiere sobre una media, σ es desconocida y la muestra es pequeña.", careful="Con σ conocida o n grande se usa directamente la Normal estándar Z.", prob=("ge", 2.0, 2.0)),
 "chi2": dict(
    name="Chi-cuadrado", fam="samp", disc=False, critical=True, crit_tail="right", xlabel="χ²", xdesc="el estadístico χ²",
    params=[P("k", "k", "grados de libertad", 1, 60, 1, 5, [2, 5, 15], "Grados de libertad k: cuántos cuadrados independientes se suman (en muestreo, n − 1).",
              "Con k pequeño es muy asimétrica a la derecha. Al aumentar k se desplaza a la derecha (media k), se ensancha (varianza 2k) y se vuelve más simétrica, parecida a una normal.")],
    dist=lambda p: stats.chi2(p["k"]), support=lambda p: (0.0, p["k"] + 5 * math.sqrt(2 * p["k"]) + 2), sampler=chi_sample,
    sim_note="Fábrica de muestras: cada repetición toma n = k + 1 valores de N(0, 1), calcula (n − 1)s²/σ² y lo agrega al histograma.",
    lead="Es la suma de k normales estándar al cuadrado. En muestreo, (n − 1)S²/σ² sigue una χ² con n − 1 grados de libertad. Se usa para inferencias sobre la varianza y en pruebas de bondad de ajuste e independencia.",
    formula=r"\chi^2=Z_1^2+\cdots+Z_k^2 \qquad \frac{(n-1)S^2}{\sigma^2}\sim\chi^2_{n-1}", moments=r"E[\chi^2]=k\qquad V[\chi^2]=2k",
    chars=["Continua, solo valores positivos.", "Asimétrica a la derecha; se simetriza al crecer k.", "Suma de χ² independientes es χ² (se suman los grados de libertad).", "Es la base de la t (denominador) y de la F (cociente de dos χ²)."],
    app="Bondad de ajuste: se generan 6 000 números con el generador aleatorio del sistema, se cuentan por categorías y la χ² indica si se apartan de la uniforme más de lo esperable por azar.",
    more=["Verificar si la varianza de la latencia supera la permitida por el SLA.", "Probar si el tipo de error es independiente del navegador (tabla de contingencia).", "Comprobar si las peticiones se reparten por igual entre réplicas."],
    when="Se analiza una varianza o se comparan frecuencias observadas con esperadas.", careful="Es muy sensible a que la población sea normal cuando se usa para la varianza.", prob=("ge", 11.0, 11.0)),
 "f": dict(
    name="F de Fisher", fam="samp", disc=False, critical=True, crit_tail="right", xlabel="F", xdesc="el estadístico F",
    params=[P("d1", "d₁", "g.l. del numerador (n₁ − 1)", 1, 60, 1, 8, [2, 8, 30], "Grados de libertad de la varianza del numerador.", "Al aumentar d₁ el pico se aleja de 0 y la curva se concentra alrededor de 1."),
            P("d2", "d₂", "g.l. del denominador (n₂ − 1)", 1, 60, 1, 12, [4, 12, 40], "Grados de libertad de la varianza del denominador.", "Al aumentar d₂ la cola derecha se adelgaza y la media d₂/(d₂ − 2) se acerca a 1.")],
    dist=lambda p: stats.f(p["d1"], p["d2"]), support=f_range, sampler=f_sample,
    sim_note="Fábrica de muestras: cada repetición toma dos muestras normales con la misma σ (n₁ = d₁ + 1 y n₂ = d₂ + 1), calcula F = s₁²/s₂² y lo agrega al histograma.",
    lead="Es el cociente de dos χ² independientes, cada una dividida entre sus grados de libertad. En muestreo, F = S₁²/S₂² cuando las dos poblaciones tienen la misma varianza. Se usa para comparar varianzas y en ANOVA.",
    formula=r"F=\frac{\chi^2_1/d_1}{\chi^2_2/d_2}=\frac{S_1^2}{S_2^2}\ \ (\sigma_1=\sigma_2)", moments=r"E[F]=\frac{d_2}{d_2-2}\ (d_2>2)",
    chars=["Continua, solo valores positivos, asimétrica a la derecha.", "Tiene dos parámetros: los grados de libertad del numerador y del denominador.", "Si F ~ F(d₁, d₂), entonces 1/F ~ F(d₂, d₁).", "Si T ~ t(ν), entonces T² ~ F(1, ν)."],
    app="Dos servidores atienden la misma carga: se mide la varianza del tiempo de respuesta de cada uno (9 y 13 mediciones). Si F = s₁²/s₂² cae en la región de rechazo, uno es significativamente más inestable.",
    more=["ANOVA: comparar el rendimiento medio de 3 algoritmos de ordenamiento.", "Comparar la variabilidad del throughput de dos proveedores de red.", "Validar si un modelo de regresión explica el uso de CPU."],
    when="Se comparan dos varianzas o varias medias (ANOVA).", careful="El orden importa: numerador y denominador no se intercambian sin invertir los grados de libertad.", prob=("ge", 3.0, 3.0)),
}

NAV = [
    ("inicio", "07:59", "El día del servidor", "none", "srv-01 arranca. Servicios en línea, monitoreo activo."),
    ("bernoulli", "08:00", "Bernoulli", "disc", "Llega el primer paquete del día. ¿Llega íntegro o con error?"),
    ("binomial", "08:30", "Binomial", "disc", "Llegan 20 paquetes por el mismo enlace. ¿Cuántos llegan con error?"),
    ("multinomial", "09:00", "Multinomial", "disc", "El servidor web responde 12 peticiones; cada respuesta es 2xx, 4xx o 5xx."),
    ("poisson", "10:00", "Poisson", "disc", "Empieza la hora pico: llegan en promedio 4 peticiones por segundo."),
    ("uniforme", "11:00", "Uniforme continua", "cont", "Falla un servicio externo; los clientes reintentan tras un jitter aleatorio entre 100 y 400 ms."),
    ("exponencial", "12:00", "Exponencial", "cont", "¿Cuánto tiempo pasa entre una petición y la siguiente?"),
    ("normal", "14:00", "Normal y Z", "cont", "Monitoreo de latencia: los pings rondan 40 ms con desviación de 8 ms."),
    ("hipergeometrica", "16:00", "Hipergeométrica", "disc", "QA revisa 10 módulos de un release de 50; 6 tienen bugs."),
    ("uniforme-discreta", "17:00", "Uniforme discreta", "disc", "El balanceador asigna cada petición a uno de 8 servidores al azar."),
    ("aproximaciones", "18:00", "Aproximaciones a la Normal", "aprox", "Pico de tráfico: 5 000 paquetes. La binomial exacta es pesada; se aproxima."),
    ("muestreo", "20:00", "t, χ² y F", "samp", "Reporte nocturno: con muestras pequeñas se comparan versiones, varianzas y servidores."),
    ("duelos", "22:00", "Duelos", "none", "Revisión del día: distribuciones que se parecen y cómo distinguirlas."),
]
NAV_IDS = [n[0] for n in NAV]
NAVD = {n[0]: n for n in NAV}

# ════════════════════════════ Utilidades de gráficas ════════════════════════════
def base_fig(xlabel="", ylabel="", height=430):
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=60, r=20, t=36, b=50),
                      font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=C["ink"], size=13),
                      xaxis=dict(title=xlabel, gridcolor=C["grid"], zeroline=False, linecolor=C["line"]),
                      yaxis=dict(title=ylabel, gridcolor=C["grid"], zeroline=False, rangemode="tozero"),
                      legend=dict(orientation="h", y=-0.2, x=0, font=dict(family="IBM Plex Mono, monospace", size=12)),
                      bargap=0.25, hoverlabel=dict(font_family="IBM Plex Mono, Consolas, monospace"), barmode="overlay")
    return fig


def _dark(fig):
    return fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


def show(fig, key=None):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def hexa(color, a):
    color = color.lstrip("#"); r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
    return f"rgba({r},{g},{b},{a})"


def ptext(spec, p):
    return ", ".join(f"{q['sym']} = {fmt(p[q['k']], 3)}" for q in spec["params"])


def mstats(spec, p):
    d = spec["dist"](p)
    m, v = (float(x) for x in d.stats(moments="mv"))
    if spec.get("name") == "t de Student":
        m = 0.0 if p["v"] > 1 else float("nan")
        v = p["v"] / (p["v"] - 2) if p["v"] > 2 else (math.inf if p["v"] > 1 else float("nan"))
    return m, v, (math.sqrt(v) if v == v and v != math.inf else v)


def mode_of(spec, p):
    if spec.get("mode_text"):
        return spec["mode_text"]
    d = spec["dist"](p); lo, hi = spec["support"](p)
    if spec["disc"]:
        xs = np.arange(int(lo), int(hi) + 1); return fmt(float(xs[np.argmax(d.pmf(xs))]), 3)
    xs = np.linspace(lo, hi, 4001); y = d.pdf(xs); y[~np.isfinite(y)] = -1
    return fmt(float(xs[np.argmax(y)]), 3)


# ════════════════════════════ Estado de cada distribución ════════════════════════════
def pkey(did, k):
    return f"{did}__{k}"


def init_params(did):
    for q in D[did]["params"]:
        st.session_state.setdefault(pkey(did, q["k"]), q["v"])
    st.session_state.setdefault(f"{did}__pins", [])


def current_params(did):
    return {q["k"]: st.session_state[pkey(did, q["k"])] for q in D[did]["params"]}


def constrain(did):
    s = st.session_state
    if did == "uniforme":
        if s[pkey(did, "a")] >= s[pkey(did, "b")]:
            s[pkey(did, "b")] = min(1000, s[pkey(did, "a")] + 10)
            if s[pkey(did, "a")] >= s[pkey(did, "b")]:
                s[pkey(did, "a")] = s[pkey(did, "b")] - 10
    if did == "uniforme-discreta" and s[pkey(did, "a")] >= s[pkey(did, "b")]:
        s[pkey(did, "b")] = min(30, s[pkey(did, "a")] + 1)
        if s[pkey(did, "a")] >= s[pkey(did, "b")]:
            s[pkey(did, "a")] = s[pkey(did, "b")] - 1
    if did == "hipergeometrica":
        N = s[pkey(did, "N")]
        s[pkey(did, "K")] = min(s[pkey(did, "K")], N)
        s[pkey(did, "n")] = max(1, min(s[pkey(did, "n")], N))


def param_sliders(did):
    spec = D[did]; init_params(did); constrain(did)
    before = current_params(did); b_stats = mstats(spec, before)
    for q in spec["params"]:
        mx = st.session_state[pkey(did, q["maxk"])] if q.get("maxk") else q["max"]
        isint = isinstance(q["step"], int)
        cast = int if isint else float
        key = pkey(did, q["k"])
        st.session_state[key] = cast(min(max(st.session_state[key], q["min"]), mx))
        st.slider(f"**{q['sym']}** · {q['label']}", min_value=cast(q["min"]), max_value=cast(mx), step=cast(q["step"]), key=key,
                  on_change=_remember, args=(did, q["k"], before, b_stats))
    return current_params(did)


def _remember(did, k, before, b_stats):
    st.session_state[f"{did}__last"] = dict(k=k, old=before[k], stats=b_stats)
    st.session_state[f"{did}__sim"] = None


# ════════════════════════════ Cálculos ════════════════════════════
def prob_calc(spec, p, typ, a, b):
    d = spec["dist"](p)
    if spec["disc"]:
        a, b = int(a), int(b)
        if typ == "eq": return float(d.pmf(a)), f"P(X = {a})", (a, a)
        if typ == "le": return float(d.cdf(b)), f"P(X ≤ {b}) = Σ P(X = x), x ≤ {b}", (-math.inf, b)
        if typ == "ge": return float(d.sf(a - 1)), f"P(X ≥ {a}) = 1 − P(X ≤ {a - 1})", (a, math.inf)
        lo, hi = min(a, b), max(a, b)
        return float(d.cdf(hi) - d.cdf(lo - 1)), f"P({lo} ≤ X ≤ {hi}) = F({hi}) − F({lo - 1})", (lo, hi)
    if typ in ("le", "eq"): return float(d.cdf(b)), f"P(X ≤ {fmt(b)}) = F({fmt(b)})", (-math.inf, b)
    if typ == "ge": return float(d.sf(a)), f"P(X ≥ {fmt(a)}) = 1 − F({fmt(a)})", (a, math.inf)
    lo, hi = min(a, b), max(a, b)
    return float(d.cdf(hi) - d.cdf(lo)), f"P({fmt(lo)} ≤ X ≤ {fmt(hi)}) = F({fmt(hi)}) − F({fmt(lo)})", (lo, hi)


def interp(spec, typ, a, b):
    d = spec["xdesc"]
    return {"eq": f"Probabilidad de que {d} sea exactamente {fmt(a)}.", "le": f"Probabilidad de que {d} sea como máximo {fmt(b)}.",
            "ge": f"Probabilidad de que {d} sea al menos {fmt(a)}.", "btw": f"Probabilidad de que {d} esté entre {fmt(min(a, b))} y {fmt(max(a, b))}."}[typ]


def crit_calc(spec, p, alpha, tail):
    d = spec["dist"](p)
    if tail == "right":
        c = float(d.ppf(1 - alpha)); return [(c, math.inf)], [c]
    if tail == "left":
        c = float(d.ppf(alpha)); return [(-math.inf, c)], [c]
    c1, c2 = float(d.ppf(alpha / 2)), float(d.ppf(1 - alpha / 2)); return [(-math.inf, c1), (c2, math.inf)], [c1, c2]


def sample(spec, p, n):
    if spec.get("sampler"):
        return spec["sampler"](p, n)
    return spec["dist"](p).rvs(size=n)


# ════════════════════════════ Gráfica principal ════════════════════════════
def dist_figure(did, p, *, cdf=False, show_mean=True, pins=(), shade=None, regions=None, hist=None,
                rule=False, zscale=False, overlay_norm=False, overlay_chi=False, height=440):
    spec = D[did]; fam = FAMS[spec["fam"]][1]; disc = spec["disc"]
    curves = [(pp, PINS[i % 4], ptext(spec, pp), False) for i, pp in enumerate(pins)] + [(p, fam, f"{spec['name']}: {ptext(spec, p)}", True)]
    lo = min(spec["support"](c[0])[0] for c in curves); hi = max(spec["support"](c[0])[1] for c in curves)
    ylabel = "F(x) = P(X ≤ x)" if cdf else ("P(X = x)" if disc else "f(x) · densidad")
    fig = base_fig(spec["xlabel"], ylabel, height)
    ymax = 0
    if disc:
        xs = np.arange(int(math.floor(lo)), int(math.ceil(hi)) + 1)
        if hist is not None and not cdf:
            vals, cnt = np.unique(hist, return_counts=True)
            fig.add_bar(x=vals, y=cnt / len(hist), name=f"datos simulados ({len(hist):,})", marker_color=hexa(C["sim"], 0.65), width=0.8)
            ymax = max(ymax, float((cnt / len(hist)).max()))
        for pp, col, lab, main in curves:
            d = spec["dist"](pp)
            if cdf:
                xx = np.arange(xs[0] - 1, xs[-1] + 2)
                fig.add_scatter(x=xx, y=d.cdf(xx), mode="lines+markers", line=dict(shape="hv", color=col, width=2.6 if main else 1.8), name=lab)
            elif main and hist is None:
                y = d.pmf(xs)
                if shade:
                    inr = (xs >= shade[0]) & (xs <= shade[1])
                    colors = [col if i else hexa(col, 0.22) for i in inr]
                    fig.add_bar(x=xs, y=y, marker_color=colors, name=lab, width=0.7)
                else:
                    fig.add_bar(x=xs, y=y, marker_color=hexa(col, 0.85 if pins else 1), name=lab, width=0.7)
                ymax = max(ymax, float(y.max()))
            else:
                y = d.pmf(xs); ymax = max(ymax, float(y.max()))
                fig.add_scatter(x=xs, y=y, mode="lines+markers", line=dict(color=col, width=1.5), marker=dict(size=8, line=dict(color="#16202C", width=1.5)), name=lab)
        fig.update_xaxes(dtick=max(1, int(round((xs[-1] - xs[0]) / 14))) if len(xs) > 15 else 1, range=[lo - 0.7, hi + 0.7])
    else:
        xs = np.linspace(lo, hi, 700)
        if hist is not None and not cdf:
            edges = np.linspace(lo, hi, int(min(60, max(18, math.sqrt(len(hist)) * 1.2))) + 1)
            dens, _ = np.histogram(hist, bins=edges); dens = dens / (len(hist) * (edges[1] - edges[0]))
            fig.add_bar(x=(edges[:-1] + edges[1:]) / 2, y=dens, width=(edges[1] - edges[0]) * 0.94, marker_color=hexa(C["sim"], 0.6), name=f"datos simulados ({len(hist):,})")
            ymax = max(ymax, float(dens.max()))
        for pp, col, lab, main in curves:
            d = spec["dist"](pp)
            y = d.cdf(xs) if cdf else d.pdf(xs)
            yy = y[(xs > lo + 0.02 * (hi - lo))]
            ymax = max(ymax, float(np.nanmax(yy[np.isfinite(yy)])) if len(yy) else 0)
            if main and not cdf:
                if rule and did == "normal":
                    for k in (3, 2, 1):
                        m_ = (xs >= p["mu"] - k * p["s"]) & (xs <= p["mu"] + k * p["s"])
                        fig.add_scatter(x=xs[m_], y=y[m_], fill="tozeroy", fillcolor=hexa(col, 0.12), line=dict(width=0), showlegend=False, hoverinfo="skip")
                if shade:
                    m_ = (xs >= shade[0]) & (xs <= shade[1])
                    xa = np.concatenate([[max(shade[0], lo)], xs[m_], [min(shade[1], hi)]])
                    fig.add_scatter(x=xa, y=d.pdf(xa), fill="tozeroy", fillcolor=hexa(col, 0.4), line=dict(width=0), name="área = probabilidad pedida", hoverinfo="skip")
                for (u, v) in (regions or []):
                    m_ = (xs >= u) & (xs <= v)
                    xa = np.concatenate([[max(u, lo)], xs[m_], [min(v, hi)]])
                    fig.add_scatter(x=xa, y=d.pdf(xa), fill="tozeroy", fillcolor=hexa(C["bad"], 0.45), line=dict(width=0), name="región de rechazo", showlegend=(u == regions[0][0]), hoverinfo="skip")
            fill = "tozeroy" if (main and not cdf and hist is None and shade is None and not regions and not pins and not rule) else None
            fig.add_scatter(x=xs, y=y, mode="lines", line=dict(color=col, width=3 if main else 2.2), fill=fill, fillcolor=hexa(col, 0.14), name=lab)
        if overlay_norm:
            fig.add_scatter(x=xs, y=stats.norm.cdf(xs) if cdf else stats.norm.pdf(xs), mode="lines", line=dict(color=C["muted"], dash="dash", width=2), name="N(0, 1)")
        if overlay_chi and not cdf:
            k = p["k"]; fig.add_scatter(x=xs, y=stats.norm.pdf(xs, k, math.sqrt(2 * k)), mode="lines", line=dict(color=C["muted"], dash="dash", width=2), name="N(k, √2k)")
        fig.update_xaxes(range=[lo, hi])
    top = 1.1 if cdf else (ymax * 1.22 if ymax > 0 else 1)
    fig.update_yaxes(range=[0, top])
    if regions:
        for u, v in regions:
            c = v if u == -math.inf else u
            fig.add_vline(x=c, line=dict(color=C["bad"], width=1.5), annotation_text=fmt(c, 3), annotation_font_color=C["bad"])
    if show_mean and not cdf:
        m, var, s = mstats(spec, p)
        if m == m and abs(m) != math.inf:
            fig.add_vline(x=m, line=dict(color=C["ink"], dash="dash", width=1.3), annotation_text=f"μ = {fmt(m, 3)}", annotation_position="top right")
            if s == s and s != math.inf and s > 0 and not rule:
                yb = top * 0.86
                fig.add_shape(type="line", x0=m - s, x1=m + s, y0=yb, y1=yb, line=dict(color=C["ink"], width=1.3))
                for xx in (m - s, m + s):
                    fig.add_shape(type="line", x0=xx, x1=xx, y0=yb - top * 0.02, y1=yb + top * 0.02, line=dict(color=C["ink"], width=1.3))
                fig.add_annotation(x=m, y=yb, text="± σ", showarrow=False, yshift=10, font=dict(family="IBM Plex Mono, Consolas, monospace", size=12))
    if rule and did == "normal" and not cdf:
        for i, (k, lab) in enumerate([(1, "68.27 %"), (2, "95.45 %"), (3, "99.73 %")]):
            yb = top * (0.95 - i * 0.075)
            fig.add_shape(type="line", x0=p["mu"] - k * p["s"], x1=p["mu"] + k * p["s"], y0=yb, y1=yb, line=dict(color=C["muted"], width=1.2))
            fig.add_annotation(x=p["mu"], y=yb, text=lab, showarrow=False, yshift=9, font=dict(family="IBM Plex Mono, Consolas, monospace", size=11, color=C["muted"]), bgcolor="#16202C")
    if zscale and did == "normal":
        ks = [k for k in range(-4, 5) if lo <= p["mu"] + k * p["s"] <= hi]
        fig.update_layout(xaxis2=dict(overlaying="x", side="top", range=[lo, hi], tickvals=[p["mu"] + k * p["s"] for k in ks],
                                      ticktext=[f"z={k}" for k in ks], tickfont=dict(color=fam, family="IBM Plex Mono, Consolas, monospace"), showgrid=False),
                          margin=dict(t=60))
        fig.add_scatter(x=[lo], y=[0], xaxis="x2", mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip")
    return fig


# ════════════════════════════ Componentes de página ════════════════════════════
def page_head(pid, title=None, lead=None):
    n = NAVD[pid]; fam_name, fam_col = FAMS[n[3]]
    st.markdown(f"""<div style="--fam:{fam_col}"><div class="log">[{n[1]}:00] <span class="lvl">INFO</span> srv-01 &nbsp;<b>{n[4]}</b></div>
    <div class="ptitle">{title or n[2]}{f'<span class="badge">{fam_name}</span>' if fam_name else ''}</div>
    {f'<div class="lead">{lead}</div>' if lead else ''}</div>""", unsafe_allow_html=True)


def info_cards(spec):
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### Fórmula"); st.latex(spec["formula"]); st.latex(spec["moments"])
        with st.container(border=True):
            st.markdown("#### Parámetros")
            for q in spec["params"]:
                st.markdown(f"**{q['sym']}** — {q['meaning']}")
    with c2:
        with st.container(border=True):
            st.markdown("#### Características"); st.markdown("\n".join(f"- {c}" for c in spec["chars"]))
        with st.container(border=True):
            st.markdown("#### Cuándo usarla"); st.markdown(spec["when"]); st.caption(f"**Cuidado:** {spec['careful']}")
    with c3:
        with st.container(border=True):
            st.markdown("#### En el servidor · srv-01"); st.markdown(spec["app"]); st.markdown("\n".join(f"- {c}" for c in spec["more"]))


def nav_buttons(pid):
    i = NAV_IDS.index(pid); c1, _, c3 = st.columns([1, 2, 1])
    if i > 0:
        a = NAV[i - 1]; c1.button(f"← {a[1]} · {a[2]}", key=f"prev_{pid}", on_click=goto, args=(a[0],), width="stretch")
    if i < len(NAV) - 1:
        b = NAV[i + 1]; c3.button(f"{b[1]} · {b[2]} →", key=f"next_{pid}", on_click=goto, args=(b[0],), width="stretch")


def goto(pid):
    st.session_state["page"] = pid


def video_mode():
    return st.session_state.get("video", False)


# ════════════════════════════ Mesa de trabajo genérica ════════════════════════════
def workbench(did):
    spec = D[did]; fam = FAMS[spec["fam"]][1]; init_params(did)
    left, right = st.columns([2.6, 1], gap="large")
    with right:
        with st.container(border=True):
            st.markdown("#### Parámetros")
            p = param_sliders(did)
            if st.button("Restablecer", key=f"{did}__reset"):
                for q in spec["params"]:
                    st.session_state[pkey(did, q["k"])] = q["v"]
                st.session_state[f"{did}__pins"] = []; st.session_state[f"{did}__last"] = None; st.rerun()
            m, v, s = mstats(spec, p)
            a1, a2 = st.columns(2); a1.metric("Media E[X]", fmt(m)); a2.metric("Varianza V[X]", fmt(v))
            a3, a4 = st.columns(2); a3.metric("Desv. estándar σ", fmt(s)); a4.metric("Moda", mode_of(spec, p))
            last = st.session_state.get(f"{did}__last")
            if last:
                q = next(x for x in spec["params"] if x["k"] == last["k"])
                m0, _, s0 = last["stats"]
                body = f"{q['effect']}<div class='delta'>{q['sym']}: {fmt(last['old'], 3)} → {fmt(p[q['k']], 3)} · media {fmt(m0, 3)} → {fmt(m, 3)} · σ {fmt(s0, 3)} → {fmt(s, 3)}</div>"
            else:
                body = "Mueve un parámetro para ver qué le pasa a la gráfica."
            st.markdown(f"<div class='effect' style='--fam:{fam}'><div class='eyebrow'>¿Qué cambia?</div>{body}</div>", unsafe_allow_html=True)

    with left:
        h = 560 if video_mode() else 440
        tabs = ["Explorar", "Comparar parámetros", "Calcular probabilidad"] + (["Valor crítico"] if spec.get("critical") else []) + ["Simular datos"]
        T = st.tabs(tabs); ti = dict(zip(tabs, T))
        extra = {}

        def toggles(key):
            cols = st.columns(4)
            cdf = cols[0].toggle("F(x) acumulada", key=f"{did}__cdf_{key}")
            mean = cols[1].toggle("Media ± σ", value=True, key=f"{did}__mean_{key}")
            ex = {}
            if did == "normal":
                ex["rule"] = cols[2].toggle("Regla 68-95-99.7", key=f"{did}__rule_{key}")
                ex["zscale"] = cols[3].toggle("Escala Z", value=True, key=f"{did}__z_{key}")
            if did == "t":
                ex["overlay_norm"] = cols[2].toggle("Normal estándar Z", value=True, key=f"{did}__on_{key}")
            if did == "chi2":
                ex["overlay_chi"] = cols[2].toggle("Aprox. normal N(k, √2k)", key=f"{did}__oc_{key}")
            return cdf, mean, ex

        with ti["Explorar"]:
            cdf, mean, ex = toggles("exp")
            show(dist_figure(did, p, cdf=cdf, show_mean=mean, height=h, **ex), key=f"{did}_f_exp")
            st.caption("Pasa el cursor sobre la gráfica para leer cada valor. Activa F(x) acumulada para ver la función de distribución.")

        with ti["Comparar parámetros"]:
            pins = st.session_state[f"{did}__pins"]
            c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 0.8])
            if c1.button("Fijar curva actual", key=f"{did}__pin", type="primary"):
                pins.append(dict(p)); del pins[:-4]
            fk = c2.selectbox("Familia de", [q["k"] for q in spec["params"]], format_func=lambda k: next(q["sym"] for q in spec["params"] if q["k"] == k), key=f"{did}__famk")
            if c3.button("Generar 3 curvas", key=f"{did}__fam"):
                q = next(x for x in spec["params"] if x["k"] == fk); pins.clear()
                for val in q["fam"]:
                    pp = dict(p); pp[fk] = val
                    if did == "hipergeometrica":
                        pp["K"] = min(pp["K"], pp["N"]); pp["n"] = min(pp["n"], pp["N"])
                    if did in ("uniforme", "uniforme-discreta") and pp["a"] >= pp["b"]:
                        continue
                    if pp != p and pp not in pins:
                        pins.append(pp)
                st.session_state[f"{did}__last"] = dict(k=fk, old=p[fk], stats=mstats(spec, p))
            if c4.button("Limpiar", key=f"{did}__clr"):
                pins.clear()
            show(dist_figure(did, p, pins=pins, show_mean=False, height=h), key=f"{did}_f_cmp")
            if pins:
                rows = [{"Curva": f"referencia {i + 1}", "Parámetros": ptext(spec, pp), "Media": fmt(mstats(spec, pp)[0], 3), "σ": fmt(mstats(spec, pp)[2], 3)} for i, pp in enumerate(pins)]
                rows.append({"Curva": "actual", "Parámetros": ptext(spec, p), "Media": fmt(m, 3), "σ": fmt(s, 3)})
                st.dataframe(rows, hide_index=True, width="stretch")
            else:
                st.caption("Fija la curva actual, mueve un parámetro y compara. O genera una familia automática de 3 curvas para un parámetro.")

        with ti["Calcular probabilidad"]:
            types = ([("eq", "P(X = a)")] if spec["disc"] else []) + [("le", "P(X ≤ b)"), ("ge", "P(X ≥ a)"), ("btw", "P(a ≤ X ≤ b)")]
            dt, da, db = spec["prob"]
            c1, c2, c3 = st.columns(3)
            typ = c1.selectbox("Evento", [t[0] for t in types], index=[t[0] for t in types].index(dt), format_func=dict(types).get, key=f"{did}__pt")
            stepv = 1 if spec["disc"] else 0.1
            cast = int if spec["disc"] else float
            a = c2.number_input("a", value=cast(da), step=cast(stepv), key=f"{did}__pa", disabled=typ == "le")
            b = c3.number_input("b", value=cast(db), step=cast(stepv), key=f"{did}__pb", disabled=typ in ("ge", "eq"))
            val, expr, rng = prob_calc(spec, p, typ, a, b)
            show(dist_figure(did, p, shade=rng, show_mean=False, height=h - 40), key=f"{did}_f_prob")
            extra_z = ""
            if did == "normal":
                z = lambda x: fmt((x - p["mu"]) / p["s"], 3)
                extra_z = f" · z = ({fmt(b)} − {fmt(p['mu'])}) / {fmt(p['s'])} = {z(b)}" if typ == "le" else (f" · z = ({fmt(a)} − {fmt(p['mu'])}) / {fmt(p['s'])} = {z(a)}" if typ == "ge" else f" · z₁ = {z(a)}, z₂ = {z(b)}")
            r1, r2 = st.columns([1, 2.4])
            r1.markdown(f"<div class='big' style='--fam:{fam}'>{fmt(val, 5)}</div>", unsafe_allow_html=True)
            r2.markdown(f"<div class='calc'><b>{expr}</b><br>{interp(spec, typ, a, b)}<br>= {pct(val)}{extra_z}</div>", unsafe_allow_html=True)

        if spec.get("critical"):
            with ti["Valor crítico"]:
                c1, c2 = st.columns(2)
                alpha = c1.selectbox("Nivel α", [0.10, 0.05, 0.025, 0.01], index=1, key=f"{did}__ca")
                tails = {"two": "Dos colas", "right": "Cola derecha", "left": "Cola izquierda"}
                tail = c2.selectbox("Región", list(tails), index=list(tails).index(spec["crit_tail"]), format_func=tails.get, key=f"{did}__ct")
                regions, vals = crit_calc(spec, p, alpha, tail)
                show(dist_figure(did, p, regions=regions, show_mean=False, height=h - 40, overlay_norm=(did == "t")), key=f"{did}_f_crit")
                sym = spec["xlabel"].split(" ")[0]
                big = f"± {fmt(vals[1], 4)}" if len(vals) == 2 and abs(vals[0] + vals[1]) < 1e-6 else " y ".join(fmt(v_, 4) for v_ in vals)
                reg = {"right": f"{sym} ≥ {fmt(vals[0], 4)}", "left": f"{sym} ≤ {fmt(vals[0], 4)}"}.get(tail, f"{sym} ≤ {fmt(vals[0], 4)} o {sym} ≥ {fmt(vals[-1], 4)}")
                r1, r2 = st.columns([1, 2.4])
                r1.markdown(f"<div class='big' style='--fam:{fam}'>{big}</div>", unsafe_allow_html=True)
                r2.markdown(f"<div class='calc'><b>Valor crítico con {ptext(spec, p)} y α = {alpha}</b><br>Región de rechazo (en rojo): {reg}<br>Si el estadístico calculado con la muestra cae en la zona roja, se rechaza H₀ con nivel de significancia α.</div>", unsafe_allow_html=True)

        with ti["Simular datos"]:
            c1, c2, c3 = st.columns([1, 1, 1])
            N = c1.selectbox("Repeticiones", [100, 1000, 10000], index=1, key=f"{did}__sn")
            run = c2.button("Simular", key=f"{did}__sr", type="primary")
            anim = c3.button("Animar paso a paso", key=f"{did}__sa")
            st.caption(spec.get("sim_note", "Se generan valores aleatorios con los parámetros actuales y su histograma se compara con la distribución teórica. Con más repeticiones el histograma se acerca a la curva."))
            if run or anim:
                st.session_state[f"{did}__sim"] = sample(spec, p, N)
            data = st.session_state.get(f"{did}__sim")
            if anim and data is not None:
                ph = st.empty()
                import time
                for frac in np.linspace(0.05, 1, 16):
                    k = max(2, int(len(data) * frac))
                    ph.plotly_chart(_dark(dist_figure(did, p, hist=data[:k], show_mean=False, height=h)), width="stretch", key=f"{did}_anim_{k}")
                    time.sleep(0.12)
            elif data is not None:
                show(dist_figure(did, p, hist=data, show_mean=False, height=h), key=f"{did}_f_sim")
            else:
                show(dist_figure(did, p, show_mean=False, height=h), key=f"{did}_f_sim0")
            if data is not None:
                st.dataframe([{"": "Media", "Teórico": fmt(m), "Simulado": fmt(float(np.mean(data)))},
                              {"": "Varianza", "Teórico": fmt(v), "Simulado": fmt(float(np.var(data, ddof=1)))},
                              {"": "Desv. estándar", "Teórico": fmt(s), "Simulado": fmt(float(np.std(data, ddof=1)))},
                              {"": "Datos generados", "Teórico": "", "Simulado": f"{len(data):,}"}], hide_index=True, width="stretch")
    if not video_mode():
        st.markdown("")
        info_cards(spec)


# ════════════════════════════ Páginas ════════════════════════════
def page_dist(did):
    page_head(did, D[did]["name"], D[did]["lead"])
    workbench(did)


def mini_fig(pid):
    fig = go.Figure(); fam = FAMS[NAVD[pid][3]][1]
    if pid in D:
        spec = D[pid]; p = {q["k"]: q["v"] for q in spec["params"]}; lo, hi = spec["support"](p); d = spec["dist"](p)
        if spec["disc"]:
            xs = np.arange(int(lo), int(hi) + 1); fig.add_bar(x=xs, y=d.pmf(xs), marker_color=fam)
        else:
            xs = np.linspace(lo, hi, 200); fig.add_scatter(x=xs, y=d.pdf(xs), fill="tozeroy", line=dict(color=fam, width=2), fillcolor=hexa(fam, 0.15))
    elif pid == "multinomial":
        fig.add_bar(x=["2xx", "4xx", "5xx"], y=[0.7, 0.2, 0.1], marker_color=[C["ok"], C["warn"], C["bad"]])
    elif pid == "aproximaciones":
        xs = np.arange(0, 25); fig.add_bar(x=xs, y=stats.binom.pmf(xs, 30, 0.3), marker_color=hexa(C["disc"], 0.45))
        xx = np.linspace(0, 24, 200); fig.add_scatter(x=xx, y=stats.norm.pdf(xx, 9, math.sqrt(6.3)), line=dict(color=C["aprox"], width=2.4))
    elif pid == "muestreo":
        xx = np.linspace(-5, 5, 200); fig.add_scatter(x=xx, y=stats.norm.pdf(xx), line=dict(color=C["muted"], dash="dash"))
        fig.add_scatter(x=xx, y=stats.t.pdf(xx, 2), fill="tozeroy", line=dict(color=C["samp"], width=2), fillcolor=hexa(C["samp"], 0.15))
    elif pid == "duelos":
        xs = np.arange(0, 13); fig.add_bar(x=xs, y=stats.poisson.pmf(xs, 4), marker_color=hexa(C["disc"], 0.7))
        xx = np.linspace(0, 13, 200); fig.add_scatter(x=xx, y=0.25 * np.exp(-0.25 * xx), line=dict(color=C["cont"], width=2.4))
    fig.update_layout(height=90, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, template="plotly_dark", bargap=0.25,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def page_inicio():
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown("<div class='eyebrow'>Atlas de distribuciones de probabilidad · Unidad 3</div><div class='ptitle' style='font-size:56px'>Un día en la vida de un servidor</div>", unsafe_allow_html=True)
        st.markdown("<div class='lead'>Este atlas sigue a <b>srv-01</b>, un servidor web, desde que se enciende a las 08:00 hasta la revisión de las 22:00. Cada hora ocurre una situación real y aparece la distribución que la describe: su gráfica, qué significan sus parámetros, qué pasa al cambiarlos, cómo calcular probabilidades y cómo simular datos.</div>", unsafe_allow_html=True)
    with c2:
        k1, k2, k3 = st.columns(3)
        for col, n, t in ((k1, "13", "distribuciones"), (k2, "4", "aproximaciones + Galton"), (k3, "6", "duelos de comparación")):
            with col.container(border=True):
                st.markdown(f"<div class='kpi'><b>{n}</b><span>{t}</span></div>", unsafe_allow_html=True)
    st.markdown("## La agenda de srv-01")
    slots = NAV[1:]
    for r in range(0, len(slots), 4):
        cols = st.columns(4)
        for col, n in zip(cols, slots[r:r + 4]):
            fam_name, fam_col = FAMS[n[3]]
            with col.container(border=True):
                st.markdown(f"<div style='display:flex;justify-content:space-between'><span class='slot-hr'>{n[1]}</span><span class='slot-tag' style='color:{fam_col}'>{fam_name or 'Comparación'}</span></div><div class='slot-name'>{n[2]}</div>", unsafe_allow_html=True)
                st.caption(n[4])
                st.plotly_chart(mini_fig(n[0]).update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"), width="stretch", config={"staticPlot": True}, key=f"mini_{n[0]}")
                st.button("Abrir", key=f"open_{n[0]}", on_click=goto, args=(n[0],), width="stretch")
    st.markdown("## Cómo se conectan")
    dot = """digraph G { rankdir=TB; bgcolor="transparent"; nodesep=0.6; ranksep=0.55;
      node [shape=box, style="rounded,filled", fillcolor="#16202C", fontname="Helvetica", fontsize=13, penwidth=1.6];
      edge [color="#5D6B7E", fontname="Courier", fontsize=10, fontcolor="#93A0B2"];
      ber [label="Bernoulli", color="#6F96FF"]; bin [label="Binomial", color="#6F96FF"]; mul [label="Multinomial", color="#6F96FF"];
      hip [label="Hipergeométrica", color="#6F96FF"]; poi [label="Poisson", color="#6F96FF"]; ud [label="Unif. discreta", color="#6F96FF"];
      exp [label="Exponencial", color="#F09A4E"]; uc [label="Unif. continua", color="#F09A4E"]; nor [label="Normal", color="#F09A4E"]; z [label="Normal estándar Z", color="#F09A4E"];
      t [label="t de Student", color="#3FC5B1"]; chi [label="Chi-cuadrado", color="#3FC5B1"]; f [label="F de Fisher", color="#3FC5B1"];
      ber -> bin [label=" suma de n"]; bin -> mul [label=" k > 2 categorías"]; bin -> hip [label=" sin reemplazo"]; bin -> poi [label=" n grande, p pequeña"];
      poi -> exp [label=" tiempo entre eventos"]; ud -> uc [label=" valores continuos"]; bin -> nor [label=" n grande (TCL)"]; poi -> nor [label=" λ grande"];
      nor -> z [label=" Z = (X − μ)/σ"]; z -> chi [label=" suma de Z²"]; z -> t [label=" Z / √(χ²/ν)"]; chi -> t; chi -> f [label=" cociente de χ²"]; }"""
    with st.container(border=True):
        st.graphviz_chart(dot, width="stretch")
    st.markdown("## Qué puedes hacer en cada distribución")
    cols = st.columns(5)
    for col, (t, d) in zip(cols, [("Explorar", "Gráfica de P(X = x) o f(x), y F(x) acumulada. Pasa el cursor para leer valores."),
                                   ("Comparar parámetros", "Fija curvas o genera una familia de 3 y observa cómo cambia la forma."),
                                   ("Calcular probabilidad", "P(X ≤ b), P(X ≥ a) o P(a ≤ X ≤ b), con el área sombreada."),
                                   ("Simular datos", "Genera datos aleatorios y compara el histograma con la curva teórica."),
                                   ("Modo video", "En la barra lateral: oculta las tarjetas de teoría y agranda las gráficas.")]):
        with col.container(border=True):
            st.markdown(f"**{t}**"); st.caption(d)


# ---------- Multinomial ----------
def mul_pmf(x1, x2, n, p1, p2):
    x3 = n - x1 - x2
    if x1 < 0 or x2 < 0 or x3 < 0: return 0.0
    return float(stats.multinomial.pmf([x1, x2, x3], n, [p1, p2, max(0.0, 1 - p1 - p2)]))


def page_multinomial():
    page_head("multinomial", "Multinomial", "Generaliza la Binomial a k ≥ 3 categorías: en n ensayos independientes, cuenta cuántos resultados caen en cada categoría. Aquí, cada respuesta del servidor es 2xx (éxito), 4xx (error del cliente) o 5xx (error del servidor).")
    s = st.session_state
    s.setdefault("mul_n", 12); s.setdefault("mul_p1", 0.70); s.setdefault("mul_p2", 0.20)
    left, right = st.columns([2.6, 1], gap="large")
    with right:
        with st.container(border=True):
            st.markdown("#### Parámetros")
            n = st.slider("**n** · peticiones en la ráfaga", 1, 30, key="mul_n")
            p1 = st.slider("**p₁** · prob. de 2xx (éxito)", 0.0, 1.0, step=0.01, key="mul_p1")
            s["mul_p2"] = min(s["mul_p2"], round(1 - p1, 2))
            p2 = st.slider("**p₂** · prob. de 4xx (error cliente)", 0.0, max(0.01, round(1 - p1, 2)), step=0.01, key="mul_p2")
            p3 = max(0.0, 1 - p1 - p2)
            st.markdown(f"<div style='display:flex;height:14px;border-radius:7px;overflow:hidden;border:1px solid {C['line']}'><span style='width:{p1*100}%;background:{C['ok']}'></span><span style='width:{p2*100}%;background:{C['warn']}'></span><span style='width:{p3*100}%;background:{C['bad']}'></span></div>"
                        f"<div class='calc' style='margin-top:4px'>2xx {fmt(p1, 2)} · 4xx {fmt(p2, 2)} · 5xx (p₃ = 1 − p₁ − p₂) {fmt(p3, 2)}</div>", unsafe_allow_html=True)
            st.dataframe([{"Categoría": l, "pᵢ": fmt(q, 2), "E[Xᵢ]": fmt(n * q, 3), "V[Xᵢ]": fmt(n * q * (1 - q), 3)} for l, q in (("2xx", p1), ("4xx", p2), ("5xx", p3))], hide_index=True, width="stretch")
            st.caption(f"Cov(X₁, X₂) = −n p₁ p₂ = {fmt(-n * p1 * p2, 3)}")
            st.markdown(f"<div class='effect' style='--fam:{C['disc']}'><div class='eyebrow'>¿Qué cambia?</div>Al subir p₁ la nube de probabilidad se mueve hacia más respuestas 2xx. Como las tres cuentas suman n, si una categoría sube las otras bajan: por eso la covarianza es negativa.</div>", unsafe_allow_html=True)
    with left:
        Z = np.full((n + 1, n + 1), np.nan); txt = np.full((n + 1, n + 1), "", dtype=object)
        for a in range(n + 1):
            for b in range(n + 1 - a):
                Z[b, a] = mul_pmf(a, b, n, p1, p2); txt[b, a] = f"2xx={a}, 4xx={b}, 5xx={n - a - b}<br>P = {fmt(Z[b, a], 6)}"
        c1, c2, c3 = st.columns(3)
        s.setdefault("mul_x1i", 9); s.setdefault("mul_x2i", 2)
        s["mul_x1i"] = min(int(s["mul_x1i"]), n)
        x1 = c1.number_input("x₁ (2xx)", 0, n, step=1, key="mul_x1i")
        s["mul_x2i"] = min(int(s["mul_x2i"]), n - x1)
        x2 = c2.number_input("x₂ (4xx)", 0, n - x1, step=1, key="mul_x2i")
        x3 = n - x1 - x2
        c3.metric("x₃ (5xx) = n − x₁ − x₂", x3)
        fig = go.Figure(go.Heatmap(z=Z, x=list(range(n + 1)), y=list(range(n + 1)), colorscale=[[0, "#1B2734"], [1, C["disc"]]], text=txt,
                                   hovertemplate="%{text}<extra></extra>", xgap=2, ygap=2, colorbar=dict(title="P", thickness=12)))
        if x3 >= 0:
            fig.add_shape(type="rect", x0=x1 - 0.5, x1=x1 + 0.5, y0=x2 - 0.5, y1=x2 + 0.5, line=dict(color=C["ink"], width=3))
        fig.add_shape(type="line", x0=n + 0.5, y0=-0.5, x1=-0.5, y1=n + 0.5, line=dict(color=C["muted"], dash="dot", width=1))
        fig.update_layout(template="plotly_dark", height=560 if video_mode() else 480, margin=dict(l=60, r=20, t=30, b=50),
                          font=dict(family="IBM Plex Sans, Segoe UI, Arial, sans-serif", color=C["ink"]), xaxis=dict(title="x₁ = respuestas 2xx", dtick=max(1, n // 12), constrain="domain", range=[-0.5, n + 0.5]),
                          yaxis=dict(title="x₂ = respuestas 4xx", dtick=max(1, n // 12), scaleanchor="x", range=[-0.5, n + 0.5], constrain="domain"),
                          title=dict(text="P(X₁ = x₁, X₂ = x₂, X₃ = n − x₁ − x₂)", font=dict(family="IBM Plex Mono, Consolas, monospace", size=13, color=C["muted"])))
        show(fig, key="mul_heat")
        if x3 < 0:
            st.error("x₁ + x₂ no puede superar n. Baja alguno de los dos valores.")
        else:
            v = mul_pmf(x1, x2, n, p1, p2); coef = math.factorial(n) // (math.factorial(x1) * math.factorial(x2) * math.factorial(x3))
            r1, r2 = st.columns([1, 2.4])
            r1.markdown(f"<div class='big' style='--fam:{C['disc']}'>{fmt(v, 5)}</div>", unsafe_allow_html=True)
            r2.markdown(f"<div class='calc'><b>{n}! / ({x1}! · {x2}! · {x3}!) · {fmt(p1, 2)}^{x1} · {fmt(p2, 2)}^{x2} · {fmt(p3, 2)}^{x3}</b><br>= {coef:,} · {fmt(p1 ** x1 * p2 ** x2 * p3 ** x3, 6)} = {pct(v)}<br>Probabilidad de que, en {n} peticiones, haya exactamente {x1} respuestas 2xx, {x2} 4xx y {x3} 5xx.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### Marginales: cada categoría es Binomial(n, pᵢ)")
            fig = base_fig("x", "P", 260); xs = np.arange(n + 1)
            for lab, q, col, off in (("2xx", p1, C["ok"], -0.27), ("4xx", p2, C["warn"], 0), ("5xx", p3, C["bad"], 0.27)):
                fig.add_bar(x=xs + off, y=stats.binom.pmf(xs, n, q), width=0.26, marker_color=col, name=f"X · {lab}")
            fig.update_layout(barmode="overlay", legend=dict(y=-0.3)); show(fig, key="mul_marg")
    with c2:
        with st.container(border=True):
            st.markdown("#### Ráfaga de peticiones")
            b1, b2 = st.columns(2)
            if b1.button("Enviar ráfaga de n", type="primary", width="stretch"):
                u = np.random.rand(n); s["mul_burst"] = np.where(u < p1, 0, np.where(u < p1 + p2, 1, 2)).tolist()
            if b2.button("Simular 1 000 ráfagas", width="stretch"):
                sims = np.random.multinomial(n, [p1, p2, p3], size=1000)
                hit = np.mean((sims[:, 0] == x1) & (sims[:, 1] == x2))
                s["mul_sim"] = dict(mean=sims.mean(0).tolist(), hit=float(hit), x=(x1, x2, x3), p=(n, p1, p2))
            burst = s.get("mul_burst")
            if burst and len(burst) == n:
                cols_ = [C["ok"], C["warn"], C["bad"]]
                st.markdown("<div class='burst'>" + "".join(f"<i style='background:{cols_[k]}'></i>" for k in burst) + "</div>", unsafe_allow_html=True)
                cnt = [burst.count(k) for k in range(3)]
                st.caption(f"2xx: {cnt[0]} · 4xx: {cnt[1]} · 5xx: {cnt[2]} · P de esta combinación = {fmt(mul_pmf(cnt[0], cnt[1], n, p1, p2), 5)}")
            sim = s.get("mul_sim")
            if sim and sim["p"] == (n, p1, p2):
                st.dataframe([{"": "Promedio 2xx", "Teórico": fmt(n * p1, 3), "1 000 ráfagas": fmt(sim["mean"][0], 3)},
                              {"": "Promedio 4xx", "Teórico": fmt(n * p2, 3), "1 000 ráfagas": fmt(sim["mean"][1], 3)},
                              {"": "Promedio 5xx", "Teórico": fmt(n * p3, 3), "1 000 ráfagas": fmt(sim["mean"][2], 3)},
                              {"": f"Frecuencia de {sim['x']}", "Teórico": fmt(mul_pmf(sim['x'][0], sim['x'][1], n, p1, p2), 4), "1 000 ráfagas": fmt(sim["hit"], 4)}], hide_index=True, width="stretch")
    if not video_mode():
        info_cards(dict(formula=r"P(X_1=x_1,\dots,X_k=x_k)=\frac{n!}{x_1!\,x_2!\cdots x_k!}\,p_1^{x_1}\cdots p_k^{x_k}",
                        moments=r"\textstyle\sum x_i=n,\ \sum p_i=1,\ E[X_i]=np_i,\ V[X_i]=np_i(1-p_i),\ \mathrm{Cov}=-np_ip_j",
                        params=[dict(sym="n", meaning="Número de ensayos (peticiones)."), dict(sym="pᵢ", meaning="Probabilidad de cada categoría; suman 1, por eso p₃ = 1 − p₁ − p₂ no es libre.")],
                        chars=["Discreta y multivariada: el resultado es un vector (x₁, x₂, x₃).", "Con k = 2 categorías es exactamente la Binomial.", "Cada componente por separado es Binomial(n, pᵢ).", "Las componentes están correlacionadas negativamente."],
                        when="Hay n ensayos independientes y cada uno cae en una de k ≥ 3 categorías con probabilidades fijas.",
                        careful="Si solo interesa una categoría frente a “el resto”, basta la Binomial.",
                        app="De 12 peticiones, con p = (0.70, 0.20, 0.10) para 2xx, 4xx y 5xx, ¿qué probabilidad hay de tener 9 éxitos, 2 errores del cliente y 1 error del servidor?",
                        more=["Tráfico clasificado por protocolo: TCP, UDP, ICMP.", "Tickets por severidad: baja, media, alta.", "Clics entre varias opciones de un menú."]))
    nav_buttons("multinomial")


# ---------- Aproximaciones ----------
APX = {
    "Binomial → Normal": dict(rule="np ≥ 5 y n(1 − p) ≥ 5", params=[("n", "n · ensayos", 5, 200, 1, 30), ("p", "p · probabilidad de éxito", 0.01, 0.99, 0.01, 0.3)],
        exact=lambda x, p: stats.binom.pmf(x, p["n"], p["p"]), lo=lambda p: 0, hi=lambda p: p["n"], mu=lambda p: p["n"] * p["p"], sd=lambda p: math.sqrt(p["n"] * p["p"] * (1 - p["p"])), kind="normal",
        cond=lambda p: [(f"np = {fmt(p['n'] * p['p'], 2)} ≥ 5", p["n"] * p["p"] >= 5), (f"n(1 − p) = {fmt(p['n'] * (1 - p['p']), 2)} ≥ 5", p["n"] * (1 - p["p"]) >= 5)],
        names=lambda p: (f"Binomial({p['n']}, {p['p']})", f"Normal(μ = {fmt(p['n'] * p['p'], 2)}, σ = {fmt(math.sqrt(p['n'] * p['p'] * (1 - p['p'])), 3)})"), calc=(8, 12)),
    "Poisson → Normal": dict(rule="λ ≥ 10", params=[("l", "λ · tasa promedio", 1, 100, 1, 12)],
        exact=lambda x, p: stats.poisson.pmf(x, p["l"]), lo=lambda p: 0, hi=lambda p: int(p["l"] + 6 * math.sqrt(p["l"]) + 4), mu=lambda p: p["l"], sd=lambda p: math.sqrt(p["l"]), kind="normal",
        cond=lambda p: [(f"λ = {p['l']} ≥ 10", p["l"] >= 10)], names=lambda p: (f"Poisson({p['l']})", f"Normal(μ = {p['l']}, σ = {fmt(math.sqrt(p['l']), 3)})"), calc=(10, 15)),
    "Binomial → Poisson": dict(rule="n ≥ 20 y p ≤ 0.05", params=[("n", "n · ensayos", 10, 500, 5, 100), ("p", "p · probabilidad de éxito", 0.002, 0.3, 0.002, 0.03)],
        exact=lambda x, p: stats.binom.pmf(x, p["n"], p["p"]), lo=lambda p: 0, hi=lambda p: p["n"], mu=lambda p: p["n"] * p["p"], sd=lambda p: math.sqrt(p["n"] * p["p"] * (1 - p["p"])), kind="disc",
        approx=lambda x, p: stats.poisson.pmf(x, p["n"] * p["p"]),
        cond=lambda p: [(f"n = {p['n']} ≥ 20", p["n"] >= 20), (f"p = {fmt(p['p'], 3)} ≤ 0.05", p["p"] <= 0.05)], names=lambda p: (f"Binomial({p['n']}, {fmt(p['p'], 3)})", f"Poisson(λ = np = {fmt(p['n'] * p['p'], 3)})"), calc=(0, 2)),
    "Hipergeométrica → Binomial": dict(rule="n/N ≤ 0.05", params=[("N", "N · población", 10, 1000, 10, 200), ("K", "K · éxitos en la población", 1, 1000, 1, 40), ("n", "n · muestra", 1, 60, 1, 10)],
        exact=lambda x, p: stats.hypergeom.pmf(x, p["N"], p["K"], p["n"]), lo=lambda p: 0, hi=lambda p: p["n"], mu=lambda p: p["n"] * p["K"] / p["N"], sd=lambda p: math.sqrt(p["n"] * p["K"] / p["N"] * (1 - p["K"] / p["N"])), kind="disc",
        approx=lambda x, p: stats.binom.pmf(x, p["n"], p["K"] / p["N"]),
        cond=lambda p: [(f"n/N = {fmt(p['n'] / p['N'], 3)} ≤ 0.05", p["n"] / p["N"] <= 0.05)], names=lambda p: (f"Hipergeométrica({p['N']}, {p['K']}, {p['n']})", f"Binomial({p['n']}, K/N = {fmt(p['K'] / p['N'], 3)})"), calc=(1, 3)),
}


def apx_figure(A, p, a, b, cc, height):
    mu, sd = A["mu"](p), A["sd"](p); lo, hi = A["lo"](p), A["hi"](p)
    v0, v1 = max(lo, int(math.floor(mu - 4.5 * sd - 1))), min(hi, int(math.ceil(mu + 4.5 * sd + 1)))
    xs = np.arange(v0, max(v0 + 2, v1) + 1); ye = A["exact"](xs, p); ne, na = A["names"](p)
    fig = base_fig("x", "probabilidad", height); inr = (xs >= a) & (xs <= b)
    fig.add_bar(x=xs, y=ye, marker_color=[hexa(C["disc"], 0.85) if i else hexa(C["disc"], 0.35) for i in inr], name=f"Exacta: {ne}", width=0.8)
    if A["kind"] == "normal":
        xx = np.linspace(xs[0] - 0.7, xs[-1] + 0.7, 600)
        u, v = (a - 0.5, b + 0.5) if cc else (a, b); m_ = (xx >= u) & (xx <= v)
        fig.add_scatter(x=xx[m_], y=stats.norm.pdf(xx[m_], mu, sd), fill="tozeroy", fillcolor=hexa(C["aprox"], 0.22), line=dict(width=0), name="área usada" + (" (con corrección)" if cc else ""))
        fig.add_scatter(x=xx, y=stats.norm.pdf(xx, mu, sd), line=dict(color=C["aprox"], width=3), name=f"Aproximación: {na}")
    else:
        fig.add_scatter(x=xs, y=A["approx"](xs, p), mode="lines+markers", line=dict(color=C["aprox"], width=1.5), marker=dict(size=9, line=dict(color="#16202C", width=1.5)), name=f"Aproximación: {na}")
    fig.update_xaxes(range=[xs[0] - 0.7, xs[-1] + 0.7])
    return fig


def approx_mass(A, x, p):
    if A["kind"] == "normal":
        mu, sd = A["mu"](p), A["sd"](p); return stats.norm.cdf(x + 0.5, mu, sd) - stats.norm.cdf(x - 0.5, mu, sd)
    return A["approx"](x, p)


def apx_tab(name):
    A = APX[name]; s = st.session_state; key = lambda k: f"apx_{name}_{k}"
    left, right = st.columns([2.6, 1], gap="large")
    with right:
        with st.container(border=True):
            st.markdown("#### Parámetros")
            for k, lab, mn, mx, stp, v in A["params"]:
                s.setdefault(key(k), v)
                if name.startswith("Hiper") and k in ("K", "n"):
                    mx = min(mx, s[key("N")]); s[key(k)] = min(s[key(k)], mx)
                cast = int if isinstance(stp, int) else float
                st.slider(f"**{lab}**", cast(mn), cast(mx), step=cast(stp), key=key(k))
            p = {k: s[key(k)] for k, *_ in A["params"]}
            mu, sd = A["mu"](p), A["sd"](p); lo, hi = A["lo"](p), A["hi"](p)
            xs_all = np.arange(lo, min(hi, int(mu + 8 * sd + 30)) + 1)
            maxerr = float(np.max(np.abs(A["exact"](xs_all, p) - approx_mass(A, xs_all, p))))
            c1, c2 = st.columns(2); c1.metric("Media", fmt(mu, 3)); c2.metric("Desv. estándar", fmt(sd, 3))
            st.metric("Error máximo por punto", fmt(maxerr, 5), help="Mayor diferencia entre la probabilidad exacta y la aproximada para un mismo x.")
            st.markdown(f"Calidad de la aproximación: **{'muy buena' if maxerr < 0.005 else 'aceptable' if maxerr < 0.02 else 'pobre'}**")
            st.markdown(f"<div class='effect' style='--fam:{C['aprox']}'><div class='eyebrow'>¿Qué observar?</div>{'Mientras más grande el parámetro, mejor coinciden las barras con la curva y menor es el error máximo. Usa “Ver convergencia” para animarlo.' if A['kind'] == 'normal' else 'Cuando se cumple la regla, los puntos de la aproximación caen casi sobre las barras exactas.'}</div>", unsafe_allow_html=True)
    with left:
        st.markdown("".join(f"<span class='chip {'ok' if ok else 'no'}'>{'✓' if ok else '✗'} {t}</span>" for t, ok in A["cond"](p)), unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1.2])
        a = c1.number_input("a", value=A["calc"][0], step=1, key=key("a")); b = c2.number_input("b", value=A["calc"][1], step=1, key=key("b"))
        a, b = int(min(a, b)), int(max(a, b))
        cc = c3.toggle("Corrección por continuidad", value=True, key=key("cc")) if A["kind"] == "normal" else False
        conv = c4.button("Ver convergencia", key=key("conv"))
        h = 520 if video_mode() else 430
        ph = st.empty()
        if conv:
            import time
            k0, lab0, mn0, mx0, stp0, _ = A["params"][0]; top = mx0 if k0 != "N" else 1000
            for val in np.linspace(mn0, top * 0.75 if k0 != "N" else top, 18):
                pp = dict(p); pp[k0] = int(round(val / stp0) * stp0) if isinstance(stp0, int) else float(round(val / stp0) * stp0)
                if name.startswith("Hiper"):
                    pp["K"] = min(pp["K"], pp["N"]); pp["n"] = min(pp["n"], pp["N"])
                fig = apx_figure(A, pp, a, b, cc, h); fig.update_layout(title=dict(text=f"{lab0.split(' ·')[0]} = {pp[k0]}", font=dict(family="IBM Plex Mono, Consolas, monospace", size=14)))
                ph.plotly_chart(_dark(fig), width="stretch", key=f"{key('anim')}_{val}"); time.sleep(0.18)
        else:
            ph.plotly_chart(_dark(apx_figure(A, p, a, b, cc, h)), width="stretch", key=key("fig"))
        xs = np.arange(max(lo, a), min(hi, b) + 1); ex = float(A["exact"](xs, p).sum()) if len(xs) else 0.0
        ne, na = A["names"](p); rows = [{"P(%d ≤ X ≤ %d)" % (a, b): f"Exacta ({ne.split('(')[0]})", "Valor": fmt(ex, 5), "Error": "—"}]
        if A["kind"] == "normal":
            sin = stats.norm.cdf(b, mu, sd) - stats.norm.cdf(a, mu, sd); con = stats.norm.cdf(b + 0.5, mu, sd) - stats.norm.cdf(a - 0.5, mu, sd)
            rows += [{"P(%d ≤ X ≤ %d)" % (a, b): "Normal sin corrección: Φ((b − μ)/σ) − Φ((a − μ)/σ)", "Valor": fmt(sin, 5), "Error": fmt(abs(sin - ex), 5)},
                     {"P(%d ≤ X ≤ %d)" % (a, b): "Normal con corrección: Φ((b + 0.5 − μ)/σ) − Φ((a − 0.5 − μ)/σ)", "Valor": fmt(con, 5), "Error": fmt(abs(con - ex), 5)}]
        else:
            ap = float(A["approx"](np.arange(max(0, a), b + 1), p).sum())
            rows.append({"P(%d ≤ X ≤ %d)" % (a, b): f"Aproximación ({na.split('(')[0]})", "Valor": fmt(ap, 5), "Error": fmt(abs(ap - ex), 5)})
        st.dataframe(rows, hide_index=True, width="stretch")


def galton_tab():
    s = st.session_state
    s.setdefault("gal_n", 12); s.setdefault("gal_p", 0.5); s.setdefault("gal_bins", None)
    left, right = st.columns([2.6, 1], gap="large")
    with right:
        with st.container(border=True):
            st.markdown("#### Tablero")
            n = st.slider("**n** · filas de clavos", 4, 18, key="gal_n"); p = st.slider("**p** · prob. de ir a la derecha", 0.1, 0.9, step=0.05, key="gal_p")
            if s["gal_bins"] is None or len(s["gal_bins"]) != n + 1 or s.get("gal_pp") != p:
                s["gal_bins"] = [0] * (n + 1); s["gal_pp"] = p
            st.markdown(f"<div class='effect' style='--fam:{C['aprox']}'><div class='eyebrow'>¿Qué pasa aquí?</div>Cada bolita choca con n clavos y en cada uno va a la derecha con probabilidad p. El casillero final es el número de veces que fue a la derecha: X ~ Binomial(n, p). Al acumular bolitas, el histograma dibuja la campana Normal(np, √(np(1 − p))).</div>", unsafe_allow_html=True)
    with left:
        c1, c2, c3 = st.columns(3)
        anim = c1.button("Soltar 40 (animado)", type="primary", width="stretch")
        if c2.button("+1 000 al instante", width="stretch"):
            add = np.bincount(np.random.binomial(n, p, 1000), minlength=n + 1); s["gal_bins"] = (np.array(s["gal_bins"]) + add).tolist()
        if c3.button("Reiniciar", width="stretch"):
            s["gal_bins"] = [0] * (n + 1)
        mu, sd = n * p, math.sqrt(n * p * (1 - p))
        if anim:
            st.plotly_chart(_dark(galton_anim(n, p)), width="stretch", key=f"gal_anim_{np.random.randint(1e9)}")
            st.caption("Pulsa ▶ para ver caer las bolitas. Al terminar se suman al histograma de abajo.")
            res = np.random.binomial(n, p, 40); s["gal_bins"] = (np.array(s["gal_bins"]) + np.bincount(res, minlength=n + 1)).tolist()
        bins = np.array(s["gal_bins"]); total = int(bins.sum())
        fig = base_fig("casillero (número de veces que fue a la derecha)", "bolitas", 360)
        fig.add_bar(x=np.arange(n + 1), y=bins, marker_color=C["disc"], name=f"casilleros: Binomial({n}, {p})")
        if total:
            xx = np.linspace(-0.5, n + 0.5, 300); fig.add_scatter(x=xx, y=total * stats.norm.pdf(xx, mu, sd), line=dict(color=C["aprox"], width=3), name=f"Normal({fmt(mu, 2)}, {fmt(sd, 3)}) × bolitas")
        fig.update_xaxes(dtick=1); show(fig, key="gal_hist")
        m = float((np.arange(n + 1) * bins).sum() / total) if total else float("nan")
        v = float((bins * (np.arange(n + 1) - m) ** 2).sum() / max(1, total - 1)) if total > 1 else float("nan")
        with right:
            with st.container(border=True):
                c1, c2 = st.columns(2); c1.metric("Bolitas", f"{total:,}"); c2.metric("Media obs. / np", f"{fmt(m, 3) if total else '—'} / {fmt(mu, 3)}")
                st.metric("Var. obs. / np(1−p)", f"{fmt(v, 3) if total > 1 else '—'} / {fmt(sd * sd, 3)}")


def galton_anim(n, p, balls=40):
    rng = np.random.default_rng(); moves = rng.random((balls, n)) < p
    expmax = balls * float(stats.binom.pmf(np.arange(n + 1), n, p).max()); scale = 4.5 / max(1.0, expmax); base = -n - 5.5
    peg_x = [k - r / 2 for r in range(n) for k in range(r + 1)]; peg_y = [-r for r in range(n) for k in range(r + 1)]
    pegs = go.Scatter(x=peg_x, y=peg_y, mode="markers", marker=dict(size=6, color=C["muted"]), hoverinfo="skip")
    frames = []
    for t in range(balls + n + 2):
        xs, ys, bins_t = [], [], np.zeros(n + 1)
        for b in range(balls):
            r = t - b
            if r < 0: continue
            if r >= n:
                bins_t[int(moves[b].sum())] += 1; continue
            k = int(moves[b, :r].sum()); xs.append(k - r / 2); ys.append(-r + 0.4)
        frames.append(go.Frame(name=str(t), data=[pegs,
            go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=12, color=C["aprox"]), hoverinfo="skip"),
            go.Bar(x=np.arange(n + 1) - n / 2, y=bins_t * scale, base=base, width=0.8, marker_color=C["disc"],
                   text=[str(int(v)) if v else "" for v in bins_t], textposition="outside", hoverinfo="skip")]))
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(height=540, template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(visible=False, range=[-n / 2 - 1, n / 2 + 1]), yaxis=dict(visible=False, range=[base - 0.3, 1.2]),
                      updatemenus=[dict(type="buttons", x=0.0, y=1.08, xanchor="left", buttons=[dict(label="▶ Soltar bolitas", method="animate",
                          args=[None, dict(frame=dict(duration=110, redraw=True), fromcurrent=True, transition=dict(duration=0))])])])
    return fig


def page_aprox():
    page_head("aproximaciones", "Aproximaciones a la Normal", "Cuando los parámetros crecen, varias distribuciones discretas toman forma de campana y se pueden calcular con la Normal. También hay aproximaciones entre discretas: Binomial ≈ Poisson e Hipergeométrica ≈ Binomial.")
    names = list(APX) + ["Tablero de Galton"]
    T = st.tabs([f"{n}" for n in names])
    for tab, n in zip(T, names):
        with tab:
            if n == "Tablero de Galton":
                galton_tab()
            else:
                st.caption(f"Regla práctica: {APX[n]['rule']}")
                apx_tab(n)
    if not video_mode():
        c1, c2, c3 = st.columns(3)
        with c1.container(border=True):
            st.markdown("#### ¿Por qué funciona?")
            st.markdown("La Binomial es la suma de n Bernoulli independientes y la Poisson(λ) es la suma de muchas Poisson pequeñas. Por el **Teorema Central del Límite**, una suma de muchas variables independientes tiende a una Normal.")
            st.markdown("#### Corrección por continuidad")
            st.markdown("Cada barra x ocupa el intervalo [x − 0.5, x + 0.5]:"); st.latex(r"P(a\le X\le b)\approx P(a-0.5<Y<b+0.5)")
        with c2.container(border=True):
            st.markdown("#### Reglas prácticas")
            st.dataframe([{"Exacta": "Binomial(n, p)", "Aproximación": "N(np, √(np(1−p)))", "Si": "np ≥ 5, n(1−p) ≥ 5"},
                          {"Exacta": "Poisson(λ)", "Aproximación": "N(λ, √λ)", "Si": "λ ≥ 10"},
                          {"Exacta": "Binomial(n, p)", "Aproximación": "Poisson(np)", "Si": "n ≥ 20, p ≤ 0.05"},
                          {"Exacta": "Hiperg.(N, K, n)", "Aproximación": "Binomial(n, K/N)", "Si": "n/N ≤ 0.05"}], hide_index=True, width="stretch")
        with c3.container(border=True):
            st.markdown("#### En el servidor · srv-01")
            st.markdown("En la hora pico llegan 5 000 paquetes y cada uno se pierde con p = 0.002. La Binomial exacta exige calcular C(5000, x); con Poisson(λ = 10) o Normal(10, 3.16) el cálculo es inmediato y el error es mínimo.")
            st.markdown("- Un log de 1 000 000 de líneas con tasa de error 0.00005: Poisson(50) ≈ N(50, 7.07).\n- Auditar 20 cuentas de 10 000: Hipergeométrica ≈ Binomial.")
    nav_buttons("aproximaciones")


# ---------- Muestreo ----------
def page_muestreo():
    page_head("muestreo", "t de Student, Chi-cuadrado y F", "Las distribuciones de muestreo describen cómo varía un estadístico (x̄, s², un cociente de varianzas) de una muestra a otra. Son la base de los intervalos de confianza y de las pruebas de hipótesis.")
    choice = st.radio("Distribución", ["t", "chi2", "f"], horizontal=True, key="samp_choice",
                      format_func={"t": "t de Student · media con σ desconocida", "chi2": "Chi-cuadrado χ² · varianza y bondad de ajuste", "f": "F de Fisher · comparar dos varianzas"}.get, label_visibility="collapsed")
    st.markdown(f"<div class='lead'>{D[choice]['lead']}</div>", unsafe_allow_html=True)
    workbench(choice)
    nav_buttons("muestreo")


# ---------- Duelos ----------
DUELS = {
 "Binomial vs Hipergeométrica": dict(sub="con y sin reemplazo", A=("Binomial", C["disc"]), B=("Hipergeométrica", PINS[0]),
    params=[("N", "N · tamaño del lote", 10, 300, 30), ("K", "K · defectuosos en el lote", 1, 300, 12), ("n", "n · tamaño de la muestra", 1, 40, 10)],
    xs=lambda p: np.arange(0, p["n"] + 1), fa=lambda x, p: stats.binom.pmf(x, p["n"], p["K"] / p["N"]), fb=lambda x, p: stats.hypergeom.pmf(x, p["N"], p["K"], p["n"]), xlabel="x = defectuosos en la muestra",
    stats=lambda p: [("Media", fmt(p["n"] * p["K"] / p["N"], 3), fmt(p["n"] * p["K"] / p["N"], 3)), ("Varianza", fmt(p["n"] * p["K"] / p["N"] * (1 - p["K"] / p["N"]), 3), fmt(p["n"] * p["K"] / p["N"] * (1 - p["K"] / p["N"]) * (p["N"] - p["n"]) / (p["N"] - 1), 3)),
                     ("Factor (N − n)/(N − 1)", "1", fmt((p["N"] - p["n"]) / (p["N"] - 1), 3)), ("n/N", fmt(p["n"] / p["N"], 3), fmt(p["n"] / p["N"], 3))],
    same=["Cuentan éxitos en una muestra de tamaño n.", "Tienen la misma media: n·K/N."],
    diff=["Binomial: con reemplazo; p = K/N es igual en cada extracción.", "Hipergeométrica: sin reemplazo; la probabilidad cambia tras cada extracción.", "La Hipergeométrica tiene menor varianza: se multiplica por (N − n)/(N − 1).", "Con N mucho mayor que n (n/N ≤ 5 %) casi coinciden. Prueba subir N."],
    rule="¿Se devuelve el elemento o la población es enorme? → **Binomial**. ¿Se saca de un lote pequeño sin devolver? → **Hipergeométrica**."),
 "Binomial vs Poisson": dict(sub="¿intentos o intervalo?", A=("Binomial", C["disc"]), B=("Poisson", PINS[0]),
    params=[("n", "n · ensayos", 5, 300, 20), ("p", "p · probabilidad de éxito", 0.005, 0.6, 0.2)],
    xs=lambda p: np.arange(0, min(p["n"], int(p["n"] * p["p"] + 5 * math.sqrt(p["n"] * p["p"]) + 5)) + 1), fa=lambda x, p: stats.binom.pmf(x, p["n"], p["p"]), fb=lambda x, p: stats.poisson.pmf(x, p["n"] * p["p"]), xlabel="x = número de éxitos / eventos",
    stats=lambda p: [("Media", fmt(p["n"] * p["p"], 3), fmt(p["n"] * p["p"], 3)), ("Varianza", fmt(p["n"] * p["p"] * (1 - p["p"]), 3), fmt(p["n"] * p["p"], 3)), ("Valor máximo de X", f"n = {p['n']}", "sin tope"),
                     ("Error máximo", "—", fmt(float(np.max(np.abs(stats.binom.pmf(np.arange(p["n"] + 1), p["n"], p["p"]) - stats.poisson.pmf(np.arange(p["n"] + 1), p["n"] * p["p"])))), 4))],
    same=["Las dos cuentan eventos: valores enteros 0, 1, 2, …", "Con λ = np tienen la misma media."],
    diff=["Binomial: número fijo n de intentos; tiene tope n.", "Poisson: cuenta en un intervalo continuo, sin número de intentos y sin tope.", "Binomial: varianza np(1 − p), menor que la media. Poisson: varianza = media.", "Con n grande y p pequeña son casi iguales. Prueba n = 300, p = 0.01."],
    rule="¿Hay un número fijo de intentos? → **Binomial**. ¿Se cuentan eventos en un intervalo de tiempo sin tope claro? → **Poisson**."),
 "Poisson vs Exponencial": dict(sub="lluvia de peticiones", A=("Poisson", C["disc"]), B=("Exponencial", C["cont"]), custom=True,
    params=[("l", "λ · peticiones por segundo", 0.5, 8.0, 3.0)],
    same=["Describen el mismo proceso de llegadas con la misma tasa λ.", "Si las llegadas son Poisson, los tiempos entre ellas son Exponenciales."],
    diff=["Poisson: discreta, cuenta cuántas peticiones hay en 1 segundo.", "Exponencial: continua, mide el tiempo entre dos peticiones.", "Poisson: media λ. Exponencial: media 1/λ.", "Si λ sube, la Poisson se mueve a la derecha y la Exponencial se comprime hacia 0."],
    rule="¿La pregunta es “cuántos”? → **Poisson**. ¿Es “cuánto tiempo”? → **Exponencial**."),
 "Normal Z vs t de Student": dict(sub="colas gruesas", A=("Normal Z", C["cont"]), B=("t de Student", C["samp"]), cont=True,
    params=[("v", "ν · grados de libertad", 1, 60, 4)], x0=-5, x1=5, fa=lambda x, p: stats.norm.pdf(x), fb=lambda x, p: stats.t.pdf(x, p["v"]), xlabel="z  /  t",
    stats=lambda p: [("Varianza", "1", fmt(p["v"] / (p["v"] - 2), 3) if p["v"] > 2 else "∞"), ("P(|X| > 2)", fmt(2 * stats.norm.sf(2), 4), fmt(2 * stats.t.sf(2, p["v"]), 4)),
                     ("Crítico 95 % (dos colas)", "±1.96", "±" + fmt(stats.t.ppf(0.975, p["v"]), 3))],
    same=["Simétricas, centradas en 0 y con forma de campana.", "Se usan para inferencias sobre la media."],
    diff=["La t tiene colas más gruesas: más valores extremos.", "La t depende de los grados de libertad; la Z no.", "Z: σ conocida o n grande. t: σ estimada con s y n pequeña.", "Con ν ≥ 30 son casi iguales. Prueba subir ν."],
    rule="¿Conoces σ o tienes una muestra grande? → **Z**. ¿Estimas σ con s y tienes pocos datos? → **t**."),
 "Uniforme discreta vs continua": dict(sub="barras o rectángulo", A=("Uniforme discreta", C["disc"]), B=("Uniforme continua", C["cont"]), mixed=True,
    params=[("a", "a · mínimo", 0, 10, 1), ("b", "b · máximo", 1, 20, 6)], xlabel="x",
    stats=lambda p: [("Media", fmt((p["a"] + p["b"]) / 2, 3), fmt((p["a"] + p["b"]) / 2, 3)), ("Varianza", fmt(((p["b"] - p["a"] + 1) ** 2 - 1) / 12, 3), fmt((p["b"] - p["a"]) ** 2 / 12, 3)),
                     ("P(X = a)", fmt(1 / (p["b"] - p["a"] + 1), 4), "0"), ("Altura", f"1/{p['b'] - p['a'] + 1} (probabilidad)", f"1/{p['b'] - p['a']} (densidad)")],
    same=["Todos los valores posibles son igual de probables.", "Misma media: (a + b)/2."],
    diff=["Discreta: solo enteros; P(X = x) = 1/(b − a + 1).", "Continua: cualquier real del intervalo; P(X = x) = 0 y se calculan áreas.", "Varianzas distintas: (k² − 1)/12 frente a (b − a)²/12.", "Barras de altura “probabilidad” frente a una densidad (altura × ancho = probabilidad)."],
    rule="¿Los resultados son enteros (servidor 1…8, un dado)? → **discreta**. ¿Es un tiempo o una medida? → **continua**."),
 "Bernoulli vs Binomial": dict(sub="uno o n intentos", A=("Bernoulli", PINS[0]), B=("Binomial", C["disc"]),
    params=[("p", "p · probabilidad de éxito", 0.05, 0.95, 0.3), ("n", "n · ensayos de la Binomial", 1, 30, 8)],
    xs=lambda p: np.arange(0, p["n"] + 1), fa=lambda x, p: stats.bernoulli.pmf(x, p["p"]), fb=lambda x, p: stats.binom.pmf(x, p["n"], p["p"]), xlabel="x = éxitos",
    stats=lambda p: [("Valores posibles", "0 o 1", f"0 … {p['n']}"), ("Media", fmt(p["p"], 3), fmt(p["n"] * p["p"], 3)), ("Varianza", fmt(p["p"] * (1 - p["p"]), 3), fmt(p["n"] * p["p"] * (1 - p["p"]), 3))],
    same=["Mismo parámetro p y ensayos de éxito o fracaso.", "Con n = 1 la Binomial es exactamente una Bernoulli."],
    diff=["Bernoulli: un solo ensayo, valores 0 o 1.", "Binomial: n ensayos, valores 0 … n.", "La Binomial es la suma de n Bernoulli independientes: media y varianza se multiplican por n."],
    rule="¿Un solo paquete? → **Bernoulli**. ¿Cuántos de n paquetes? → **Binomial**."),
}


def duel_sliders(name, d):
    s = st.session_state; p = {}
    for k, lab, mn, mx, v in d["params"]:
        key = f"duel_{name}_{k}"; s.setdefault(key, v)
        if name.startswith("Binomial vs Hiper") and k in ("K", "n"):
            mx = min(mx, s[f"duel_{name}_N"]); s[key] = min(s[key], mx)
        if name.startswith("Uniforme") and k == "b":
            mn = s[f"duel_{name}_a"] + 1; s[key] = max(s[key], mn)
        isint = isinstance(v, int) and isinstance(mn, int)
        step = 1 if isint else (0.005 if mn < 0.1 else 0.05 if mx <= 1 else 0.5)
        cast = int if isint else float
        p[k] = st.slider(f"**{lab}**", cast(mn), cast(mx), step=cast(step), key=key)
    return p


def rain_section(p):
    l = p["l"]; rng = np.random.default_rng()
    st.caption("Cada punto es una petición que llega a srv-01. Pulsa ▶ para ver 20 segundos de llegadas. Arriba de cada segundo está cuántas llegaron (eso alimenta la Poisson); la separación entre puntos consecutivos alimenta la Exponencial.")
    gaps = rng.exponential(1 / l, int(l * 30) + 50); arr = np.cumsum(gaps); arr = arr[arr < 20]
    counts = np.bincount(np.floor(arr).astype(int), minlength=20)[:20]
    frames = []
    for t in np.arange(0.25, 20.01, 0.25):
        ws = 0 if t <= 10 else 10; vis = arr[(arr >= ws) & (arr <= t)]
        done = [s_ for s_ in range(ws, min(int(t), ws + 10))]
        frames.append(go.Frame(name=f"{t:.2f}", data=[
            go.Scatter(x=vis, y=np.zeros_like(vis), mode="markers", marker=dict(size=12, color=C["disc"])),
            go.Scatter(x=[s_ + 0.5 for s_ in done], y=[0.8] * len(done), mode="text", text=[str(counts[s_]) for s_ in done], textfont=dict(family="IBM Plex Mono, Consolas, monospace", size=16, color=C["disc"])),
            go.Scatter(x=[t, t], y=[-0.4, 1], mode="lines", line=dict(color=C["ink"], width=2))],
            layout=dict(xaxis=dict(range=[ws, ws + 10]))))
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(height=200, template="plotly_dark", showlegend=False, margin=dict(l=20, r=20, t=10, b=30),
                      xaxis=dict(range=[0, 10], dtick=1, ticksuffix=" s", gridcolor=C["line"]), yaxis=dict(visible=False, range=[-0.5, 1.1]),
                      updatemenus=[dict(type="buttons", x=0, y=1.25, xanchor="left", direction="left", buttons=[
                          dict(label="▶ Iniciar", method="animate", args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                          dict(label="⏸ Pausar", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])])
    show(fig, key=f"rain_{l}")
    st.markdown("#### Con muchos segundos de datos")
    secs = st.select_slider("Segundos simulados", [30, 100, 300, 1000, 3000], value=300, key="rain_secs")
    big = np.cumsum(rng.exponential(1 / l, int(l * secs * 1.3) + 100)); big = big[big < secs]
    cnt = np.bincount(np.floor(big).astype(int), minlength=secs)[:secs]; g = np.diff(big)
    c1, c2 = st.columns(2)
    with c1:
        hi = int(l + 5 * math.sqrt(l) + 3); xs = np.arange(hi + 1)
        fig = base_fig("peticiones en 1 segundo", "frecuencia relativa", 300)
        fig.add_bar(x=xs, y=np.bincount(cnt, minlength=hi + 1)[:hi + 1] / len(cnt), marker_color=hexa(C["sim"], 0.7), name="observado")
        fig.add_scatter(x=xs, y=stats.poisson.pmf(xs, l), mode="lines+markers", line=dict(color=C["disc"]), name=f"Poisson({l})")
        st.markdown("**Peticiones por segundo → Poisson(λ)**"); show(fig, key=f"rain_p_{l}_{secs}")
        st.caption(f"{secs} segundos · media observada {fmt(cnt.mean(), 3)} (teórica λ = {l}) · varianza {fmt(cnt.var(ddof=1), 3)}")
    with c2:
        x1 = 5 / l; fig = base_fig("segundos entre peticiones", "densidad", 300)
        dens, edges = np.histogram(g, bins=25, range=(0, x1)); dens = dens / (len(g) * (edges[1] - edges[0]))
        fig.add_bar(x=(edges[:-1] + edges[1:]) / 2, y=dens, width=(edges[1] - edges[0]) * 0.94, marker_color=hexa(C["sim"], 0.7), name="observado")
        xx = np.linspace(0, x1, 300); fig.add_scatter(x=xx, y=l * np.exp(-l * xx), line=dict(color=C["cont"], width=3), name=f"Exponencial({l})")
        st.markdown("**Tiempo entre peticiones → Exponencial(λ)**"); show(fig, key=f"rain_e_{l}_{secs}")
        st.caption(f"{len(g):,} intervalos · media observada {fmt(g.mean(), 3)} s (teórica 1/λ = {fmt(1 / l, 3)} s)")


def page_duelos():
    page_head("duelos", "Duelos de distribuciones", "Pares que se confunden con facilidad. Cada duelo los muestra en la misma gráfica, con los mismos parámetros, y termina con una regla rápida para decidir cuál usar.")
    name = st.radio("Duelo", list(DUELS), horizontal=True, key="duel", index=2, format_func=lambda n: f"{n}")
    d = DUELS[name]
    left, right = st.columns([2.6, 1], gap="large")
    with right:
        with st.container(border=True):
            st.markdown("#### Parámetros compartidos"); p = duel_sliders(name, d)
            if not d.get("custom"):
                st.dataframe([{"": r[0], d["A"][0]: r[1], d["B"][0]: r[2]} for r in d["stats"](p)], hide_index=True, width="stretch")
    with left:
        if d.get("custom"):
            rain_section(p)
        else:
            h = 520 if video_mode() else 430
            if d.get("cont"):
                fig = base_fig(d["xlabel"], "f(x)", h); xx = np.linspace(d["x0"], d["x1"], 600)
                fig.add_scatter(x=xx, y=d["fa"](xx, p), line=dict(color=d["A"][1], width=3), fill="tozeroy", fillcolor=hexa(d["A"][1], 0.08), name=d["A"][0])
                fig.add_scatter(x=xx, y=d["fb"](xx, p), line=dict(color=d["B"][1], width=3), name=d["B"][0])
            elif d.get("mixed"):
                a, b = p["a"], p["b"]; k = b - a + 1; fig = base_fig(d["xlabel"], "P(X = x)  ·  f(x)", h)
                fig.add_scatter(x=[a - 1.2, a, a, b, b, b + 1.2], y=[0, 0, 1 / (b - a), 1 / (b - a), 0, 0], fill="tozeroy", fillcolor=hexa(d["B"][1], 0.15), line=dict(color=d["B"][1], width=3), name=d["B"][0])
                fig.add_bar(x=np.arange(a, b + 1), y=[1 / k] * k, width=0.3, marker_color=d["A"][1], name=d["A"][0]); fig.update_xaxes(dtick=1)
            else:
                xs = d["xs"](p); fig = base_fig(d["xlabel"], "P(X = x)", h)
                fig.add_bar(x=xs - 0.2, y=d["fa"](xs, p), width=0.38, marker_color=d["A"][1], name=d["A"][0])
                fig.add_bar(x=xs + 0.2, y=d["fb"](xs, p), width=0.38, marker_color=d["B"][1], name=d["B"][0])
                fig.update_xaxes(dtick=1 if len(xs) <= 25 else None)
            show(fig, key=f"duel_fig_{name}")
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("#### En qué se parecen"); st.markdown("\n".join(f"- {x}" for x in d["same"]))
    with c2.container(border=True):
        st.markdown("#### En qué se diferencian"); st.markdown("\n".join(f"- {x}" for x in d["diff"]))
    st.info(d["rule"])
    nav_buttons("duelos")


# ════════════════════════════ Barra lateral y navegación ════════════════════════════
if "page" not in st.session_state:
    _q = st.query_params.get("p", "inicio")
    st.session_state["page"] = _q if _q in NAV_IDS else "inicio"
with st.sidebar:
    st.markdown("<div class='eyebrow'>Unidad 3 · Probabilidad</div><div style=\"font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:26px;line-height:1.05;color:#E4EAF2\">Atlas de Distribuciones</div><div style='color:#93A0B2;font-size:13px;margin:6px 0 10px'>Un día en la vida de srv-01, contado con probabilidad.</div>", unsafe_allow_html=True)
    dots = {"disc": "🔵", "cont": "🟠", "aprox": "🟣", "samp": "🟢", "none": "⚫"}
    st.radio("Recorrido del día", NAV_IDS, key="page", format_func=lambda i: f"{NAVD[i][1]}  {dots[NAVD[i][3]]}  {NAVD[i][2]}", label_visibility="collapsed")
    st.markdown("<div style='font-size:12px;color:#93A0B2;margin-top:8px'>🔵 Discreta · 🟠 Continua · 🟣 Aproximación · 🟢 Muestreo</div>", unsafe_allow_html=True)
    st.divider()
    st.toggle("Modo video", key="video", help="Oculta las tarjetas de teoría y agranda las gráficas para grabar la pantalla.")
    st.caption("Consejo: para grabar, colapsa esta barra con la flecha « y usa los botones de abajo de cada página para avanzar por el día.")

PAGES = {"inicio": page_inicio, "multinomial": page_multinomial, "aproximaciones": page_aprox, "muestreo": page_muestreo, "duelos": page_duelos}
pid = st.session_state["page"]
if pid in PAGES:
    PAGES[pid]()
else:
    page_dist(pid)
    nav_buttons(pid)
