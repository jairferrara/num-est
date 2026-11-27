# Template de artículo UNAM

Template de escritos científicos (artículos, reportes, proyectos) en carta a
12 pt. El texto va completamente en negro; los colores del template Beamer UNAM
(`../unam-beamer-template`) se usan solo en los enlaces y están disponibles para
gráficas, de modo que las figuras del escrito y de la presentación rimen.

## Estructura

```
unam-paper-template/
├── template.tex      % documento de ejemplo (punto de partida)
├── template.pdf      % PDF compilado del ejemplo
├── unampaper.sty     % estilo: márgenes, título, índice, enlaces, colores
├── README.md
├── images/           % figuras del escrito (\graphicspath{{images/}})
└── ejemplo/          % escrito original del que se extrajo el formato
    ├── main.tex
    └── *.pdf
```

## Uso

Copia la carpeta completa y edita `template.tex`. Compila con `pdflatex`
(o `latexmk -pdf template.tex`). `unampaper.sty` debe estar junto al `.tex`.

```latex
\usepackage{unampaper}   % al final del preámbulo (carga hyperref)
```

### Formato

- `article`, carta, 12 pt, márgenes de 2 cm.
- Párrafos separados por `0.75em`, sin sangría.
- Español (`es-tabla`, punto decimal con `es-nodecimaldot`), numeración de páginas romana.
- Portada: título, autor, resumen e índice entre dos líneas.
- Secciones: Teoría, Cómputo, Resultados, Uso de los modelos LLM,
  Conclusiones y bibliografía con `thebibliography`.

### Macros

| Macro / entorno   | Descripción                                              |
| :-                | :-                                                       |
| `\maketitle`      | Título, autor y fecha (solo si se define `\date`)        |
| `\unamindice`     | Índice entre dos líneas                                  |
| `\unamrule`       | Línea a todo lo ancho                                    |
| `PromptBlock`     | Texto literal en letra pequeña, divisible entre páginas  |
| `\upcite{clave}`  | Cita en superíndice                                      |
| `\un{m/s}`        | Unidades separadas del número                            |
| `\mlr{expr}`      | Fórmula agrandada dentro del texto                       |

### Colores (solo enlaces y gráficas)

| Nombre          | RGB            | Uso en el documento                     |
| :-              | :-             | :-                                      |
| `UNAMDarkBlue`  | 3, 75, 119     | Referencias internas y citas            |
| `UNAMLightBlue` | 52, 113, 183   | URLs                                    |
| `UNAMGold`      | 228, 172, 36   | Disponible para gráficas                |
| `UNAMGray`      | 246, 246, 246  | Disponible para gráficas                |

En `template.tex` también se definen `myblue`, `myred`, `mygreen`, `mypurple`
y `myorange` para diagramas TikZ (los mismos del template de Beamer).

## Licencia

Creative Commons Attribution 4.0 International.
