# Template Beamer UNAM

Template mínimo de presentaciones Beamer con la identidad de la UNAM
(Universidad Nacional Autónoma de México). Formato 16:9.

Adaptado del tema UVT de Alexandru Fikl
([alexfikl/uvt-beamer](https://github.com/alexfikl/uvt-beamer), v0.4, licencia CC-BY 4.0),
manteniendo el mismo estilo, colores, distribución y fuentes, pero con logos,
nombres y referencias de la UNAM.

## Estructura

```
unam-beamer-template/
├── template.tex               % documento de ejemplo (punto de partida)
├── template.pdf               % PDF compilado del ejemplo
├── README.md
├── beamerthemeunam.sty        % tema principal (\usetheme{unam})
├── beamercolorthemeunam.sty   % colores (\usecolortheme{unam})
├── assets/
│   ├── unam-logo.png          % logo con texto, va en el encabezado de cada diapositiva
│   └── unam-escudo-gris.png   % escudo gris: fondo de la portada y marca de agua
└── images/                    % figuras de la presentación (\graphicspath{{images/}})
```

## Uso

Copia la carpeta completa y edita `template.tex`. Compila con `pdflatex`
(o `latexmk -pdf template.tex`). Los `.sty` y la carpeta `assets/` deben
estar junto al `.tex`.

```latex
\usetheme[nomyriadpro]{unam}
```

### Opciones del tema

| Opción        | Descripción                                          |
| :-            | :-                                                   |
| `nomyriadpro` | No cargar *Myriad Pro*; usa Helvetica (recomendado)  |
| `progressbar` | Barra de progreso dorada en el pie de página         |
| `showframe`   | Muestra los márgenes (depuración)                    |
| `layoutgrid`  | Cuadrícula para revisar alineación (depuración)      |

### Macros

| Macro              | Descripción                                              |
| :-                 | :-                                                       |
| `\unamheaderlogo`  | Logo del encabezado (por defecto `assets/unam-logo`)     |
| `\unamtitlelogo`   | Logo de la portada y de las portadas de sección          |
| `\unamslidelogo`   | Marca de agua transparente en cada diapositiva           |
| `\unammotto`       | Lema en el pie de página (vacío por defecto)             |
| `\venue`           | Texto debajo de la fecha en la portada                   |
| `\titleframe`      | Genera la portada                                        |

Cada `\section{}` genera automáticamente una diapositiva de sección.

### Colores

| Nombre          | RGB            | Uso                                  |
| :-              | :-             | :-                                   |
| `UNAMDarkBlue`  | 3, 75, 119     | Títulos, bloques                     |
| `UNAMLightBlue` | 52, 113, 183   | Bloques de ejemplo, degradado        |
| `UNAMGold`      | 228, 172, 36   | Viñetas, `\alert`, bloques de alerta |

En `template.tex` también se definen `myblue`, `myred`, `mygreen`,
`mypurple` y `myorange` para diagramas TikZ.

## Contenido del ejemplo

Portada, portada de sección, listas (`itemize` en tres niveles, `enumerate`,
`description`), bloques (`block`, `alertblock`, `exampleblock`), ecuaciones,
columnas con figura, animaciones con `[<+->]` y diapositiva de cierre.

## Licencia

Creative Commons Attribution 4.0 International (heredada del tema original).
Los logotipos y el escudo son propiedad de la UNAM.
