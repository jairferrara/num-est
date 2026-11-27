#!/usr/bin/env bash
# Arma el video de la animación del toy model para la presentación (pympress + multimedia).
#
# Uso (desde esta carpeta):
#   ./hacer_video.sh            # usa los PNG que ya están en frames/
#   ./hacer_video.sh --todo     # además recompila anim_toy.tex y vuelve a sacar los PNG
#
# Resultado: ../animacion_toy.webm (VP9), junto a dilution.pdf (el PDF solo apunta al video).
# 12 s a 24 cuadros/s: 0-4 s sin dilución, 4-10 s dilución, 10-12 s estado final fijo.
# Necesita ffmpeg con libvpx-vp9 (el ffmpeg de Perlmutter no trae libx264); para --todo también
# pdflatex y ghostscript. SALIDA=... cambia el archivo de salida.
set -euo pipefail
cd "$(dirname "$0")"

FPS=24
ANCHO=1920                                  # píxeles de ancho del video
SALIDA=${SALIDA:-../animacion_toy.webm}

if [[ "${1:-}" == "--todo" ]]; then
    pdflatex -interaction=nonstopmode "\def\fps{$FPS}\input{anim_toy}" > /dev/null
    rm -rf frames && mkdir frames
    # resolución alta fija (la página mide ~15 cm); ffmpeg la ajusta a ANCHO píxeles
    gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r400 \
       -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
       -sOutputFile=frames/frame_%03d.png anim_toy.pdf
fi

# gs numera desde 001
ffmpeg -y -framerate "$FPS" -i frames/frame_%03d.png \
       -vf "scale=$ANCHO:-2,format=yuv420p" \
       -c:v libvpx-vp9 -crf 30 -b:v 0 \
       "$SALIDA"

echo "Listo: $SALIDA"
