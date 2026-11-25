#!/usr/bin/env bash
# Arma el video de la animación del toy model para la presentación (pympress + multimedia).
#
# Uso (desde esta carpeta):
#   ./hacer_video.sh            # usa los PNG que ya están en frames/
#   ./hacer_video.sh --todo     # además recompila anim_toy.tex y vuelve a sacar los PNG
#
# Resultado: ../animacion_toy.mp4, junto a dilution.pdf (el PDF solo apunta al video).
# 12 s a 24 cuadros/s: 0-4 s sin dilución, 4-10 s dilución, 10-12 s estado final fijo.
# Necesita ffmpeg con libx264; para --todo también pdflatex y ghostscript.
set -euo pipefail
cd "$(dirname "$0")"

FPS=24
ANCHO=1920   # píxeles de ancho del video

if [[ "${1:-}" == "--todo" ]]; then
    pdflatex -interaction=nonstopmode anim_toy.tex > /dev/null
    rm -rf frames && mkdir frames
    # resolución para que la página mida ANCHO píxeles de ancho
    ANCHO_PT=$(gs -q -dNODISPLAY -dNOSAFER -c "(anim_toy.pdf) (r) file runpdfbegin 1 pdfgetpage /MediaBox pget pop aload pop pop pop exch pop == quit")
    DPI=$(python3 -c "print(round($ANCHO*72/$ANCHO_PT, 3))")
    gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r"$DPI" \
       -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
       -sOutputFile=frames/frame_%03d.png anim_toy.pdf
    # 2 s finales fijos: repetir el último cuadro (t = 10 s) 2*FPS veces
    ULT=$(ls frames | sort | tail -1)
    N=$(ls frames | wc -l)
    for ((j = N + 1; j <= N + 2*FPS; j++)); do
        cp "frames/$ULT" "$(printf 'frames/frame_%03d.png' "$j")"
    done
fi

# gs numera desde 001
ffmpeg -y -framerate "$FPS" -i frames/frame_%03d.png \
       -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" \
       -c:v libx264 -crf 18 -preset slow -movflags +faststart \
       ../animacion_toy.mp4

echo "Listo: $(cd .. && pwd)/animacion_toy.mp4"
