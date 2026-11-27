#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J farm
#SBATCH -o logs/farm_%A_%a.out

# Cada tarea del array es un nodo completo que toma bloques libres de la lista de farm.py (en orden de
# prioridad, con locks). El número de tareas no importa: se pueden enviar más, o volver a enviar una sola.
# preempt: factor 1 las primeras 2 h y 0.5 después; si lo interrumpen, --requeue lo vuelve a encolar,
# la tarea recupera su lock y farm.py se salta los bloques válidos que ya existen en datos/.
#
# Enviar desde esta carpeta:  mkdir -p logs && sbatch --array=0-2 farm.sbatch
# Avance:                     python farm.py --estado

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# código congelado por envío: todas las tareas, y sus reencolados, corren la misma copia de farm.py
# aunque farm.py se edite mientras el job está en la cola
F=logs/farm_${SLURM_ARRAY_JOB_ID}.py
[ -e "$F" ] || { cp farm.py "$F.$SLURM_ARRAY_TASK_ID" && mv -f "$F.$SLURM_ARRAY_TASK_ID" "$F"; }

echo "inicio $(date) en $(hostname), job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, reinicio ${SLURM_RESTART_COUNT:-0}, código $F"
srun -n 1 -c 256 /global/homes/j/jairf/.conda/envs/cosmo/bin/python "$F" --salida datos
echo "fin $(date)"
