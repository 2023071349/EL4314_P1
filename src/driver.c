#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Declaraciones de las funciones DEFINIDAS EN NASM, no implementaciones C.
   El compilador utiliza estos prototipos para preparar cada llamada; el
   enlazador resuelve los símbolos global de uno de los dos archivos .asm.
   El Makefile enlaza una sola variante por ejecutable, pues ambas exportan
   los mismos nombres. No se necesita un archivo de encabezado independiente.

   Convención System V AMD64 (Linux x86-64): los argumentos enteros/punteros
   ocupan RDI, RSI, RDX, RCX, R8 y R9; los float usan, por separado, XMM0,
   XMM1, etc. Un int de 32 bits se lee en ESI o EDX, según su posición.
   - sum_array: arr en RDI, n en ESI; devuelve float en XMM0.
   - compute_stats: arr en RDI, n en ESI, &mean en RDX, &var en RCX,
     &min en R8, &max en R9. Devuelve void: escribe mediante esos punteros.
   - normalize_array: in en RDI, out en RSI, n en EDX, mean en XMM0 y
     stddev en XMM1. Los float no consumen posiciones de registros enteros.

   El llamado (NASM) debe conservar RBX, RBP y R12-R15, y restituir RSP.
   RAX, RCX, RDX, RSI, RDI, R8-R11 y todos los XMM/YMM son volátiles:
   el llamador (C, mediante GCC) guarda lo que necesite conservar allí.
   Estos kernels solo usan registros volátiles y no hacen llamadas internas.
   GCC alinea RSP a 16 bytes antes de CALL; al entrar, RSP+8 está alineado,
   porque CALL apila la dirección de retorno. RET la recupera. La alineación
   de la pila es distinta de los 32 bytes exigidos para las cargas YMM.
   Contrato: n > 0, datos finitos y arreglos de n float alineados a 32 bytes. */
extern float sum_array(float *arr, int n);
extern void compute_stats(float *arr, int n, float *mean, float *var,
                          float *min, float *max);
extern void normalize_array(float *in, float *out, int n,
                            float mean, float stddev);

_Static_assert(sizeof(int) == 4, "Se requiere int de 32 bits");
_Static_assert(sizeof(float) == 4, "Se requiere float de 32 bits");

/* Contador TSC ordenado. No es el contador de ciclos de CPU de perf:
   avanza a frecuencia de referencia, aunque cambie la frecuencia del núcleo. */
static unsigned long long read_ticks(void)
{
    unsigned int low, high;
    __asm__ volatile("lfence\n\trdtsc\n\tlfence"
                     : "=a"(low), "=d"(high) : : "memory");
    return ((unsigned long long)high << 32) | low;
}

/* Flujo: interpretar argumentos -> leer/validar -> reservar memoria ->
   llamar a NASM -> validar resultados -> escribir binario y JSON -> liberar.
   argc incluye el nombre del programa; argv contiene las rutas opcionales
   y, como cuarto argumento del usuario, el número de repeticiones. */
int main(int argc, char **argv)
{
    FILE *file = NULL;
    float *input = NULL, *output = NULL;
    int n, status = EXIT_FAILURE;
    int repetitions = 1;
    double *samples_ms = NULL;
    unsigned long long *samples_ticks = NULL;
    float sum, mean, variance, minimum, maximum, stddev;
    struct timespec before, after;
    unsigned long long ticks_before, ticks_after;
    unsigned long long ticks = 0;
    double milliseconds;
    const char *input_path = argc > 1 ? argv[1] : "input.dat";
#ifdef VECTOR_BUILD
    const char *output_path = argc > 2 ? argv[2] : "output_vector.dat";
    const char *stats_path = argc > 3 ? argv[3] : "stats_vector.json";
    /* GCC comprueba CPU y soporte del SO antes de llamar a NASM AVX2.
       Esto no calcula datos ni usa intrínsecos SIMD. */
    if (!__builtin_cpu_supports("avx2")) {
        fprintf(stderr, "ERROR: AVX2 no disponible en este equipo\n");
        return EXIT_FAILURE;
    }
#else
    const char *output_path = argc > 2 ? argv[2] : "output_scalar.dat";
    const char *stats_path = argc > 3 ? argv[3] : "stats_scalar.json";
#endif

    if (argc > 5) {
        fprintf(stderr, "Uso: %s [entrada.dat [salida.dat [estadisticas.json [repeticiones]]]]\n", argv[0]);
        return EXIT_FAILURE;
    }
    /* strtol permite detectar texto sobrante y desbordamiento con errno. */
    if (argc == 5) {
        char *end;
        errno = 0;
        long value = strtol(argv[4], &end, 10);
        if (errno || end == argv[4] || *end || value < 1 || value > 100000) {
            fprintf(stderr, "ERROR: repeticiones debe ser un entero entre 1 y 100000\n");
            return EXIT_FAILURE;
        }
        repetitions = (int)value;
    }
    samples_ms = malloc((size_t)repetitions * sizeof(*samples_ms));
    samples_ticks = malloc((size_t)repetitions * sizeof(*samples_ticks));
    if (!samples_ms || !samples_ticks) {
        fprintf(stderr, "ERROR: memoria insuficiente para mediciones\n");
        goto cleanup;
    }
    file = fopen(input_path, "rb");
    if (!file) {
        perror("Abrir entrada");
        goto cleanup;
    }
    /* Entrada BINARIA: int32 N (4 bytes), seguido de N valores IEEE 754
       float32 (4 bytes cada uno), sin texto ni separadores. Linux x86-64
       es little endian: el orden de bytes coincide con el generador. */
    if (fread(&n, sizeof(n), 1, file) != 1 || n <= 0) {
        fprintf(stderr, "ERROR: encabezado incompleto o N <= 0\n");
        goto cleanup;
    }
    /* Comprobar el tamaño antes de reservar memoria para una entrada dañada. */
    if (fseek(file, 0, SEEK_END) != 0) {
        perror("Examinar entrada");
        goto cleanup;
    }
    if (ftell(file) != 4L + 4L * n || fseek(file, 4, SEEK_SET) != 0) {
        fprintf(stderr, "ERROR: el tamaño de entrada no coincide con N\n");
        goto cleanup;
    }
    /* La reserva de 32 bytes de alineación habilita vmovaps con YMM.
       Cada posición i ocupa 4 bytes; el grupo de ocho ocupa 32. El arreglo
       de salida se reserva por separado. free libera ambas reservas. */
    if (posix_memalign((void **)&input, 32, (size_t)n * sizeof(float)) != 0 ||
        posix_memalign((void **)&output, 32, (size_t)n * sizeof(float)) != 0) {
        fprintf(stderr, "ERROR: memoria insuficiente\n");
        goto cleanup;
    }
    if (fread(input, sizeof(float), (size_t)n, file) != (size_t)n) {
        fprintf(stderr, "ERROR: lectura incompleta\n");
        goto cleanup;
    }
    if (fclose(file) != 0) {
        file = NULL;
        perror("Cerrar entrada");
        goto cleanup;
    }
    file = NULL;
    /* NaN e infinito no pertenecen al contrato numérico de los kernels. */
    for (int i = 0; i < n; ++i) {
        if (!isfinite(input[i])) {
            fprintf(stderr, "ERROR: entrada con NaN o infinito\n");
            goto cleanup;
        }
    }

    /* La E/S y las validaciones quedan fuera de la medición.
       Cada intervalo rodea una llamada NASM. sqrtf obtiene sigma de varianza. */
    /* En benchmark, repetición -1 calienta código, caché y páginas de salida.
       Se descarta. La ejecución normal (una repetición) no añade calentamiento. */
    for (int repetition = repetitions > 1 ? -1 : 0;
         repetition < repetitions; ++repetition) {
        milliseconds = 0;
        ticks = 0;
        if (clock_gettime(CLOCK_MONOTONIC, &before) != 0) goto clock_error;
        ticks_before = read_ticks();
        sum = sum_array(input, n);
        ticks_after = read_ticks();
        if (clock_gettime(CLOCK_MONOTONIC, &after) != 0) goto clock_error;
        milliseconds += (after.tv_sec - before.tv_sec) * 1000.0 +
                        (after.tv_nsec - before.tv_nsec) / 1000000.0;
        ticks += ticks_after - ticks_before;

        if (clock_gettime(CLOCK_MONOTONIC, &before) != 0) goto clock_error;
        ticks_before = read_ticks();
        /* & proporciona direcciones donde NASM escribe los cuatro float.
           Primera pasada: media = suma/N y extremos. Segunda pasada:
           varianza poblacional = suma((x-media)^2)/N, no división por N-1.
           Las acumulaciones por bloques limitan cadenas largas de redondeo. */
        compute_stats(input, n, &mean, &variance, &minimum, &maximum);
        ticks_after = read_ticks();
        if (clock_gettime(CLOCK_MONOTONIC, &after) != 0) goto clock_error;
        milliseconds += (after.tv_sec - before.tv_sec) * 1000.0 +
                        (after.tv_nsec - before.tv_nsec) / 1000000.0;
        ticks += ticks_after - ticks_before;
        /* sqrtf calcula sigma en precisión simple, fuera del intervalo NASM.
           Datos finitos pueden producir overflow: se comprueba el resultado. */
        stddev = sqrtf(variance);
        if (!isfinite(sum) || !isfinite(mean) || !isfinite(variance) || !isfinite(stddev)) {
            fprintf(stderr, "ERROR: desbordamiento en estadísticas float32\n");
            goto cleanup;
        }

        if (clock_gettime(CLOCK_MONOTONIC, &before) != 0) goto clock_error;
        ticks_before = read_ticks();
        /* z[i] = (x[i]-media)/sigma. Si sigma=0, NASM escribe ceros para
           evitar 0/0; ocurre, por ejemplo, con un arreglo constante. */
        normalize_array(input, output, n, mean, stddev);
        ticks_after = read_ticks();
        if (clock_gettime(CLOCK_MONOTONIC, &after) != 0) goto clock_error;
        milliseconds += (after.tv_sec - before.tv_sec) * 1000.0 +
                        (after.tv_nsec - before.tv_nsec) / 1000000.0;
        ticks += ticks_after - ticks_before;
        if (repetition >= 0) {
            samples_ms[repetition] = milliseconds;
            samples_ticks[repetition] = ticks;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (!isfinite(output[i])) {
            fprintf(stderr, "ERROR: normalización no finita\n");
            goto cleanup;
        }
    }

    /* Salida BINARIA: exactamente N float32 normalizados, SIN encabezado.
       La referencia conoce N a partir del archivo de entrada. */
    file = fopen(output_path, "wb");
    if (!file) { perror("Abrir salida"); goto cleanup; }
    if (fwrite(output, sizeof(float), (size_t)n, file) != (size_t)n) {
        fprintf(stderr, "ERROR: no se pudo escribir la salida\n");
        goto cleanup;
    }
    if (fclose(file) != 0) {
        file = NULL; perror("Cerrar salida"); goto cleanup;
    }
    /* Salida de estadísticas: JSON de texto, separado del binario. %.9g
       ofrece suficientes cifras para recuperar un float32. Las listas
       contienen todas las mediciones; kernel_ms/tsc_ticks son la última.
       Los float se promocionan a double al pasarlos a fprintf (varargs). */
    file = fopen(stats_path, "w");
    if (!file) { perror("Abrir estadísticas"); goto cleanup; }
    if (fprintf(file,
        "{\"n\":%d,\"sum\":%.9g,\"mean\":%.9g,\"var\":%.9g,"
        "\"stddev\":%.9g,\"min\":%.9g,\"max\":%.9g,"
        "\"kernel_ms\":%.9g,\"tsc_ticks\":%llu,\"repetitions\":%d,\"samples_ms\":[",
        n, sum, mean, variance, stddev, minimum, maximum, milliseconds, ticks, repetitions) < 0) {
        fprintf(stderr, "ERROR: no se pudieron escribir las estadísticas\n");
        goto cleanup;
    }
    for (int i = 0; i < repetitions; ++i)
        fprintf(file, "%s%.12g", i ? "," : "", samples_ms[i]);
    fprintf(file, "],\"samples_tsc\":[");
    for (int i = 0; i < repetitions; ++i)
        fprintf(file, "%s%llu", i ? "," : "", samples_ticks[i]);
    fprintf(file, "]}\n");
    if (ferror(file)) {
        fprintf(stderr, "ERROR: no se pudieron escribir las mediciones\n");
        goto cleanup;
    }
    if (fclose(file) != 0) {
        file = NULL; perror("Cerrar estadísticas"); goto cleanup;
    }
    file = NULL;
    printf("N=%d\nsuma=%.9g\nmedia=%.9g\nvarianza=%.9g\n"
           "desviacion=%.9g\nminimo=%.9g\nmaximo=%.9g\n"
           "kernel_ms=%.9g\nciclos_referencia_TSC=%llu\n",
           n, sum, mean, variance, stddev, minimum, maximum, milliseconds, ticks);
    status = EXIT_SUCCESS;
    goto cleanup;

clock_error:
    perror("clock_gettime");
cleanup:
    /* Ruta común de éxito y error: cerrar el archivo todavía abierto y
       liberar cada reserva. free(NULL) es válido; evita duplicar limpieza.
       EXIT_FAILURE se mantiene hasta completar todas las escrituras. */
    if (file) fclose(file);
    free(input);
    free(output);
    free(samples_ms);
    free(samples_ticks);
    return status;
}
