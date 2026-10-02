; Implementación escalar SSE para Linux x86-64, ABI System V.
; El programa principal rechaza n <= 0. Aquí se devuelve cero sin leer
; el vector si se recibe ese tamaño; los punteros de salida deben ser válidos.
; ABI: RBX/RBP/R12-R15 y RSP deben conservarse; aquí no se modifican.
; RAX/RCX/RDX/RSI/RDI/R8-R11 y XMM/YMM son volátiles, por eso no se
; guardan con push/pop. RET usa la dirección de retorno apilada por CALL.
; global publica los símbolos para que el enlazador resuelva las llamadas C.
bits 64
default rel

section .text
global sum_array
global compute_stats
global normalize_array

; float sum_array(const float *arr, int n)
; rdi = arr, esi = n; resultado xmm0. Solo registros volátiles.
sum_array:
    xorps xmm0, xmm0
    xorps xmm8, xmm8
    test esi, esi
    jle .done
    xor eax, eax
.loop:
    addss xmm0, [rdi + rax*4]
    inc eax
    ; Cada 4096 elementos, incorporar la suma parcial al total float32.
    test eax, 4095
    jnz .next
    addss xmm8, xmm0
    xorps xmm0, xmm0
.next:
    cmp eax, esi
    jl .loop
.done:
    addss xmm0, xmm8
    ret

; void compute_stats(const float *arr, int n, float *mean,
;                    float *variance, float *min, float *max)
; rdi, esi, rdx, rcx, r8, r9 según System V.
; xmm0 = suma/media, xmm1 = mínimo, xmm2 = máximo,
; xmm3 = n, xmm4 = suma de cuadrados del bloque, xmm5 = auxiliar,
; xmm8 = total entre bloques (primero suma, después cuadrados).
compute_stats:
    xorps xmm0, xmm0
    xorps xmm1, xmm1
    xorps xmm2, xmm2
    xorps xmm4, xmm4
    xorps xmm8, xmm8
    test esi, esi
    jle .store
    movss xmm1, [rdi]
    movss xmm2, xmm1
    xor eax, eax
.first_pass:
    movss xmm5, [rdi + rax*4]
    addss xmm0, xmm5
    minss xmm1, xmm5
    maxss xmm2, xmm5
    inc eax
    test eax, 4095
    jnz .first_next
    addss xmm8, xmm0
    xorps xmm0, xmm0
.first_next:
    cmp eax, esi
    jl .first_pass
    addss xmm0, xmm8
    cvtsi2ss xmm3, esi
    divss xmm0, xmm3
    xorps xmm8, xmm8
    xor eax, eax
; Segunda pasada: acumular (x-media)^2; al final dividir entre N.
.second_pass:
    movss xmm5, [rdi + rax*4]
    subss xmm5, xmm0
    mulss xmm5, xmm5
    addss xmm4, xmm5
    inc eax
    test eax, 4095
    jnz .second_next
    addss xmm8, xmm4
    xorps xmm4, xmm4
.second_next:
    cmp eax, esi
    jl .second_pass
    addss xmm4, xmm8
    divss xmm4, xmm3
.store:
    movss [rdx], xmm0
    movss [rcx], xmm4
    movss [r8], xmm1
    movss [r9], xmm2
    ret

; void normalize_array(const float *in, float *out, int n,
;                      float mean, float stddev)
; rdi = in, rsi = out, edx = n, xmm0 = media, xmm1 = sigma.
; Se permite in == out. Una sigma nula produce ceros.
normalize_array:
    test edx, edx
    jle .done
    xor eax, eax
    xorps xmm2, xmm2
    ucomiss xmm1, xmm2
    jp .loop
    je .zero_loop
.loop:
    movss xmm2, [rdi + rax*4]
    subss xmm2, xmm0
    divss xmm2, xmm1
    movss [rsi + rax*4], xmm2
    inc eax
    cmp eax, edx
    jl .loop
    ret
.zero_loop:
    movss [rsi + rax*4], xmm2
    inc eax
    cmp eax, edx
    jl .zero_loop
.done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
