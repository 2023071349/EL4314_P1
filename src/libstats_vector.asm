; AVX2, ocho float32 por iteración. Linux x86-64, ABI System V.
; El driver garantiza bases alineadas a 32 bytes y n > 0.
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

; RDI = arreglo, ESI = N. Devuelve la suma en XMM0.
; EAX = índice; R10D = N redondeado hacia abajo a múltiplo de ocho.
sum_array:
    vxorps ymm0, ymm0, ymm0
    vxorps xmm8, xmm8, xmm8
    test esi, esi
    jle .done
    xor eax, eax
    mov r10d, esi
    and r10d, -8
.vector_loop:
    cmp eax, r10d
    jge .reduce
    vmovaps ymm1, [rdi + rax*4]
    vaddps ymm0, ymm0, ymm1
    add eax, 8
    test eax, 4095
    jz .reduce
    jmp .vector_loop
.reduce:
    ; Ocho carriles -> cuatro -> dos -> uno.
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vaddss xmm8, xmm8, xmm0
    vxorps ymm0, ymm0, ymm0
    cmp eax, r10d
    jl .vector_loop
    vmovaps xmm0, xmm8
.tail_loop:
    cmp eax, esi
    jge .done
    vaddss xmm0, xmm0, [rdi + rax*4]
    inc eax
    jmp .tail_loop
.done:
    vzeroupper
    ret

; RDI = entrada, ESI = N; RDX/RCX/R8/R9 apuntan a media/var/min/max.
; Dos pasadas, igual que la versión escalar. Solo registros volátiles.
compute_stats:
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm4, ymm4, ymm4
    vxorps xmm8, xmm8, xmm8
    test esi, esi
    jle .store
    ; Inicializar todos los carriles con el primer valor permite N < 8.
    vbroadcastss ymm1, [rdi]
    vmovaps ymm2, ymm1
    xor eax, eax
    mov r10d, esi
    and r10d, -8
.first_vector:
    cmp eax, r10d
    jge .first_reduce
    vmovaps ymm3, [rdi + rax*4]
    vaddps ymm0, ymm0, ymm3
    vminps ymm1, ymm1, ymm3
    vmaxps ymm2, ymm2, ymm3
    add eax, 8
    test eax, 4095
    jz .sum_block
    jmp .first_vector
.sum_block:
    ; Reducir solo la suma: los mínimos/máximos siguen entre bloques.
    vextractf128 xmm6, ymm0, 1
    vaddps xmm0, xmm0, xmm6
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vaddss xmm8, xmm8, xmm0
    vxorps ymm0, ymm0, ymm0
    jmp .first_vector
.first_reduce:
    vextractf128 xmm6, ymm0, 1
    vaddps xmm0, xmm0, xmm6
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vaddss xmm0, xmm0, xmm8
    ; Mínimo: combinar mitades, parejas separadas por dos y adyacentes.
    vextractf128 xmm6, ymm1, 1
    vminps xmm1, xmm1, xmm6
    vpermilps xmm6, xmm1, 0x4e
    vminps xmm1, xmm1, xmm6
    vpermilps xmm6, xmm1, 0xb1
    vminps xmm1, xmm1, xmm6
    ; La misma reducción con máximo.
    vextractf128 xmm6, ymm2, 1
    vmaxps xmm2, xmm2, xmm6
    vpermilps xmm6, xmm2, 0x4e
    vmaxps xmm2, xmm2, xmm6
    vpermilps xmm6, xmm2, 0xb1
    vmaxps xmm2, xmm2, xmm6
.first_tail:
    cmp eax, esi
    jge .mean
    vmovss xmm3, [rdi + rax*4]
    vaddss xmm0, xmm0, xmm3
    vminss xmm1, xmm1, xmm3
    vmaxss xmm2, xmm2, xmm3
    inc eax
    jmp .first_tail
.mean:
    vxorps xmm7, xmm7, xmm7
    vcvtsi2ss xmm7, xmm7, esi
    vdivss xmm0, xmm0, xmm7
    vbroadcastss ymm5, xmm0
    vxorps xmm8, xmm8, xmm8
    xor eax, eax
; Segunda pasada vectorial: desviaciones al cuadrado en ocho carriles.
.second_vector:
    cmp eax, r10d
    jge .second_reduce
    vmovaps ymm3, [rdi + rax*4]
    vsubps ymm3, ymm3, ymm5
    vmulps ymm3, ymm3, ymm3
    vaddps ymm4, ymm4, ymm3
    add eax, 8
    test eax, 4095
    jz .square_block
    jmp .second_vector
.square_block:
    vextractf128 xmm6, ymm4, 1
    vaddps xmm4, xmm4, xmm6
    vhaddps xmm4, xmm4, xmm4
    vhaddps xmm4, xmm4, xmm4
    vaddss xmm8, xmm8, xmm4
    vxorps ymm4, ymm4, ymm4
    jmp .second_vector
.second_reduce:
    vextractf128 xmm6, ymm4, 1
    vaddps xmm4, xmm4, xmm6
    vhaddps xmm4, xmm4, xmm4
    vhaddps xmm4, xmm4, xmm4
    vaddss xmm4, xmm4, xmm8
.second_tail:
    cmp eax, esi
    jge .variance
    vmovss xmm3, [rdi + rax*4]
    vsubss xmm3, xmm3, xmm0
    vmulss xmm3, xmm3, xmm3
    vaddss xmm4, xmm4, xmm3
    inc eax
    jmp .second_tail
.variance:
    vdivss xmm4, xmm4, xmm7
.store:
    vmovss [rdx], xmm0
    vmovss [rcx], xmm4
    vmovss [r8], xmm1
    vmovss [r9], xmm2
    vzeroupper
    ret

; RDI entrada, RSI salida, EDX N, XMM0 media, XMM1 sigma.
normalize_array:
    test edx, edx
    jle .done
    xor eax, eax
    mov r10d, edx
    and r10d, -8
    vxorps ymm2, ymm2, ymm2
    vucomiss xmm1, xmm2
    je .zero_vector
    vbroadcastss ymm0, xmm0
    vbroadcastss ymm1, xmm1
.vector_loop:
    cmp eax, r10d
    jge .tail_loop
    vmovaps ymm2, [rdi + rax*4]
    vsubps ymm2, ymm2, ymm0
    vdivps ymm2, ymm2, ymm1
    vmovaps [rsi + rax*4], ymm2
    add eax, 8
    jmp .vector_loop
.tail_loop:
    cmp eax, edx
    jge .done
    vmovss xmm2, [rdi + rax*4]
    vsubss xmm2, xmm2, xmm0
    vdivss xmm2, xmm2, xmm1
    vmovss [rsi + rax*4], xmm2
    inc eax
    jmp .tail_loop
.zero_vector:
    cmp eax, r10d
    jge .zero_tail
    vmovaps [rsi + rax*4], ymm2
    add eax, 8
    jmp .zero_vector
.zero_tail:
    cmp eax, edx
    jge .done
    vmovss [rsi + rax*4], xmm2
    inc eax
    jmp .zero_tail
.done:
    vzeroupper
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
