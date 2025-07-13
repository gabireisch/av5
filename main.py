# Exemplo de uso
from frequencia import CadeiaAtomica


Ns = [100, 1000, 10000]
tipos = ['homogenea', 'ternaria']
k = 1.0

for N in Ns:
    for tipo in tipos:
        print(f"\n===== N = {N}, Tipo = {tipo} =====")
        sistema = CadeiaAtomica(N=N, tipo=tipo, k=k)
        freq, modos = sistema.calcular_frequencias_e_modos()
        print("Primeiras 5 frequências:", freq[:5])

        if N == 100:
            sistema.plotar_frequencias(freq)
            sistema.plotar_modos_normais(modos, freq, n_modos=3)
