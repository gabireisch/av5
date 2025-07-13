import numpy as np
import matplotlib.pyplot as plt

class Atomo:
    def __init__(self, massa: float):
        self.massa = massa

class CadeiaAtomica:
    def __init__(self, N: int, tipo: str = 'homogenea', k: float = 1.0):
        self.N = N
        self.k = k
        self.tipo = tipo.lower()
        self.atom_list = self._criar_cadeia()
        self.massas = [atomo.massa for atomo in self.atom_list]

    def _criar_cadeia(self):
        atomos = []
        m = 1.0
        if self.tipo == 'homogenea':
            for _ in range(self.N):
                atomos.append(Atomo(m))
        elif self.tipo == 'ternaria':
            pattern = [m, 3*m, 5*m]
            for i in range(self.N):
                atomos.append(Atomo(pattern[i % 3]))
        else:
            raise ValueError("Tipo de cadeia inválido. Use 'homogenea' ou 'ternaria'.")
        return atomos

    def montar_matrizes(self):
        M = np.diag(self.massas)
        K = np.zeros((self.N, self.N))
        for i in range(self.N):
            K[i, i] += 2 * self.k
            K[i, (i - 1) % self.N] -= self.k
            K[i, (i + 1) % self.N] -= self.k
        return M, K

    def calcular_frequencias_e_modos(self):
        M, K = self.montar_matrizes()
        A = np.linalg.inv(M) @ K
        eigvals, eigvecs = np.linalg.eig(A)

        idx = np.argsort(np.real(eigvals))
        eigvals = np.real(eigvals[idx])
        eigvecs = np.real(eigvecs[:, idx])
        freq = np.sqrt(np.abs(eigvals))

        return freq, eigvecs

    def plotar_frequencias(self, freq):
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(freq)+1), freq, marker='o')
        plt.title(f"Frequências Naturais - {self.tipo.capitalize()} (N={self.N})")
        plt.xlabel("Modo")
        plt.ylabel("Frequência (rad/s)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plotar_modos_normais(self, modos, freq, n_modos=3):
        plt.figure(figsize=(10, 6))
        for i in range(n_modos):
            desloc = modos[:, i]
            desloc = desloc / np.max(np.abs(desloc))  # Normalizar
            plt.subplot(n_modos, 1, i+1)
            plt.plot(range(self.N), desloc, marker='o')
            plt.title(f"Modo {i+1} - Freq: {freq[i]:.2f} rad/s")
            plt.xlabel("Índice do Átomo")
            plt.ylabel("Deslocamento")
            plt.grid(True)
        plt.tight_layout()
        plt.show()
