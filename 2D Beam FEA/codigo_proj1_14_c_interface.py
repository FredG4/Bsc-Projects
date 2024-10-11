import math as m
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class Sistema:

    def __init__(self, ficheiro, diretorio, fator):  # definição da classe
        self.coord = []  # l=#pt ; c=x , y
        self.connect = []  # l=elem ; c= p1, p2
        self.front = []  # l=#pt ; c=cond front em x , cond front em y (1==n tá preso, 0==ta preso)
        self.forca = []  # l=#pt ; c=modulo , angulo
        self.e = 0  # módulo de Young
        self.a = 0  # cross area
        self.ea = 0  # módulo de Young * cross area (constante pelo sistema todo)
        self.n_pont = 0  # número de pontos
        self.n_elem = 0  # número de elementos
        self.n_desloc = 0  # número de deslocamentos

        self.fatorg = fator
        self.diretorio = diretorio
        self.ficheiro = ficheiro
        self.ler(ficheiro)

    def ler(self, ficheiro):  # função de leitura
        with open(ficheiro, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            line = line.strip().lower()
            if line == 'coordenadas':
                self.coord = [list(map(float, pair.split(','))) for pair in lines[i + 1].strip().split(';') if pair]

            elif line == 'conectividades':
                self.connect = [list(map(int, pair.split(','))) for pair in lines[i + 1].strip().split(';') if pair]

            elif line == 'condicoes fronteira':
                self.front = [list(map(int, pair.split(','))) for pair in lines[i + 1].strip().split(';') if pair]

            elif line == 'forcas existentes':
                self.forca = [list(map(float, pair.split(','))) for pair in lines[i + 1].strip().split(';') if pair]

            elif line == 'modulo de young (pa)':
                self.e = float(lines[i + 1].strip())

            elif line == 'area cross section (m^2)':
                self.a = float(lines[i + 1].strip())

        self.ea = self.a * self.e
        self.n_pont = len(self.coord)
        self.n_elem = len(self.connect)
        self.n_desloc = 2 * self.n_pont

    def f_dist(self, e):  # comprimento de um elemento
        p1 = self.connect[e][0]
        p2 = self.connect[e][1]
        return m.sqrt(((self.coord[p1][0] - self.coord[p2][0]) ** 2) + ((self.coord[p1][1] - self.coord[p2][1]) ** 2))

    def f_ang(self, e):  # ângulo do elemento em relação ao eixo x global
        p1 = self.connect[e][0]
        p2 = self.connect[e][1]
        cat_x = self.coord[p2][0] - self.coord[p1][0]
        cat_y = self.coord[p2][1] - self.coord[p1][1]
        return m.atan2(cat_y, cat_x)

    @staticmethod
    def f_xcomp(mod, a):  # componente x de uma força
        return mod * (m.cos(m.radians(a)))

    @staticmethod
    def f_ycomp(mod, a):  # componente y de uma força
        return mod * (m.sin(m.radians(a)))

    def f_k_elem(self, e):  # matriz k do elemento e
        k_elem = np.zeros((4, 4))
        a = self.f_ang(e)
        h_e = self.f_dist(e)
        coef = (self.ea) / (h_e)
        c = m.cos(a)
        s = m.sin(a)

        k_elem[0][0] = k_elem[2][2] = coef * (c ** 2)
        k_elem[0][2] = k_elem[2][0] = -coef * (c ** 2)
        k_elem[1][1] = k_elem[3][3] = coef * (s ** 2)
        k_elem[1][3] = k_elem[3][1] = -coef * (s ** 2)
        k_elem[0][1] = k_elem[1][0] = k_elem[2][3] = k_elem[3][2] = coef * (c * s)
        k_elem[0][3] = k_elem[3][0] = k_elem[1][2] = k_elem[2][1] = -coef * (c * s)

        return k_elem

    def f_k_tot(self):  # matriz k completa
        k_tot = np.zeros(((self.n_desloc), (self.n_desloc)))
        for e in range(self.n_elem):
            p1 = self.connect[e][0]
            p2 = self.connect[e][1]
            et = [2 * p1, (2 * p1) + 1, 2 * p2, (2 * p2) + 1]
            k_e = self.f_k_elem(e)
            for i in range(4):
                for j in range(4):
                    c = et[i]
                    l = et[j]
                    k_tot[c][l] += k_e[i][j]

        return k_tot

    def f_correcao(self, k_tot):  # matriz correspondente ao sistema de equações
        zeros = np.zeros((self.n_desloc, 1))
        k_tot_ala = np.hstack((k_tot, zeros))

        for p in range(self.n_pont):
            mod = self.forca[p][0]
            a = self.forca[p][1]
            cfx = self.front[p][0]
            cfy = self.front[p][1]

            # forças na coluna final (matriz alargada)
            k_tot_ala[2 * p][self.n_desloc] = cfx * self.f_xcomp(mod, a)
            k_tot_ala[(2 * p) + 1][self.n_desloc] = cfy * self.f_ycomp(mod, a)

            # correção dos casos em que u=0
            if cfx == 0:
                for g in range(self.n_desloc):
                    if g == 2 * p:
                        k_tot_ala[2 * p][g] = 1
                    else:
                        k_tot_ala[2 * p][g] = 0

            if cfy == 0:
                for h in range(self.n_desloc):
                    if h == (2 * p) + 1:
                        k_tot_ala[(2 * p) + 1][h] = 1
                    else:
                        k_tot_ala[(2 * p) + 1][h] = 0

        return k_tot_ala

    def f_resolver(self, k_tot_ala):  # resolver a matriz alargada
        # eliminação de Gauss para triangularização
        for i in range(self.n_desloc):
            if k_tot_ala[i][i] == 0:
                raise ValueError(f"Divisão por zero detectada na linha {i}.")
            k_tot_ala[i] = k_tot_ala[i] / k_tot_ala[i][i]

            for j in range(i + 1, self.n_desloc):
                fator = k_tot_ala[j][i] / k_tot_ala[i][i]
                k_tot_ala[j] = k_tot_ala[j] - fator * k_tot_ala[i]

        # substituição retroativa para obter vetor u
        vetor_u = np.zeros(self.n_desloc)
        for i in range(self.n_desloc - 1, -1, -1):
            sumat = np.dot(k_tot_ala[i, i:self.n_desloc], vetor_u[i:self.n_desloc])
            vetor_u[i] = (k_tot_ala[i, -1] - sumat) / k_tot_ala[i, i]

        return vetor_u

    def f_reacoes(self, k_tot, vetor_u):  # vetor reações
        vetor_f = np.dot(k_tot, vetor_u)  # multiplicar k_tot com u
        vetor_reacoes = np.zeros(self.n_desloc)

        for p in range(self.n_pont):  # subtrair as forças
            mod = self.forca[p][0]
            a = self.forca[p][1]
            vetor_reacoes[2 * p] = vetor_f[2 * p] - self.f_xcomp(mod, a)
            vetor_reacoes[(2 * p) + 1] = vetor_f[(2 * p) + 1] - self.f_ycomp(mod, a)

        return vetor_reacoes

    def f_tensoes(self, vetor_u):  # vetor com as tensões
        vetor_tensoes = np.zeros(self.n_elem)

        for e in range(self.n_elem):
            p1 = self.connect[e][0]
            p2 = self.connect[e][1]
            a = self.f_ang(e)
            c = m.cos(a)
            s = m.sin(a)

            # obter os deslocamentos
            u_p1 = (c * vetor_u[2 * p1]) + (s * vetor_u[(2 * p1) + 1])
            u_p2 = (c * vetor_u[2 * p2]) + (s * vetor_u[(2 * p2) + 1])
            L = self.f_dist(e)

            vetor_tensoes[e] = self.e * ((u_p2 - u_p1) / L)

        return vetor_tensoes

    def f_atualizar_coords(self, vetor_u):  # atualizar as coordenadas para a imagem
        coord_atua = np.copy(self.coord)
        for p in range(self.n_pont):  # somar em todas as coordenadas os deslocamentos respectivos
            coord_atua[p][0] += self.fatorg * vetor_u[2 * p]
            coord_atua[p][1] += self.fatorg * vetor_u[(2 * p) + 1]

        return coord_atua

    def f_escala_cor(self, vetor_tensoes):  # escala de cor para a representação
        v_min = np.min(vetor_tensoes)
        v_max = np.max(vetor_tensoes)
        espetro = v_max - v_min
        q1 = v_min + espetro / 9
        q2 = v_min + (2 * espetro) / 9
        q3 = v_min + (3 * espetro) / 9
        q4 = v_min + (4 * espetro) / 9
        q5 = v_min + (5 * espetro) / 9
        q6 = v_min + (6 * espetro) / 9
        q7 = v_min + (7 * espetro) / 9
        q8 = v_min + (8 * espetro) / 9
        escala = sorted(set([v_min, q1, q2, q3, q4, q5, q6, q7, q8, v_max]))

        return escala

    def f_formscie(self, x, pos):
        return f"{x:.4e}".replace("+", "")

    def f_mostrar(self, coord_atua, vetor_tensoes, escala):  # representação gráfica dos deslocamentos
        x_i = np.zeros(self.n_pont)
        y_i = np.zeros(self.n_pont)
        x_d = np.zeros(self.n_pont)
        y_d = np.zeros(self.n_pont)

        # criar pontos originais e deformados
        for p in range(self.n_pont):
            x_i[p] = self.coord[p][0]
            y_i[p] = self.coord[p][1]

            x_d[p] = coord_atua[p][0]
            y_d[p] = coord_atua[p][1]

        plt.figure(figsize=(10, 8))
        plt.scatter(x_i, y_i, color='gray', label='Sistema inicial', alpha=0.5)
        plt.scatter(x_d, y_d, color='black', label='Sistema deformado', alpha=0.5)

        cores = ['blue', 'darkturquoise', 'limegreen', 'greenyellow', 'yellow',
                 'gold', 'orange', 'orangered', 'red']

        for e in range(self.n_elem):
            p1 = self.connect[e][0]
            p2 = self.connect[e][1]
            tensao = vetor_tensoes[e]

            plt.plot([x_i[p1], x_i[p2]], [y_i[p1], y_i[p2]], color='gray', linewidth=2, alpha=0.5)

            cor = next((cores[i] for i in range(len(escala) - 1) if escala[i] <= tensao < escala[i + 1]), cores[-1])
            plt.plot([x_d[p1], x_d[p2]], [y_d[p1], y_d[p2]], color=cor, linewidth=3, alpha=0.8)

        plt.title("Demonstração gráfica dos deslocamentos por um fator de " + str(self.fatorg))
        plt.xlabel('Eixo X (m)')
        plt.ylabel('Eixo Y (m)')
        plt.legend()
        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')

        # Barra de cores
        norm = mpl.colors.BoundaryNorm(boundaries=escala, ncolors=len(cores))
        cmap = mpl.colors.ListedColormap(cores)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                            ax=plt.gca(), orientation='vertical', shrink=0.8, aspect=30, pad=0.1)
        cbar.set_label('Tensão (Pa)', labelpad=20, fontsize=12, rotation=-90)
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.f_formscie))

        plt.show()

    def projeto_1(self):  # função que engloba tudo
        k_tot = self.f_k_tot()
        k_tot_ala = self.f_correcao(k_tot)
        vetor_u = self.f_resolver(k_tot_ala)
        vetor_r_par = np.reshape(self.f_reacoes(k_tot, vetor_u), (-1, 2))
        vetor_t = self.f_tensoes(vetor_u)
        vetor_u_par = np.reshape(vetor_u, (-1, 2))
        e_fs = self.f_formscie(self.e, None)
        a_fs = self.f_formscie(self.a, None)

        diret = os.path.join(self.diretorio, 'Resultados.txt')  # escrever num ficheiro
        with open(diret, 'w') as file:
            file.write("Informações e resolução com base no ficheiro:\n" + str(self.ficheiro) + "\n\n")
            file.write("\nCoordenadas originais [x, y (m)]:\n" + str(self.coord) + "\n")
            file.write("\nConectividades:\n" + str(self.connect) + "\n")
            file.write("\nCondições fronteira {0 significa apoio nessa direção}:\n" + str(self.front) + "\n")
            file.write("\nForças aplicadas [módulo (N), ângulo (graus)]:\n" + str(self.forca) + "\n")
            file.write(f"\nMódulo de Young (Pa): {e_fs}   ;   Área da secção transversal (m^2): {a_fs}\n")
            file.write("\n\nVetor dos deslocamentos [x, y (m)]:\n" + str(vetor_u_par) + "\n")
            file.write("\nVetor das reações [x, y (N)]:\n" + str(vetor_r_par) + "\n")
            file.write("\nVetor das tensões (Pa):\n" + str(vetor_t) + "\n")

        print("\nValores guardados em 'Resultados.txt'")
        self.f_mostrar(self.f_atualizar_coords(vetor_u), vetor_t, self.f_escala_cor(vetor_t))

        return vetor_u, self.f_atualizar_coords(vetor_u), vetor_t


#### interface feita pelo chat gpt para complementar o codigo feito
# o calculo do fator automatico esta igual a da versao sem interface, no entanto esta algures no meio do codigo da interface

def compute_auto_factor(sistema_obj):
    """
    Compute the scaling factor based on the mean ratio between displacements and element lengths.
    """
    try:
        k_tot = sistema_obj.f_k_tot()
        k_tot_ala = sistema_obj.f_correcao(k_tot)
        vetor_u = sistema_obj.f_resolver(k_tot_ala)
        cumpr = np.zeros(sistema_obj.n_elem)
        for e in range (sistema_obj.n_elem):
            cumpr[e] = sistema_obj.f_dist(e)
        cumpr_max = np.max(cumpr)
        u_max = np.max(np.abs(vetor_u))
        return round(cumpr_max / (10*u_max))
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao calcular fator automático: {e}")
        return 1.0

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Análise Estrutural - Interface Gráfica")

        # Variables
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.fator = tk.DoubleVar(value=200.0)  # default value
        self.auto_fator = tk.BooleanVar(value=False)

        # Layout
        self.create_widgets()

    def create_widgets(self):
        padding = {'padx': 10, 'pady': 10}

        # Input File Selection
        input_frame = ttk.Frame(self.root)
        input_frame.pack(fill='x', **padding)

        ttk.Label(input_frame, text="Ficheiro de Entrada:").pack(side='left')
        ttk.Entry(input_frame, textvariable=self.input_file, width=50).pack(side='left', padx=5)
        ttk.Button(input_frame, text="Selecionar", command=self.select_input_file).pack(side='left')

        # Output Directory Selection
        output_frame = ttk.Frame(self.root)
        output_frame.pack(fill='x', **padding)

        ttk.Label(output_frame, text="Diretório de Saída:").pack(side='left')
        ttk.Entry(output_frame, textvariable=self.output_dir, width=50).pack(side='left', padx=5)
        ttk.Button(output_frame, text="Selecionar", command=self.select_output_dir).pack(side='left')

        # Fator Input
        fator_frame = ttk.Frame(self.root)
        fator_frame.pack(fill='x', **padding)

        self.auto_fator_check = ttk.Checkbutton(fator_frame, text="Calcular Fator Automaticamente", variable=self.auto_fator, command=self.toggle_fator_entry)
        self.auto_fator_check.pack(side='left')

        self.fator_entry = ttk.Entry(fator_frame, textvariable=self.fator, width=10)
        self.fator_entry.pack(side='left', padx=5)
        ttk.Label(fator_frame, text="Fator").pack(side='left')

        # Run Button
        run_frame = ttk.Frame(self.root)
        run_frame.pack(fill='x', **padding)

        self.run_button = ttk.Button(run_frame, text="Executar Simulação", command=self.run_simulation)
        self.run_button.pack()

    def select_input_file(self):
        file_path = filedialog.askopenfilename(title="Selecionar Ficheiro de Entrada", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            self.input_file.set(file_path)

    def select_output_dir(self):
        dir_path = filedialog.askdirectory(title="Selecionar Diretório de Saída")
        if dir_path:
            self.output_dir.set(dir_path)

    def toggle_fator_entry(self):
        if self.auto_fator.get():
            self.fator_entry.config(state='disabled')
        else:
            self.fator_entry.config(state='normal')

    def run_simulation(self):
        ficheiro = self.input_file.get()
        diretorio = self.output_dir.get()
        fator = self.fator.get()

        # Validate inputs
        if not ficheiro:
            messagebox.showerror("Erro", "Por favor, selecione um ficheiro de entrada.")
            return
        if not diretorio:
            messagebox.showerror("Erro", "Por favor, selecione um diretório de saída.")
            return
        if not os.path.isfile(ficheiro):
            messagebox.showerror("Erro", "O ficheiro de entrada selecionado não existe.")
            return
        if not os.path.isdir(diretorio):
            messagebox.showerror("Erro", "O diretório de saída selecionado não existe.")
            return

        try:
            # Instantiate Sistema
            sistema_obj = Sistema(ficheiro, diretorio, fator)

            # If automatic fator is selected, compute it
            if self.auto_fator.get():
                fator = compute_auto_factor(sistema_obj)
                sistema_obj.fatorg = fator

            # Run the project
            vetor_u, coord_atua, vetor_t = sistema_obj.projeto_1()

            # Notify user
            messagebox.showinfo("Sucesso", f"Simulação concluída!\nResultados salvos em {os.path.join(diretorio, 'Resultados.txt')}")

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro durante a simulação:\n{e}")

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
