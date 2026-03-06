import tkinter as tk    #libreria para crear ventanas, botones, campos de texto es la interfaz grafica                                        
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt #libreria para dibujar los graficos de los datos en el eje X y Y
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import csv  #libreria para la lectura de archivos csv
import math     # se utilzia para calculos matematicos
from collections import Counter # Libreria para contar elementos repetidos, para los Knn
import numpy as np # Libreria para calculos matematicos y matrices

# ==================== ALGORITMOS ====================

def linear_regression(X, Y):    # funccion para calcular la recta de regresion
    n = len(X)                  # calcula cuantos datos hay en el archivo csv
    x_mean = sum(X) / n         #calcula la media de X
    y_mean = sum(Y) / n         #calcula la media de Y
    num = sum((X[i] - x_mean) * (Y[i] - y_mean) for i in range(n)) #Calcula el numerador de la fórmula de la pendiente.
    den = sum((X[i] - x_mean) ** 2 for i in range(n))               #den = sum((X[i] - x_mean) ** 2 for i in range(n))
    m = num / den                                                   #Calcula la pendiente de la recta.
    b = y_mean - m * x_mean                                         #Calcula la intersección con el eje Y.
    return m, b                                                     #Devuelve la pendiente y el intercepto m y b

def mse(X, Y, m, b): #Funcion para calcular el error cuadratico
    return sum((Y[i] - (m * X[i] + b)) ** 2 for i in range(len(X))) / len(X)    #Calcula el error entre: valor real Y y valor preducho mX +b

def predict_linear(x, m, b):    #Funcion para hacer una prediccion
    return m * x + b            #ecuacion de la forma pendiente-intersección de la ecuación de una recta.

def euclidean_distance(p1, p2): #Funcion para calcular la distancia entre 2 puntos
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2))) #fórmula de la distancia entre dos puntos

def knn_predict(X_train, Y_train, x_new, k):    #Funcion principal del algoritmoo KNN
    distances = [(euclidean_distance(x_new, X_train[i]), Y_train[i]) for i in range(len(X_train))] #Calcula la distancia del nuevo punto a todos los puntos del dataset.
    distances.sort(key=lambda x: x[0]) #Ordena por distancia menor a mayor.
    k_nearest = distances[:k]   #Selecciona los K vecinos más cercanos.
    labels = [label for _, label in k_nearest]  #Obtiene solo las clases de esos vecinos.
    return Counter(labels).most_common(1)[0][0] #Devuelve la clase que más se repite.

# Inicio de la Interfaz GUI

class AIApp: #Clase que define toda la aplicacion
    def __init__(self, master):
        self.master = master
        master.title("Modelos IA con CSV Automático") #titulo de la ventana
        master.geometry("1100x750") #tamaño de la ventana

        self.X = [] #datos x para la regresion lineal
        self.Y = [] #Datos Y para la regresion lineal
        self.X_knn = []     #datos x para KNN
        self.Y_knn = []     #datos Y para KNN
        self.knn_dimension = 0 #Número de dimensiones del dataset.

        tabs = ttk.Notebook(master) #Crea el diseño de las pestañas
        self.tab_lr = ttk.Frame(tabs) #crea la pestaña de regresion lineal
        self.tab_knn = ttk.Frame(tabs) #crea la pestaña de KNN
        tabs.add(self.tab_lr, text="Regresión Lineal") #asigna nombres a cada pestaña
        tabs.add(self.tab_knn, text="KNN") #asigan nombres a la pestaña
        tabs.pack(expand=1, fill="both") #hace que ocupen toda la ventana se va a visualizar con una linea resta

#Interfaz grafica Regresion Lineal
        ttk.Button(self.tab_lr, text="Cargar CSV", command=self.load_csv_lr).pack(pady=5) #Crea el boton para cargar CSV

        self.entry_x = ttk.Entry(self.tab_lr) #Campo para ingresar X
        self.entry_x.pack()
        self.entry_x.insert(0, "Valor X")

        ttk.Button(self.tab_lr, text="Predecir", command=self.run_lr).pack(pady=5) #Crea el botón de predicción

        self.label_lr = ttk.Label(self.tab_lr) #Etiqueta para mostrar resultados
        self.label_lr.pack()

        self.fig_lr, self.ax_lr = plt.subplots(figsize=(5, 4)) #Crea el grafico X y Y con el tamaño 5,4
        self.canvas_lr = FigureCanvasTkAgg(self.fig_lr, master=self.tab_lr) #usa canvas para mostrar el grafico en la pantalla al usar la libreria tkinter
        self.canvas_lr.get_tk_widget().pack()

#Interfaz grafica de KNN 
        ttk.Button(self.tab_knn, text="Cargar CSV", command=self.load_csv_knn).pack(pady=5)#Crea el boton para cargar CSV

        ttk.Label(self.tab_knn, text="Valor K:").pack()
        self.entry_k = ttk.Entry(self.tab_knn, width=5)
        self.entry_k.pack()
        self.entry_k.insert(0, "3")

        self.frame_entries = ttk.Frame(self.tab_knn)
        self.frame_entries.pack(pady=5)

        ttk.Button(self.tab_knn, text="Predecir", command=self.run_knn).pack(pady=5)

        self.label_knn = ttk.Label(self.tab_knn, text="", font=("Arial", 12))
        self.label_knn.pack(pady=5)

        self.fig_knn, self.ax_knn = plt.subplots(figsize=(6, 5))
        self.canvas_knn = FigureCanvasTkAgg(self.fig_knn, master=self.tab_knn)
        self.canvas_knn.get_tk_widget().pack()

#Funcion para detectar si el archivo CSV usa ; o , para la separacion

    def detect_csv_delimiter(self, path): 
        with open(path, 'r') as f:
            sample = f.read(1024)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=";,") #uso de csv.Sniffer para detectar automaticamente
                return dialect.delimiter
            except:
                return ','  # default

#Funcion para abrir el archivo CSV de Regresion

    def load_csv_lr(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")]) #Abre explorador de archivos.
        if not path:
            return
        delimiter = self.detect_csv_delimiter(path) #Detecta el separador si es ; o ,
        self.X, self.Y = [], []
        with open(path) as f:
            reader = csv.reader(f, delimiter=delimiter)
            next(reader, None)  # saltar encabezado si existe
            for row in reader:
                try:
                    x, y = float(row[0]), float(row[1]) #lee cada FIla y guarda los datos 
                    self.X.append(x)
                    self.Y.append(y)
                except:
                    pass
        messagebox.showinfo("OK", f"Datos cargados. {len(self.X)} puntos.")

#Funcion para abrir el archivo CSV de KNN

    def load_csv_knn(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")]) #Abre explorador de archivos.
        if not path:
            return
        delimiter = self.detect_csv_delimiter(path) #Detecta el separador si es ; o ,
        self.X_knn, self.Y_knn = [], []
        with open(path) as f:
            reader = csv.reader(f, delimiter=delimiter)
            next(reader, None)  # saltar encabezado si lo tiene
            first_data_row = next(reader, None)
            if first_data_row is None:
                messagebox.showerror("Error", "Archivo CSV vacío o sin datos")
                return
            self.knn_dimension = len(first_data_row) - 1 #Detecta cuántas dimensiones tiene el dataset
            self.create_dynamic_entries(self.knn_dimension)
            try:
                features = [float(x) for x in first_data_row[:-1]]
                label = first_data_row[-1]
                self.X_knn.append(features)
                self.Y_knn.append(label)
            except:
                pass
            for row in reader:
                try:
                    features = [float(x) for x in row[:-1]] #lee cada FIla y guarda los datos 
                    label = row[-1]
                    self.X_knn.append(features)
                    self.Y_knn.append(label)
                except:
                    pass
        messagebox.showinfo("OK", f"Datos KNN cargados con dimensión {self.knn_dimension}")

    def create_dynamic_entries(self, dim): #Funicion para crear campos segun la dimension si se carga un archivo de mas de 2 dimensiones no se graficara
        for widget in self.frame_entries.winfo_children():
            widget.destroy()
        self.entries = []
        ttk.Label(self.frame_entries, text="Ingrese punto a predecir (cada dimensión):").pack()
        for i in range(dim):
            frame = ttk.Frame(self.frame_entries)
            frame.pack(pady=2, anchor='w')
            ttk.Label(frame, text=f"Dim {i + 1}: ").pack(side='left') #Asigna el numero de dimensiones que detecto en el CSV
            entry = ttk.Entry(frame, width=10)
            entry.pack(side='left')
            self.entries.append(entry)

 #Funcion para ejecutar la Regresion lineal

    def run_lr(self):
        if not self.X:
            messagebox.showwarning("Error", "Carga datos primero")
            return
        try:
            x_val = float(self.entry_x.get())
        except:
            messagebox.showerror("Error", "Ingrese un número válido")
            return
        m, b = linear_regression(self.X, self.Y)    # calcula la Regresion Lineal
        error = mse(self.X, self.Y, m, b)           #calcula el error MSE
        y_pred = predict_linear(x_val, m, b)        #hace la prediccion
        self.label_lr.config(text=f"y = {m:.2f}x + {b:.2f}\nMSE = {error:.3f}\nPredicción = {y_pred:.2f}") #muesta el resultado en pantalla
        self.ax_lr.clear()
        self.ax_lr.scatter(self.X, self.Y, color="blue") #dibuj alos puntos de color azul
        x_line = np.linspace(min(self.X), max(self.X), 100)
        self.ax_lr.plot(x_line, m * x_line + b, color="red") #diu¿buja la linea de color rojo
        self.ax_lr.grid(True)
        self.canvas_lr.draw()
        
#Funcion para ejecutar KNN
    def run_knn(self):
        if not self.X_knn:
            messagebox.showwarning("Error", "Carga datos primero")
            return
        try:
            k = int(self.entry_k.get())
            if k <= 0 or k > len(self.X_knn):
                messagebox.showerror("Error", "K inválido") #comprueba que se asigen un Valor en K 
                return
        except:
            messagebox.showerror("Error", "K debe ser entero") 
            return
        try:
            point = [float(entry.get()) for entry in self.entries]
        except:
            messagebox.showerror("Error", "Ingrese valores numéricos válidos en todas las dimensiones")
            return
        if len(point) != self.knn_dimension:
            messagebox.showerror("Error", f"Debe ingresar {self.knn_dimension} valores")
            return
        prediction = knn_predict(self.X_knn, self.Y_knn, point, k) #Predice la clase
        self.label_knn.config(text=f"Clase asignada: {prediction}") #Muestra el Resultado
        if self.knn_dimension != 2: #Si la dimension es 2D la dibuja, entonces si se sube mas dimensiones no la dibuja pero si muestra los datos
            self.ax_knn.clear()
            self.ax_knn.text(0.5, 0.5, "Visualización solo disponible para datos 2D",
                             ha='center', va='center', fontsize=14)
            self.canvas_knn.draw()
            return
        self.ax_knn.clear()
        X_array = np.array(self.X_knn)
        classes = list(set(self.Y_knn))
        class_to_num = {c: i for i, c in enumerate(classes)}
        y_numeric = np.array([class_to_num[c] for c in self.Y_knn])
        x_min, x_max = X_array[:, 0].min() - 1, X_array[:, 0].max() + 1
        y_min, y_max = X_array[:, 1].min() - 1, X_array[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),    #usa np.meshgrid para generar el plano de prediccion
                             np.linspace(y_min, y_max, 200))
        Z = []
        for xv, yv in zip(xx.ravel(), yy.ravel()):
            pred = knn_predict(self.X_knn, self.Y_knn, [xv, yv], k)
            Z.append(class_to_num[pred])
        Z = np.array(Z).reshape(xx.shape)
        cmap_light = ListedColormap(['#FFBBBB', '#BBBBFF', '#BBFFBB', '#FFFFBB']) #cmap_light para colores de fondo
        cmap_bold = ListedColormap(['red', 'blue', 'green', 'yellow']) #cmap_bold para colores de puntos
        self.ax_knn.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light) 
        self.ax_knn.scatter(X_array[:, 0], X_array[:, 1],
                            c=y_numeric,
                            cmap=cmap_bold,
                            edgecolor='black',
                            s=80)
        self.ax_knn.scatter(point[0], point[1],
                            color='black',
                            marker='X',
                            s=150)
        self.ax_knn.set_xlim(x_min, x_max)
        self.ax_knn.set_ylim(y_min, y_max)
        self.ax_knn.grid(True)
        self.canvas_knn.draw()

#Inicio de la aplicaion Grafica
root = tk.Tk()  #crea la ventana principal
app = AIApp(root) #Inicialos la Aplicaion con las Funciones crradas anteriormente
root.mainloop() #mantiene la ventana abierta sin cerrarse
