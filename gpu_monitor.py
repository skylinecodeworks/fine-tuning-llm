import torch
import matplotlib.pyplot as plt
import time

# Comprobar si hay GPU disponible
if not torch.cuda.is_available():
    print("No GPU detectada por PyTorch.")
    exit()

device = torch.device("cuda")
device_name = torch.cuda.get_device_name(device)

# Inicializar gráficos
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))

mem_used_vals = []
mem_reserved_vals = []
time_vals = []

line1, = ax.plot([], [], label='Memoria usada (MB)', color='red')
line2, = ax.plot([], [], label='Memoria reservada (MB)', color='blue')

ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Memoria (MB)")
ax.set_title(f"Uso de GPU - {device_name}")
ax.legend()
ax.grid(True)

start_time = time.time()

try:
    while True:
        current_time = time.time() - start_time
        mem_used = torch.cuda.memory_allocated(device) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**2

        time_vals.append(current_time)
        mem_used_vals.append(mem_used)
        mem_reserved_vals.append(mem_reserved)

        line1.set_xdata(time_vals)
        line1.set_ydata(mem_used_vals)

        line2.set_xdata(time_vals)
        line2.set_ydata(mem_reserved_vals)

        ax.relim()
        ax.autoscale_view()

        plt.draw()
        plt.pause(1)  # Esperar 1 segundo
except KeyboardInterrupt:
    print("\nMonitor detenido por el usuario.")

