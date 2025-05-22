import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule



# CUDA ядро, которое выполняет простую операцию (например, сложение)
kernel_code = """
__global__ void load_kernel(float *a, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float val = a[idx];
        // Выполняем бесконечный цикл для нагрузки
        while (true) {
            val = val * 1.000001f + 0.000001f;
            if (val > 1e10f) val = 0.0f; // чтобы не выйти за границы
        }
    }
}
"""

def load_gpu():
    # Размер данных
    N = 1024 * 1024  # 1 миллион элементов

    # Создаем массив на CPU
    host_array = np.ones(N, dtype=np.float32)

    # Выделяем память на GPU и копируем данные
    a_gpu = cuda.mem_alloc(host_array.nbytes)
    cuda.memcpy_htod(a_gpu, host_array)

    # Компилируем ядро
    mod = SourceModule(kernel_code)
    load_kernel = mod.get_function("load_kernel")

    # Запускаем ядро с большим количеством потоков для нагрузки
    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    print("Запуск нагрузки на GPU.")
    
    
    # Запускаем ядро в отдельном потоке или просто вызываем его один раз,
    # но внутри ядра бесконечный цикл, поэтому оно будет работать долго.
    load_kernel(a_gpu, np.int32(N), block=(block_size,1,1), grid=(grid_size,1))
    
    # В этом случае ядро работает бесконечно,
    # чтобы остановить его после времени, нужно использовать другие подходы,
    # например, запуск через subprocess или использовать CUDA streams.
    
def start():
    try:
        load_gpu()  # нагрузка
        print("Нагрузка завершена.")
    except KeyboardInterrupt:
        print("Прервано пользователем.")