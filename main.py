import ctypes
from libs.load_gpu_lib import *
import threading
from colorama import init, Fore

init()
Text = Fore

def cpu():
    print("Запуск нагрузки на CPU.")
    cpu_lib = ctypes.CDLL("libs/load_cpu_lib.so")

    cpu_lib.cpu_load()
    cpu_lib.start_cpu_load()


def gpu():
    load_gpu()
    start()



def main():
    with open('libs/logo', 'r') as f:
        logo = f.read()
        f.close()
    
    print(Text.RED + logo + Text.RESET)
    print('1 - CPU 2 - GPU')
    mode = input("Enter the mode: ")

    if mode == '1':
        cpu()
    elif mode == '2':
        gpu()
    else:
        print("Error!!!")

if __name__ == '__main__':
    main()