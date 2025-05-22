#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h> // для sysconf

volatile bool running = true; // Переменная для управления циклом
pthread_t* threads = NULL;
unsigned int num_threads = 0;

// Объявление прототипа функции потока
void* cpu_load(void* arg);

// Функция, создающая потоки для нагрузки CPU
void start_cpu_load() {
    // Получение количества доступных ядер
    #ifdef _SC_NPROCESSORS_ONLN
        num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    #else
        num_threads = 1; // По умолчанию 1 поток
    #endif

    threads = malloc(sizeof(pthread_t) * num_threads);
    if (threads == NULL) {
        perror("Failed to allocate memory for threads");
        return;
    }

    for (unsigned int i = 0; i < num_threads; ++i) {
        if (pthread_create(&threads[i], NULL, cpu_load, NULL) != 0) {
            perror("Failed to create thread");
            free(threads);
            threads = NULL;
            return;
        }
    }
}

// Функция для остановки потоков и освобождения ресурсов
void stop_cpu_load() {
    if (threads == NULL) return;

    running = false;

    for (unsigned int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    threads = NULL;
}

// Реализация функции потока
void* cpu_load(void* arg) {
    while (running) {
        // Можно оставить пустой цикл или выполнять какую-то работу
        // Например, небольшая задержка:
        // usleep(1000); // задержка 1 мс
    }
    return NULL;
}