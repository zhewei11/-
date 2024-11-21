#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 自定義比較函數，用於 qsort 進行局部排序
int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

// 合併排序好的陣列
void merge(int *arr1, int size1, int *arr2, int size2, int *result) {
    int i = 0, j = 0, k = 0;
    while (i < size1 && j < size2) {
        if (arr1[i] < arr2[j]) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }
    while (i < size1) {
        result[k++] = arr1[i++];
    }
    while (j < size2) {
        result[k++] = arr2[j++];
    }
}

int main(int argc, char** argv) {
    int n, rank, size;
    int *data = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // 獲取輸入數據
        scanf("%d", &n);
        data = (int *)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            scanf("%d", &data[i]);
        }
    }

    // 廣播 n 給所有處理器
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 計算每個進程負責的部分
    int local_size = n / size;
    int remainder = n % size;
    if (rank < remainder) {
        local_size++;
    }
    int *local_data = (int *)malloc(local_size * sizeof(int));

    // 計算每個進程的偏移量並將資料分發給各個進程
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = n / size;
        if (i < remainder) {
            sendcounts[i]++;
        }
        displs[i] = sum;
        sum += sendcounts[i];
    }

    MPI_Scatterv(data, sendcounts, displs, MPI_INT, local_data, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // 在每個進程上進行局部排序
    qsort(local_data, local_size, sizeof(int), compare);

    // 逐步合併排序結果
    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            if (rank + step < size) {
                int recv_size;
                MPI_Recv(&recv_size, 1, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int *recv_data = (int *)malloc(recv_size * sizeof(int));
                MPI_Recv(recv_data, recv_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                int *merged_data = (int *)malloc((local_size + recv_size) * sizeof(int));
                merge(local_data, local_size, recv_data, recv_size, merged_data);
                
                free(local_data);
                free(recv_data);
                local_data = merged_data;
                local_size += recv_size;
            }
        } else {
            int near = rank - step;
            MPI_Send(&local_size, 1, MPI_INT, near, 0, MPI_COMM_WORLD);
            MPI_Send(local_data, local_size, MPI_INT, near, 0, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    // 最終結果在 rank 0 上顯示
    if (rank == 0) {
        for (int i = 0; i < local_size; i++) {
            printf("%d ", local_data[i]);
        }
        free(data);
    }

    free(local_data);
    free(sendcounts);
    free(displs);
    MPI_Finalize();
    return 0;
}
