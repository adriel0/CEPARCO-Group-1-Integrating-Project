#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846

void dft(int len) {
	float* x = (float*)malloc(len * sizeof(float));
	float* xr = (float*)malloc(len * sizeof(float));
	float* xi = (float*)malloc(len * sizeof(float));
	int k, n, N = len;
	float theta;

	for (int i = 0; i < len; i++) {
		printf("Enter value [%d]: ", i);
		scanf("%f", x[i]);
	}

	for (k = 0; k < N; k++) {
		xr[k] = 0;
		xi[k] = 0;
		for (n = 0; n < N; n++) {
			theta = (2 * PI * k * n) / N;
			xr[k] = xr[k] + x[n] * cos(theta);
			xi[k] = xi[k] - x[n] * sin(theta);
		}
	}

	for (int i = 0; i < len; i++) {

	}

}

// for test only, change for CUDA implementation
int main() {
	int signal_len;

	printf("Enter number of elements: ");
	scanf("%d", &signal_len);

	dft(signal_len);
	return 0;
}
