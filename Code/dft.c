#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

void dft(float* x, int N) {
	float* xr = (float*)malloc(N * sizeof(float));
	float* xi = (float*)malloc(N * sizeof(float));
	int k, n;
	float theta;

	for (k = 0; k < N; k++) {
		xr[k] = 0;
		xi[k] = 0;
		for (n = 0; n < N; n++) {
			theta = (2 * PI * k * n) / N;
			xr[k] = xr[k] + x[n] * cos(theta);
			xi[k] = xi[k] - x[n] * sin(theta);
		}
		printf("%f + j(%f)\n", xr[k], xi[k]);
	}

	free(xr);
	free(xi);
}

// for test only, change for CUDA implementation
int main() {
	int N = 10;
	
	float* x = (float*)malloc(N * sizeof(float));
	
	for(int i = 0; i < N; i++) {
		x[i] = (float)i;
	}

	dft(x, N);
	free(x);
	return 0;
}
