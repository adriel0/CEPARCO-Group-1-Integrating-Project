#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

void idft(float* xr, float* xi) {
	int N = sizeof(xr)/sizeof(xr[0]);
	float* y = (float*)malloc(N * sizeof(float));
	float theta;
	
	for (int n = 0; n < N; n++){
		for (int k = 0; k < N; k++){
			theta = (2 * PI * k * n)/N;
			y[n] = y[n] + xr[k] * cos(theta) + xi[k] * sin(theta);
		}
		y[n] = y[n]/N;
		printf("%f\n", y[n]);
	}

	free(y);
}

int main() {
	int N = 10;
	float xr[10], xi[10];

	for (int i = 0; i < N; i++){
		xr[i] = (float)i;
		xi[i] = (float)(i*2);
	}	

	idft(&xr, &xi);

	return 0;
}
