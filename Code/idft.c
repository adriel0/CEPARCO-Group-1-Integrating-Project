#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

void idft(float* xr, float* xi) {
	int N = sizeof(xr)/sizeof(xr[0]);
	float* x = (float*)malloc(N * sizeof(float));
	
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
