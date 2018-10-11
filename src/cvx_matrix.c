#include <stdlib.h>
#include <stdio.h>

#include "cvx_matrix.h"

int cvxmat_alloc(cvx_mat *mat, int rows, int cols)
{
    mat->vals = malloc(rows * cols * sizeof(double));
    if (mat->vals == NULL) {
        fprintf(stderr, "*** ERROR: Out-of-memory rows = %d  cols = %d ***\n", rows, cols);
        return 0;
    }

    mat->rows = rows;
    mat->cols = cols;
    mat->N = rows*cols;

    // Slows us slightly, but much safer
    for (int i = 0; i < mat->N; i++) {
        mat->vals[i] = 0.0;
    }

    return 1;
}

double cvxmat_get(cvx_mat *mat, int row, int col) 
{
    return mat->vals[col + mat->cols * row];
}

void cvxmat_set(cvx_mat *mat, int row, int col, double val) 
{
    mat->vals[col + mat->cols * row] = val;
}

void cvxmat_EWinvert(cvx_mat *mat) 
{
    for (int i = 0; i < mat->N; i++) {
        mat->vals[i] = 1.0/mat->vals[i];
    }
}

void cvxmat_EWmultIP(cvx_mat *mat0, cvx_mat *mat1)
{
    for (int i = 0; i < mat0->N; i++) {
        mat0->vals[i] *= mat1->vals[i];
    }
}

void cvxmat_subractMat(cvx_mat *mat0, cvx_mat *mat1, cvx_mat *mat2)
{
    for (int i = 0; i < mat0->N; i++) {
        mat0->vals[i] = mat1->vals[i] - mat2->vals[i];
    }
}

void cvxmat_multMatMat(cvx_mat *C, cvx_mat *A, cvx_mat *B)
{
    double sum;
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            sum = 0.0;
            for (int k = 0; k < A->cols; k++) {
                //sum += A[i,k] * B[k,j]
                sum += A->vals[k + A->cols * i] * B->vals[j + B->cols * k];
            }
            C->vals[j + C->cols * i] = sum;
        }
    }
}

void cvxmat_multAtA(cvx_mat *C, cvx_mat *A)
{
    double sum;
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            sum = 0.0;
            for (int k = 0; k < A->cols; k++) {
                //sum += A[i,k] * B[k,j]
                sum += A->vals[i + A->cols * k] * A->vals[j + A->cols * k];
            }
            C->vals[j + C->cols * i] = sum;
        }
    }
}



void cvxmat_multAx(cvx_mat *b, cvx_mat *A, cvx_mat *x)
{
    double sum;
    for (int i = 0; i < A->rows; i++) {
        sum = 0.0;
        for (int j = 0; j < A->cols; j++) {
            sum += A->vals[A->cols * i + j] * x->vals[j];
        }
        b->vals[i] = sum;
    }
}

void cvxmat_multAx2(cvx_mat *b, cvx_mat *A, cvx_mat *x)
{
    double sum;
    double *Apos = &(A->vals[0]);
    for (int i = 0; i < A->rows; i++) {
        double *xpos = &(x->vals[0]);
        sum = 0.0;
        for (int j = 0; j < A->cols; j++) {
            sum += (*Apos++) * (*xpos++);
        }
        b->vals[i] = sum;
    }
}


void cvxmat_multAtx(cvx_mat *b, cvx_mat *A, cvx_mat *x)
{
    double sum;
    for (int i = 0; i < A->cols; i++) {
        sum = 0.0;
        for (int j = 0; j < A->rows; j++) {
            sum += A->vals[A->rows * j + i] * x->vals[j];
        }
        b->vals[i] = sum;
    }
}



void cvxmat_subractMatMult1(cvx_mat *mat0, double v0, cvx_mat *mat1, cvx_mat *mat2)
{
    for (int i = 0; i < mat0->N; i++) {
        mat0->vals[i] = v0 * mat1->vals[i] - mat2->vals[i];
    }
}

void cvxmat_updateG(cvx_mat *G, double relax, cvx_mat *xbar)
{
    // G=p*xbar+(1-p)*G;
    for (int i = 0; i < G->N; i++) {
        G->vals[i] = relax * xbar->vals[i] + (1.0-relax) * G->vals[i];
    }
}

void cvxmat_setvals(cvx_mat *mat, double val)
{   
    for (int i = 0; i < mat->N; i++) {
        mat->vals[i] = val;
    }
}

int copyNewMatrix(cvx_mat *mat_in, cvx_mat *mat_out)
{
    cvxmat_alloc(mat_out, mat_in->rows, mat_in->cols);

    int N = mat_out->rows * mat_out->cols;
    for (int i = 0; i < N; i++) {
        mat_out->vals[i] = mat_in->vals[i];
    }

    return 1;
}


/*
int main (void)
{
    printf ("In matrix.c main function\n");
    
    cvx_mat mat;
    allocMatrix(&mat, 100, 100);
    printf ("Matrix rows = %d, cols = %d\n", mat.rows, mat.cols);

    cvx_mat mat2;
    allocMatrix(&mat2, 1000000000, 1000000000);
    printf ("Matrix rows = %d, cols = %d\n", mat2.rows, mat2.cols);

    return 0;
}
*/