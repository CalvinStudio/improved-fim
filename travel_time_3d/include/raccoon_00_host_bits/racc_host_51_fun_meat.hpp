#pragma once
#ifndef RACC_FUN_MEAT
#define RACC_FUN_MEAT
//**********************************Developer****************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//*******************************************************************************
#include "racc_host_50_fun_bones.hpp"
//************************************文件安全相关*********************************
// Check if file is opened successfully.
namespace racc
{
	// Check if filename length out of limit.
	inline void FileNameTooLong(int filenamelenth, int limit)
	{
		/* ********************************************************************
	  ** 函数说明:  检查文件名是否超过程序限制
	  ** 参数说明:
	  ** lenth 文件名字符串长度;
	  ** limit 文件名长度限制;
	  ** ********************************************************************/
		if (filenamelenth > limit)
		{
			printf(" Filename too long, press ENTER to exit program...");
			RACC_ERROR_EXIT;
		}
	}

	inline void EnvironBitCheck()
	{
		printf(" SIZE of types:\n");
		printf(" Size of \"short      \" is __%zd__ in bit\n", 8 * sizeof(short));
		printf(" Size of \"int        \" is __%zd__ in bit\n", 8 * sizeof(int));
		printf(" Size of \"float      \" is __%zd__ in bit\n", 8 * sizeof(float));
		printf(" Size of \"double     \" is __%zd__ in bit\n", 8 * sizeof(double));
		printf(" Size of \"long double\" is __%zd__ in bit\n",
			   8 * sizeof(long double));
		printf("\n LIMIT of types:\n");
#ifdef _WIN32
		printf(" int_16: %d ~ %d\n", INT16_MIN, INT16_MAX);
		printf(" int_32: %d ~ %d\n", INT32_MIN, INT32_MAX);
#endif
	}

	//********************************************************************************************

	inline void Normlz(int n, float *x, float *a)
	{
		int i;
		float s;
		s = 0.0;
		for (i = 0; i < n; i++)
		{
			s = s + x[i] * x[i];
		}
		s = sqrt(s);
		*a = s;
		for (i = 0; i < n; i++)
		{
			x[i] = x[i] / (s);
		}
	}

	inline void MatMcl(float *AB, float **A, int Arow, int Acol, float *B)
	{
		for (int i = 0; i < Arow; i++)
		{
			AB[i] = 0;
			for (int k = 0; k < Acol; k++)
			{
				AB[i] = AB[i] + A[i][k] * B[k];
			}
		}
	}

	inline void avpu(float **A, int Arow, int Acol, float *u, float *v, float alfa,
					 float beta)
	{
		float *AV = (float *)calloc(Arow, sizeof(float));
		if (AV)
		{
			MatMcl(AV, A, Arow, Acol, v);
			for (int i = 0; i < Arow; i++)
				u[i] = AV[i] - alfa * u[i];
			free(AV);
			AV = 0;
		}
	}

	inline void atupv(float **A, int Arow, int Acol, float *u, float *v, float alfa,
					  float beta)
	{
		int i, j;
		float **AT, *Au;
		AT = alloc2d<float>(Acol, Arow);
		Au = (float *)calloc(Acol, sizeof(float));
		if (Au)
		{
			for (i = 0; i < Arow; i++)
			{
				for (j = 0; j < Acol; j++)
				{
					AT[j][i] = A[i][j];
				}
			}
			MatMcl(Au, AT, Acol, Arow, u);
			for (i = 0; i < Acol; i++)
			{
				v[i] = Au[i] - beta * v[i];
			}
			free(Au);
		}
	}

	inline void pstomo(float **A, int Arow, int Acol, float *b, int itmax)
	{
		int i, iter;
		float beta, alfa, *u, *v, *w, *x, *Vel, *ax;
		float phibar, rhobar, phi, rho, c, s, teta, tt;
		u = (float *)calloc(Arow, sizeof(float));
		v = (float *)calloc(Acol, sizeof(float));
		w = (float *)calloc(Acol, sizeof(float));
		x = (float *)calloc(Acol, sizeof(float));
		Vel = (float *)calloc(Acol, sizeof(float));
		ax = (float *)calloc(Arow, sizeof(float));
		tt = 1;
		beta = 1.0;
		alfa = 1.0;
		for (i = 0; i < Arow; i++)
			u[i] = b[i];
		for (i = 0; i < Acol; i++)
		{
			v[i] = 0;
			x[i] = 0;
		}
		Normlz(Arow, u, &beta);
		atupv(A, Arow, Acol, u, v, alfa, beta);
		Normlz(Acol, v, &alfa);
		for (i = 0; i < Acol; i++)
		{
			w[i] = v[i];
		}
		phibar = beta;
		rhobar = alfa;
		printf("LSQR...\n");
		for (iter = 1; iter <= itmax; iter++)
		{
			if (tt > 0.0001)
			{
				tt = 0;
				avpu(A, Arow, Acol, u, v, alfa, beta);
				Normlz(Arow, u, &beta);
				atupv(A, Arow, Acol, u, v, alfa, beta);
				Normlz(Acol, v, &alfa);
				rho = sqrt(rhobar * rhobar + beta * beta);
				c = rhobar / rho;
				s = beta / rho;
				teta = s * alfa;
				rhobar = -c * alfa;
				phi = c * phibar;
				phibar = s * phibar;
				for (i = 0; i < Acol; i++)
				{
					x[i] = x[i] + (phi / rho) * w[i];
					w[i] = v[i] - (teta / rho) * w[i];
				}
				MatMcl(ax, A, Arow, Acol, x);
				for (i = 0; i < Arow; i++)
				{
					tt = tt + (ax[i] - b[i]) * (ax[i] - b[i]);
				}
				cout << "iter= " << iter << " tt=" << tt << endl;
			}
			else
				break;
		}
		cout << "LSQR Finished!" << endl;
		// FILE* fp3;
		// fp3 = fopen("Vel.txt", "w");
		// for (i = 0; i < Arow; i++)
		//{
		//	fprintf(fp3, "%12.5f\Acol", x[i]);
		//}
		// fclose(fp3);
		free(u);
		free(v);
		free(w);
		free(x);
		free(Vel);
		free(ax);
	}

	//*******************************************************************************
	inline bool ModelMeshCompare(MeshFrame a, MeshFrame b)
	{
		if (a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
			a.d_rows == b.d_rows && a.d_cols == b.d_cols &&
			a.l_rows == b.l_rows && a.l_cols == b.l_cols &&
			a.r_rows == b.r_rows && a.r_cols == b.r_cols)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	inline bool ModelGridCompare(GridFrame a, GridFrame b)
	{
		if (a.n_rows == b.n_rows && a.n_cols == b.n_cols && a.n_slices == b.n_slices &&
			a.d_rows == b.d_rows && a.d_cols == b.d_cols && a.d_slices == b.d_slices &&
			a.l_rows == b.l_rows && a.l_cols == b.l_cols && a.l_slices == b.l_slices &&
			a.r_rows == b.r_rows && a.r_cols == b.r_cols && a.r_slices == b.r_slices)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	inline bool FrameCompare(Frame a, Frame b)
	{
		if (a.n_rows == b.n_rows && a.n_cols == b.n_cols && a.n_slices == b.n_slices &&
			a.d_rows == b.d_rows && a.d_cols == b.d_cols && a.d_slices == b.d_slices &&
			a.l_rows == b.l_rows && a.l_cols == b.l_cols && a.l_slices == b.l_slices &&
			a.r_rows == b.r_rows && a.r_cols == b.r_cols && a.r_slices == b.r_slices)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	inline char *str2charx(string a)
	{
		char *b;
		b = racc::new1d<char>(FILENAME_MAX);
#ifdef __linux__
		strcpy(b, a.c_str());
#else
		strcpy_s(b, FILENAME_MAX, a.c_str());
#endif
		return b;
	}

	inline double randu()
	{
		return double(std::rand()) / RAND_MAX;
	}

	inline bool Jacobi(double *matrix, int dim, double *eigenvectors, double *eigenvalues, double precision, int max)
	{
		for (int i = 0; i < dim; i++)
		{
			eigenvectors[i * dim + i] = 1.0f;
			for (int j = 0; j < dim; j++)
			{
				if (i != j)
					eigenvectors[i * dim + j] = 0.0f;
			}
		}

		int nCount = 0; //current iteration
		while (1)
		{
			//find the largest element on the off-diagonal line of the matrix
			double dbMax = matrix[1];
			int nRow = 0;
			int nCol = 1;
			for (int i = 0; i < dim; i++)
			{ //row
				for (int j = 0; j < dim; j++)
				{ //column
					double d = fabs(matrix[i * dim + j]);
					if ((i != j) && (d > dbMax))
					{
						dbMax = d;
						nRow = i;
						nCol = j;
					}
				}
			}

			if (dbMax < precision) //precision check
				break;
			if (nCount > max) //iterations check
				break;
			nCount++;

			double dbApp = matrix[nRow * dim + nRow];
			double dbApq = matrix[nRow * dim + nCol];
			double dbAqq = matrix[nCol * dim + nCol];
			//compute rotate angle
			double dbAngle = 0.5 * atan2(-2 * dbApq, dbAqq - dbApp);
			double dbSinTheta = sin(dbAngle);
			double dbCosTheta = cos(dbAngle);
			double dbSin2Theta = sin(2 * dbAngle);
			double dbCos2Theta = cos(2 * dbAngle);
			matrix[nRow * dim + nRow] = dbApp * dbCosTheta * dbCosTheta +
										dbAqq * dbSinTheta * dbSinTheta + 2 * dbApq * dbCosTheta * dbSinTheta;
			matrix[nCol * dim + nCol] = dbApp * dbSinTheta * dbSinTheta +
										dbAqq * dbCosTheta * dbCosTheta - 2 * dbApq * dbCosTheta * dbSinTheta;
			matrix[nRow * dim + nCol] = 0.5 * (dbAqq - dbApp) * dbSin2Theta + dbApq * dbCos2Theta;
			matrix[nCol * dim + nRow] = matrix[nRow * dim + nCol];

			for (int i = 0; i < dim; i++)
			{
				if ((i != nCol) && (i != nRow))
				{
					int u = i * dim + nRow; //p
					int w = i * dim + nCol; //q
					dbMax = matrix[u];
					matrix[u] = matrix[w] * dbSinTheta + dbMax * dbCosTheta;
					matrix[w] = matrix[w] * dbCosTheta - dbMax * dbSinTheta;
				}
			}

			for (int j = 0; j < dim; j++)
			{
				if ((j != nCol) && (j != nRow))
				{
					int u = nRow * dim + j; //p
					int w = nCol * dim + j; //q
					dbMax = matrix[u];
					matrix[u] = matrix[w] * dbSinTheta + dbMax * dbCosTheta;
					matrix[w] = matrix[w] * dbCosTheta - dbMax * dbSinTheta;
				}
			}

			//compute eigenvector
			for (int i = 0; i < dim; i++)
			{
				int u = i * dim + nRow; //p
				int w = i * dim + nCol; //q
				dbMax = eigenvectors[u];
				eigenvectors[u] = eigenvectors[w] * dbSinTheta + dbMax * dbCosTheta;
				eigenvectors[w] = eigenvectors[w] * dbCosTheta - dbMax * dbSinTheta;
			}
		}

		for (int i = 0; i < dim; i++)
		{
			eigenvalues[i] = matrix[i * dim + i];
		}
		return true;
	}

}
#endif