#ifndef RACC_HOST_TIO_MEAT
#define RACC_HOST_TIO_MEAT
#include "racc_host_60_Tio_bones.hpp"
namespace racc
{
	//---------------------  output 2D matrix   ----------------------------
	//    _path: destination file path;
	//    _n_rows: row number;   _n_cols: coloum number;   _n_rows, _n_cols: first index of _data
	//    _format: = 0(default), output data in binary without dimension
	//             = 1, output data in binary with dimension
	//             = -1, output data in ASCII
	template <typename T>
	inline void MatOutPut(string _path, T **_data, uint32_t _n_rows, uint32_t _n_cols, int _row_l, int _col_l, int _format)
	/*  output 2D double matrix [order in file is _n_cols then _n_rows]        */
	{
		if (_format == 0)
		{ // output data in binary without dimension
			FILE *fp_m = fopen(_path.c_str(), "wb");
			fwrite(&_data[_row_l][_col_l], _n_rows * _n_cols * sizeof(double), 1, fp_m);
			fclose(fp_m);
		}
		else if (_format == 1)
		{ // output data in binary with dimension
			FILE *fp_m = fopen(_path.c_str(), "wb");
			fwrite(&_n_rows, sizeof(int), 1, fp_m);
			fwrite(&_n_cols, sizeof(int), 1, fp_m);
			fwrite(&_data[_row_l][_col_l], _n_rows * _n_cols * sizeof(double), 1, fp_m);
			fclose(fp_m);
		}
		else if (_format == -1)
		{ // output data in ASCII
			FILE *fp_m = fopen(_path.c_str(), "wt");
			for (int i = _row_l; i < _n_rows + _row_l; i++)
			{
				for (int j = _col_l; j < _n_cols + _col_l; j++)
					fprintf(fp_m, "%e\t", _data[i][j]);
				fprintf(fp_m, "\n");
			}
			fclose(fp_m);
		}
	}

	//---------------------  input 3D ralloc3d   ------------------------------
	//    _path: destination file path;
	//    _n_rows: row number;   _n_cols: coloum number;   _n_slices: depth
	//    _format: =-1, ASCII format
	//             =0(default), binary format. Order in file is [ _n_slices -> _n_cols -> _n_rows ];
	//             =1, binary format. Order in file is [ _n_slices -> _n_rows -> _n_cols ];
	//             =2, binary format. Order in file is [ _n_cols -> _n_slices -> _n_rows ];
	//             =3, binary format. Order in file is [ _n_cols -> _n_rows -> _n_slices ];
	//             =4, binary format. Order in file is [ _n_rows -> _n_slices -> _n_cols ];
	//             =5, binary format. Order in file is [ _n_rows -> _n_cols -> _n_slices ];
	//	  _isdim: =false(default) 	input data in binary without dimension
	//	          =true 	input data in binary with dimension
	template <typename T>
	T ***InputTensor(string _path, uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, int _format, bool _isdim)
	/*  input 3D float ralloc3d   */
	{
		T ***data = NULL;

		FILE *fp_m;
		if (_format != -1 && _isdim == true)
		{
			fp_m = fopen(_path.c_str(), "rb");
			fread(&_n_rows, sizeof(int), 1, fp_m);
			fread(&_n_cols, sizeof(int), 1, fp_m);
			fread(&_n_slices, sizeof(int), 1, fp_m);
		}
		else if (_format != -1 && _isdim == false)
		{
			fp_m = fopen(_path.c_str(), "rb");
		}
		else
		{
			fp_m = fopen(_path.c_str(), "rt");
		}

		switch (_format)
		{
		case -1:
			data = racc::ralloc3d<T>(0, _n_rows - 1, 0, _n_cols - 1, 0, _n_slices - 1);
			for (int i = 0; i < _n_rows; i++)
			{
				for (int j = 0; j < _n_cols; j++)
				{
					for (int k = 0; k < _n_slices; k++)
						fscanf(fp_m, "%f\n", &data[i][j][k]);
				}
			}
			break;
		case 0:
			data = racc::ralloc3d<T>(0, _n_rows - 1, 0, _n_cols - 1, 0, _n_slices - 1);
			fread(data[0][0], _n_rows * _n_cols * _n_slices * sizeof(T), 1, fp_m);
			break;
		case 1:
			data = racc::ralloc3d<T>(0, _n_cols - 1, 0, _n_rows - 1, 0, _n_slices - 1);
			fread(data[0][0], _n_rows * _n_cols * _n_slices * sizeof(T), 1, fp_m);
			break;
		case 2:
			data = racc::ralloc3d<T>(0, _n_rows - 1, 0, _n_slices - 1, 0, _n_cols - 1);
			fread(data[0][0], _n_rows * _n_cols * _n_slices * sizeof(T), 1, fp_m);
			break;
		case 3:
			data = racc::ralloc3d<T>(0, _n_slices - 1, 0, _n_rows - 1, 0, _n_cols - 1);
			fread(data[0][0], _n_rows * _n_cols * _n_slices * sizeof(T), 1, fp_m);
			break;
		case 4:
			data = racc::ralloc3d<T>(0, _n_cols - 1, 0, _n_slices - 1, 0, _n_rows - 1);
			fread(data[0][0], _n_rows * _n_cols * _n_slices * sizeof(T), 1, fp_m);
			break;
		case 5:
			data = racc::ralloc3d<T>(0, _n_slices - 1, 0, _n_cols - 1, 0, _n_rows - 1);
			fread(data[0][0], _n_rows * _n_cols * _n_slices * sizeof(T), 1, fp_m);
			break;
		}

		fclose(fp_m);
		return data;
	}
	//-----------------------------------------------------------------------------
	template <typename T>
	void OutputTensor(string _path, T ***_data, uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, int _row_l, int _col_l, int _nz1, int _format)
	{
		if (_format == 0)
		{ // output data in binary without dimension
			FILE *fp_m = fopen(_path.c_str(), "wb");
			fwrite(&_data[_row_l][_col_l][_nz1], _n_rows * _n_cols * _n_slices * sizeof(T), 1, fp_m);
			fclose(fp_m);
		}
		else if (_format == 1)
		{ // output data in binary with dimension
			FILE *fp_m = fopen(_path.c_str(), "wb");
			fwrite(&_n_rows, sizeof(int), 1, fp_m);
			fwrite(&_n_cols, sizeof(int), 1, fp_m);
			fwrite(&_n_slices, sizeof(int), 1, fp_m);
			fwrite(&_data[_row_l][_col_l][_nz1], _n_rows * _n_cols * _n_slices * sizeof(T), 1, fp_m);
			fclose(fp_m);
		}
		else if (_format == -1)
		{ // output data in ASCII
			FILE *fp_m = fopen(_path.c_str(), "wt");
			for (int i = _row_l; i < _n_rows + _row_l; i++)
				for (int j = _col_l; j < _n_cols + _col_l; j++)
					for (int k = _nz1; k < _n_slices + _nz1; k++)
						fprintf(fp_m, "%d, %d, %d, %e\n", i, j, k, _data[i][j][k]);
			fclose(fp_m);
		}
	}

	template <typename T>
	void MeshOutPut(string file_path, T **Mat, MeshFrame Mesh)
	{
		_ASSERT_IS_REAL_NUMBER;
		string EndStr = file_path.substr(file_path.length() - 4, file_path.length());
		string EndStr1 = ".grd";
		if (EndStr == EndStr1)
		{
			cout << "save .grd File:" << file_path << endl;
			ofstream fout(file_path);
			if (fout.is_open())
			{
				fout << "DSAA" << endl;
				fout << Mesh.n_rows << " " << Mesh.n_cols << endl;
				fout << Mesh.l_rows << " " << Mesh.r_rows << endl;
				fout << Mesh.l_cols << " " << Mesh.r_cols << endl;
				fout << racc::min(Mat, Mesh.n_rows, Mesh.n_cols) << " " << racc::max(Mat, Mesh.n_rows, Mesh.n_cols) << endl;
				for (int j = 0; j < Mesh.n_cols; j++)
				{
					for (int i = 0; i < Mesh.n_rows; i++)
					{
						fout << Mat[i][j] << " ";
					}
					fout << endl;
				}
				fout.close();
			}
			else
			{
				printf("MeshOutPut():[ERROR]:Can't create GRD output file %s ,exit\n", file_path.c_str());
				RACC_ERROR_EXIT;
			}
		}
		else
		{
			printf("save X-Y-V File:%s\n", file_path.c_str());
			ofstream fout(file_path);
			if (fout.is_open())
			{
				for (int i = 0; i < Mesh.n_rows; i++)
				{
					for (int j = 0; j < Mesh.n_cols; j++)
					{
						fout << Mesh.l_rows + i * Mesh.d_rows << " " << Mesh.l_cols + j * Mesh.d_cols << " " << Mat[i][j] << endl;
					}
				}
				fout.close();
			}
			else
			{
				printf("MeshOutPut():[ERROR]:Can't create XYV output file %s ,exit\n", file_path.c_str());
				RACC_ERROR_EXIT;
			}
		}
	}

	template <typename T>
	void CubeOutPut(string file_path, T ***Cube, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices)
	{
		ofstream fout(file_path);
		if (fout.is_open())
		{
			for (int i = 0; i < n_rows; i++)
			{
				for (int j = 0; j < n_cols; j++)
				{
					for (int k = 0; k < n_slices; k++)
					{
						fout << i << " " << j << " " << k << " " << Cube[i][j][k] << endl;
					}
				}
			}
			cout << "save X-Y-Z-V File [" << file_path << "]" << endl;
		}
		else
		{
			cout << "GridOutPut:[ERROR]:Can't create output file [" << file_path << "]" << endl;
			RACC_ERROR_EXIT;
		}
	}

	template <typename T>
	void CubeOutPut(string file_path, string FileName, T ***Cube, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices)
	{
#ifdef _WIN32
		file_path.append("\\");
#endif
#ifdef __linux__
		file_path.append("/");
#endif
		file_path.append(FileName);
		CubeOutPut(file_path, Cube, n_rows, n_cols, n_slices);
	}
	template <typename T>
	void GridSliceOutPut(string file_path, GridFrame Grid, T ***Cube, string Axis, int Num)
	{
		T **mat;
		MeshFrame mesh;
		if (Axis == "X")
		{
			mat = alloc2d<T>(Grid.n_cols, Grid.n_slices);
			mesh.n_rows = Grid.n_cols;
			mesh.n_cols = Grid.n_slices;
			mesh.d_rows = Grid.d_cols;
			mesh.d_cols = Grid.d_slices;
			mesh.l_rows = Grid.l_cols;
			mesh.l_cols = Grid.l_slices;
			mesh.r_rows = mesh.l_rows + mesh.d_rows * (mesh.n_rows - 1);
			mesh.r_cols = mesh.l_cols + mesh.d_cols * (mesh.n_cols - 1);
			for (int i = 0; i < Grid.n_cols; i++)
			{
				for (int j = 0; j < Grid.n_slices; j++)
				{
					mat[i][j] = Cube[Num][i][j];
				}
			}
			MeshOutPut<T>(file_path, mat, mesh);
		}
		if (Axis == "Y")
		{
			mat = alloc2d<T>(Grid.n_rows, Grid.n_slices);
			mesh.n_rows = Grid.n_rows;
			mesh.n_cols = Grid.n_slices;
			mesh.d_rows = Grid.d_rows;
			mesh.d_cols = Grid.d_slices;
			mesh.l_rows = Grid.l_rows;
			mesh.l_cols = Grid.l_slices;
			mesh.r_rows = mesh.l_rows + mesh.d_rows * (mesh.n_rows - 1);
			mesh.r_cols = mesh.l_cols + mesh.d_cols * (mesh.n_cols - 1);
			for (int i = 0; i < Grid.n_rows; i++)
			{
				for (int j = 0; j < Grid.n_slices; j++)
				{
					mat[i][j] = Cube[i][Num][j];
				}
			}
			MeshOutPut<T>(file_path, mat, mesh);
		}
		if (Axis == "Z")
		{
			mat = alloc2d<T>(Grid.n_rows, Grid.n_cols);
			mesh.n_rows = Grid.n_rows;
			mesh.n_cols = Grid.n_cols;
			mesh.d_rows = Grid.d_rows;
			mesh.d_cols = Grid.d_cols;
			mesh.l_rows = Grid.l_rows;
			mesh.l_cols = Grid.l_cols;
			mesh.r_rows = mesh.l_rows + mesh.d_rows * (mesh.n_rows - 1);
			mesh.r_cols = mesh.l_cols + mesh.d_cols * (mesh.n_cols - 1);
			for (int i = 0; i < Grid.n_rows; i++)
			{
				for (int j = 0; j < Grid.n_cols; j++)
				{
					mat[i][j] = Cube[i][j][Num];
				}
			}
			MeshOutPut<T>(file_path, mat, mesh);
		}
		if (Axis == "DIAG")
		{
			if (Grid.n_rows != Grid.n_cols)
			{
				printf("Unable to output diagonal section!!!");
				return;
			}
			else
			{
				mat = alloc2d<T>(Grid.n_rows, Grid.n_slices);
				mesh.n_rows = Grid.n_rows;
				mesh.n_cols = Grid.n_slices;
				mesh.d_rows = Grid.d_rows * sqrt(2.0);
				mesh.d_cols = Grid.d_slices;
				mesh.l_rows = 0;
				mesh.l_cols = 0;
				mesh.r_rows = mesh.l_rows + mesh.d_rows * (mesh.n_rows - 1);
				mesh.r_cols = mesh.l_cols + mesh.d_cols * (mesh.n_cols - 1);
				for (int i = 0; i < Grid.n_rows; i++)
				{
					for (int j = 0; j < Grid.n_slices; j++)
					{
						mat[i][j] = Cube[i][i][j];
					}
				}
				MeshOutPut<T>(file_path, mat, mesh);
			}
		}
	}

	template <typename T>
	void GridSliceOutPut(string file_path, string FileName, GridFrame Grid, T ***Cube, string Axis, int Num)
	{
#ifdef _WIN32
		file_path.append("\\");
#endif
#ifdef __linux__
		file_path.append("/");
#endif
		file_path.append(FileName);
		GridOutPut(file_path, Cube, Axis, Num);
	}

	template <typename T>
	void GridOutPut(string file_path, T ***Cube, GridFrame grid)
	{
		FILE *fptime;
		const char *FilePathChar = file_path.c_str();
		printf("save X-Y-Z-V File:%s\n", file_path.c_str());
		if (FilePathChar != NULL)
		{
			if ((fptime = fopen(FilePathChar, "w")) == NULL)
			{
				printf("GridOutPut:[ERROR]:Can't create output file %s ,exit\n", FilePathChar);
				RACC_ERROR_EXIT;
			}

			if (typeid(Cube[0][0][0]).name()[0] == 'd' || typeid(Cube[0][0][0]).name()[0] == 'f')
			{
				double x, y, z;
				for (int i = 0; i < grid.n_rows; i++)
				{
					for (int j = 0; j < grid.n_cols; j++)
					{
						for (int k = 0; k < grid.n_slices; k++)
						{
							x = grid.l_rows + i * grid.d_rows;
							y = grid.l_cols + j * grid.d_cols;
							z = grid.l_slices + k * grid.d_slices;
							fprintf(fptime, "%f, %f, %f, %10.8e\n", x, y, z, Cube[i][j][k]);
						}
					}
				}
			}
			else if (typeid(Cube[0][0][0]).name()[0] == 'i')
			{
				double x, y, z;
				for (int i = 0; i < grid.n_rows; i++)
				{
					for (int j = 0; j < grid.n_cols; j++)
					{
						for (int k = 0; k < grid.n_slices; k++)
						{
							x = grid.l_rows + i * grid.d_rows;
							y = grid.l_cols + j * grid.d_cols;
							z = grid.l_slices + k * grid.d_slices;
							fprintf(fptime, "%f %f %f %d\n", x, y, z, Cube[i][j][k]);
						}
					}
				}
			}
		}
		fclose(fptime);
	}

	template <typename T>
	void GridOutPut(string file_path, string FileName, T ***Cube, GridFrame grid)
	{
#ifdef _WIN32
		file_path.append("\\");
#endif
#ifdef __linux__
		file_path.append("/");
#endif
		file_path.append(FileName);
		GridOutPut(file_path, Cube, grid);
	}

	template <typename T>
	void ReadGrdFile(string pathname, MeshFrame &mesh, T **&grddata)
	{
		char cdum[4];
		float vmin, vmax;
		FILE *fp;
		fp = fopen(pathname.data(), "rt");
		if (fp)
		{
			fscanf(fp, "%s", cdum);
			fscanf(fp, "/n");
			fscanf(fp, "%d", &mesh.n_rows);
			fscanf(fp, "%d", &mesh.n_cols);
			fscanf(fp, "/n");
			fscanf(fp, "%lf", &mesh.l_rows);
			fscanf(fp, "%lf", &mesh.r_rows);
			fscanf(fp, "/n");
			fscanf(fp, "%lf", &mesh.l_cols);
			fscanf(fp, "%lf", &mesh.r_cols);
			fscanf(fp, "/n");
			fscanf(fp, "%f", &vmin);
			fscanf(fp, "%f", &vmax);
			fscanf(fp, "/n");
			mesh.d_rows = (mesh.r_rows - mesh.l_rows) / (mesh.n_rows - 1);
			mesh.d_cols = (mesh.r_cols - mesh.l_cols) / (mesh.n_cols - 1);
			grddata = ralloc2d<T>(0, mesh.n_rows - 1, 0, mesh.n_cols - 1);
			for (int j = 0; j < mesh.n_cols; j++)
			{
				for (int i = 0; i < mesh.n_rows; i++)
				{
					fscanf(fp, "%lf", &grddata[i][j]);
				}
				fscanf(fp, "/n");
			}
			fclose(fp);
		}
		else
		{
			printf("ReadGrdFile:[ERROR]:File Open Error!");
			RACC_ERROR_EXIT;
		}
	}

	template <typename T>
	void OutputVector(string _path, T *_data, uint64_t _n_elem, int _n_l, int _format)
	{
		if (_format == 0)
		{ // output data in binary without dimension
			FILE *fp_m = fopen(_path.c_str(), "wb");
			fwrite(&_data[_n_l], _n_elem * sizeof(T), 1, fp_m);
			fclose(fp_m);
		}
		else if (_format == 1)
		{ // output data in binary with dimension
			FILE *fp_m = fopen(_path.c_str(), "wb");
			fwrite(&_n_elem, sizeof(int), 1, fp_m);
			fwrite(&_data[_n_l], _n_elem * sizeof(T), 1, fp_m);
			fclose(fp_m);
		}
		else if (_format == -1)
		{ // output data in ASCII
			FILE *fp_m = fopen(_path.c_str(), "wt");
			for (int i = _n_l; i < _n_elem + _n_l; i++)
			{
				fprintf(fp_m, "%e\n", _data[i]);
			}
			fclose(fp_m);
		}
	}

	template <typename T>
	void vecprint_number(long n_l, long n_r, T *vec)
	{
		cout.precision(10);
		if (n_r - n_l < 20)
		{
			_PLOT_LINE;
			for (int i = n_l; i <= n_r; i++)
			{
				std::cout << "[" << i << "]:\t" << vec[i] << std::endl;
			}
			_PLOT_LINE;
		}
		else if (n_r - n_l >= 20)
		{
			_PLOT_LINE;
			for (int i = n_l; i < n_l + 10; i++)
			{
				std::cout << "[" << i << "]:\t" << vec[i] << std::endl;
			}
			printf("...\n");
			for (int i = n_r - 10; i <= n_r; i++)
			{
				std::cout << "[" << i << "]:\t" << vec[i] << std::endl;
			}
			_PLOT_LINE;
		}
	}

	template <typename T>
	void vecprint_struct(long n_l, long n_r, T *vec)
	{
		if (n_r - n_l < 40)
		{
			_PLOT_LINE;
			for (int i = n_l; i <= n_r; i++)
			{
				cout << "[" << i << "]:\t", vec[i].print();
			}
			_PLOT_LINE;
		}
		else if (n_r - n_l >= 40)
		{
			_PLOT_LINE;
			for (int i = n_l; i < n_l + 20; i++)
			{
				cout << "[" << i << "]:\t", vec[i].print();
			}
			printf("...\n");
			for (int i = n_r - 20; i <= n_r; i++)
			{
				cout << "[" << i << "]:\t", vec[i].print();
			}
			_PLOT_LINE;
		}
	}

	template <typename T>
	void matprint(T **mat, uint32_t n_rows, uint32_t n_cols)
	{
		int setw_num = 11;
		cout.setf(ios::left);
		if (n_rows <= 20 && n_cols <= 10)
		{
			_PLOT_LINE;
			for (int i = 0; i < n_rows; i++)
			{
				for (int j = 0; j < n_cols; j++)
				{
					cout << std::setw(setw_num) << mat[i][j];
				}
				cout << endl;
			}
			_PLOT_LINE;
		}
		else if (n_rows > 20 && n_cols <= 10)
		{
			_PLOT_LINE;
			for (int i = 0; i < 10; i++)
			{
				for (int j = 0; j < n_cols; j++)
				{
					cout << std::setw(setw_num) << mat[i][j];
				}
				cout << endl;
			}
			printf(".........\n");
			for (int i = n_rows - 10; i < n_rows; i++)
			{
				for (int j = 0; j < n_cols; j++)
				{
					cout << std::setw(setw_num) << mat[i][j];
				}
				cout << endl;
			}
			_PLOT_LINE;
		}
		else if (n_rows <= 20 && n_cols > 10)
		{
			_PLOT_LINE;
			for (int i = 0; i < n_rows; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					cout << std::setw(setw_num) << mat[i][j];
				}
				printf("...");
				for (int j = n_cols - 5; j < n_cols; j++)
				{
					cout << std::setw(setw_num) << mat[i][j];
				}
				cout << endl;
			}
			_PLOT_LINE;
		}
		else if (n_rows > 20 && n_cols > 10)
		{
			_PLOT_LINE;
			for (int i = 0; i < 10; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					cout << std::setw(setw_num) << mat[i][j];
				}
				printf("...");
				for (int j = n_cols - 5; j < n_cols; j++)
				{
					cout << std::setw(setw_num) << mat[i][j];
				}
				cout << endl;
			}
			printf(".........\n");
			for (int i = n_rows - 10; i < n_rows; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					cout << std::setw(setw_num) << mat[i][j];
				}
				printf("...");
				for (int j = n_cols - 5; j < n_cols; j++)
				{
					cout << std::setw(setw_num) << mat[i][j];
				}
				cout << endl;
			}
			_PLOT_LINE;
		}
	}
}
#endif