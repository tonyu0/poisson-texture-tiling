#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
// #include <opencv2/cudawarping.hpp>
// #include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
// #ifndef WITH_CUDA
// #define WITH_CUDA
// #endif

void SetupCUDA();
void AffineTransform(cv::Mat &src, cv::Mat &dst, float a, float b, float c, float d, float tx, float ty);
void GenerateSeamlessImage(cv::Mat &src);

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <image_file_name>" << std::endl;
		return 1;
	}

	// start a main process
	std::cout << "========== Start main process ==========" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	const char *filename = argv[1];
	double scale_x = 1.f;
	double scale_y = 1.f;
	// if false, the size will be original width * height, otherwise, the size will be nearest power of two of resulted image.
	// TODO: vertical and horizontal repeat setting from argv
	// int horizontal_repeat = atoi(argv[2]);
	// int vertical_repeat = atoi(argv[3]);

	cv::useOptimized();
#ifdef WITH_CUDA
	SetupCUDA();
#endif

	// Load the image from the specified file path
	cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);

	// Check if the image was loaded successfully
	if (img.empty())
	{
		std::cerr << "Error: Could not read the image file or the file does not exist." << std::endl;
		return -1;
	}

	std::cout << "[Current target] " << "Width: " << img.rows << ", Height: " << img.cols << ", Channel: " << img.channels() << std::endl;
	// 1. scaling
	cv::Mat imgScaled;
	AffineTransform(img, imgScaled, scale_x, 0, 0, scale_y, 0, 0);
	{
		// finish the scaling process
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Scaling target image finished. Elapsed time: " << elapsed << " ms" << std::endl;
		start = end;
	}
	std::cout << "[Current target] " << "Width: " << imgScaled.rows << ", Height: " << imgScaled.cols << ", Channel: " << imgScaled.channels() << std::endl;
	cv::imwrite("img_before_poisson_texture_tiling.png", imgScaled);

	// 2. generating seamless image
	GenerateSeamlessImage(imgScaled);
	{
		// finish the tiling process
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << " Generating seamless image finished. Elapsed time: " << elapsed << " ms" << std::endl;
		start = end;
	}

	// 3. tiling 3x3
	cv::Mat one[3], two[3], three;
	for (int i = 0; i < 3; ++i)
	{
		one[i] = imgScaled;
	}
	for (int i = 0; i < 3; ++i)
	{
		hconcat(one, 3, two[i]);
	}
	vconcat(two, 3, three);
	// cv::namedWindow("Test Result", cv::WINDOW_AUTOSIZE);
	// cv::imshow("Test Result", imgScaled);
	// cv::imshow("Final Result Cropped", imgScaled);
	// cv::imshow("Tiled Result", three);
	// cv::waitKey(0);

	// const cv::Rect roi(0, 0, width, height);
	// imgScaled = finalResult(roi).clone();

	cv::imwrite("img_after_poisson_texture_tiling.png", imgScaled);
	cv::imwrite("tiled_poisson.png", three);
	cv::destroyAllWindows();

	return 0;
}

#ifdef WITH_CUDA
void SetupCUDA()
{
	int device = cv::cuda::getCudaEnabledDeviceCount();
	printf_s("Number of CUDA devices: %d\n", device);
	int getD = cv::cuda::getDevice();
	cv::cuda::setDevice(getD);
}
#endif

void AffineTransform(cv::Mat &src, cv::Mat &dst, float scale_x, float b, float c, float scale_y, float tx, float ty)
{
	int width = src.cols;
	int height = src.rows;
	int channel = src.channels();
	int resized_width = (int)width * scale_x;
	int resized_height = (int)height * scale_y;

	dst = cv::Mat::zeros(resized_height, resized_width, CV_MAKE_TYPE(8U, channel));
	cv::Mat M = (cv::Mat_<float>(2, 3) << scale_x, b, tx, c, scale_y, ty);
#ifdef WITH_CUDA
	cv::cuda::GpuMat gSrc, gDst;
	// upload src and dst matrix
	gSrc.upload(src);
	gDst.upload(dst);
	cv::cuda::warpAffine(gSrc, gDst, M, gDst.size());
	gDst.download(dst);
#else
	cv::warpAffine(src, dst, M, dst.size());
#endif
}

/**
 * @brief Solve Poisson equation using SOR method (improved gauss seidel method)
 * Time complexity: O(k * N) (N: number of pixels, k: number of iterations until convergence)
 * Space complexity: O(N)
 * Comment: if use the fact that a target matrix is sparse, time order will improve to O(k * N)
 */
void PoissonSolver_SOR(cv::Mat &src)
{
	int col = src.cols;
	int row = src.rows;
	cv::Mat dst = src.clone();
	// this is a sparse matrix since this process only see 4-adjacent
	static constexpr double EPS = 1e-5;
	static constexpr double PI = 3.1415926535897932384626;
	static constexpr int iteration = 20000;
	static constexpr int dx[4]{0, -1, 0, 1};
	static constexpr int dy[4]{1, 0, -1, 0};
	// left and right boundary conditions
	for (int y = 0; y < row; ++y)
	{
		dst.at<double>(y, 0) = (src.at<double>(y, 0) + src.at<double>(y, col - 1)) / 2;
		dst.at<double>(y, col - 1) = dst.at<double>(y, 0);
	}
	// top and bottom boundary conditions
	for (int x = 0; x < col; ++x)
	{
		dst.at<double>(0, x) = (src.at<double>(0, x) + src.at<double>(row - 1, x)) / 2;
		dst.at<double>(row - 1, x) = dst.at<double>(0, x);
	}

	double rect = 0.5 * (cos(PI / src.rows) + cos(PI / src.cols));
	double omega = 2. / (1 + sqrt(1 - rect * rect));
	// optimum omega: https://www.sciencedirect.com/science/article/pii/S0893965908001523
	for (int it = 0; it < iteration; ++it)
	{
		bool ok = true;
		for (int y = 1; y < row - 1; ++y)
		{
			for (int x = 1; x < col - 1; ++x)
			{
				double fq = 0.0, vpq = 0.0;
				for (int i = 0; i < 4; ++i)
				{
					int ny = y + dy[i], nx = x + dx[i];
					fq += dst.at<double>(ny, nx);
					vpq += src.at<double>(y, x) - src.at<double>(ny, nx); // guided vector depend on what to do
				}
				double fp = (fq + vpq) / 4.;
				double err = fabs(fp - dst.at<double>(y, x));
				if (err > EPS)
				{
					ok = false;
				}
				dst.at<double>(y, x) = (1. - omega) * dst.at<double>(y, x) + omega * fp;
			}
		}
		if (ok)
		{
			break;
		}
	}
	src = dst.clone();
}

/**
 * @brief Solve Poisson equation using SimplicialLDLT (direct method)
 * Time complexity: O(N^1.5) when decomposing, O(N log N) when solving
 * Space complexity: O(NlogN) (would increase due to fill-in?)
 */
void PoissonSolver_SimplicialLDLT(cv::Mat &src)
{
	int col = src.cols;
	int row = src.rows;
	int n = col * row;
	// solve n consecutive linear equations
	auto A = Eigen::SparseMatrix<double>(n, n);
	auto b = Eigen::VectorXd(n);
	// this is a sparse matrix since this process only see 4-adjacent
	static constexpr int dx[5]{0, -1, 0, 1, 0};
	static constexpr int dy[5]{-1, 0, 0, 0, 1};
	A.reserve(n * 5); // each pixel, 4 directions
	std::vector<double> _b(n);
	for (int y = 0; y < row; ++y)
	{
		for (int x = 0; x < col; ++x)
		{
			int id = y * col + x;
			A.startVec(id);
			double fq = 0.0, vpq = 0.0;
			for (int i = 0; i < 5; ++i)
			{
				int ny = y + dy[i], nx = x + dx[i];
				int nid = ny * col + nx;
				double coef = (id == nid ? 4.0 : -1.0); // diagonal: 4.0
				if (ny == -1 || ny == row)
				{
					// top or bottom boundary condition
					// TODO: improve this condition to get better result.
					vpq += (src.at<double>(0, x) + src.at<double>(row - 1, x)) / 2;
				}
				else if (nx == -1 || nx == col)
				{
					// left or right boundary condition
					// TODO: improve this condition to get better result.
					vpq += (src.at<double>(y, 0) + src.at<double>(y, col - 1)) / 2;
				}
				else
				{
					vpq += src.at<double>(y, x) - src.at<double>(ny, nx);
					A.insertBack(nid, id) = coef;
				}
			}
			_b[id] = vpq;
		}
	}
	A.finalize();
	b = Eigen::Map<Eigen::VectorXd>(&_b[0], _b.size());

	// solve Ax = b
	Eigen::VectorXd x;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
	if (solver.info() != Eigen::Success)
	{
		std::cerr << "decomposition failed" << std::endl;
	}
	x = solver.solve(b);
	if (solver.info() != Eigen::Success)
	{
		std::cerr << "solving failed" << std::endl;
	}

	// copy result to pixel
	for (int i = 0; i < row; ++i)
	{
		double *ptr = src.ptr<double>(i);
		int base = i * col;
		for (int j = 0; j < col; ++j)
		{
			ptr[j] = x[base + j];
		}
	}
}

void GenerateSeamlessImage(cv::Mat &src)
{
	// TODO: separate the build matrix process

	cv::Mat channels[3];
	cv::split(src, channels);
	for (cv::Mat &ch : channels)
	{
		ch.convertTo(ch, CV_64F);
		// PoissonSolver_SOR(ch);
		PoissonSolver_SimplicialLDLT(ch);
		ch.convertTo(ch, CV_8U);
	}
	cv::merge(channels, 3, src);
}