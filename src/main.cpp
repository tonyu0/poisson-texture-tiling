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
	// 1. scaling (TODO)
	cv::Mat imgScaled;
	AffineTransform(img, imgScaled, scale_x, 0, 0, scale_y, 0, 0);
	{
		// finish the scaling process
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		// std::cout << "Scaling target image finished. Elapsed time: " << elapsed << " ms" << std::endl;
		start = end;
	}
	// std::cout << "[Current target] " << "Width: " << imgScaled.rows << ", Height: " << imgScaled.cols << ", Channel: " << imgScaled.channels() << std::endl;

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
	// cv::destroyAllWindows();

	// const cv::Rect roi(0, 0, width, height);
	// imgScaled = finalResult(roi).clone();

	cv::imwrite("poisson_texture_tiling_output.png", imgScaled);
	cv::imwrite("poisson_texture_tiling_output_tiled3x3.png", three);

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
	int w = src.cols;
	int h = src.rows;
	int n = w * h;
	// 1. Compute the gradient-based Laplacian from the input image
	// 2. Construct a sparse Laplacian matrix with periodic boundary conditions
	auto b = Eigen::VectorXd(n);
	b.setZero();
	// this is a sparse matrix since this process only see 4-adjacent
	static constexpr int numDir = 4;
	static constexpr int dx[numDir]{0, -1, 1, 0};
	static constexpr int dy[numDir]{-1, 0, 0, 1};

	std::vector<Eigen::Triplet<double>> _A;
	_A.reserve(n * (numDir + 1)); // each pixel, 4 directions + themselves = number of nun-zero elements

	std::vector<double> _b(n);
	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			int id = y * w + x;
			_A.push_back(Eigen::Triplet<double>(id, id, 4.));

			double fq = 0.;
			double vpq = 0.;
			for (int i = 0; i < numDir; ++i)
			{
				int ny = y + dy[i];
				int nx = x + dx[i];

				if (0 <= ny && ny < h && 0 <= nx && nx < w)
				{
					vpq += src.at<double>(y, x) - src.at<double>(ny, nx);
				}
				else
				{
					// periodic boundary conditions
					// and let the gradient between (y, x) and (ny, nx) be 0
					ny = (ny + h) % h;
					nx = (nx + w) % w;
				}
				int nid = ny * w + nx;
				_A.push_back(Eigen::Triplet<double>(id, nid, -1.));

				_b[id] = vpq;
			}
		}
	}
	// Load all the data into a matrix at once (maybe auto sorting)
	Eigen::SparseMatrix<double> A(n, n);
	A.setFromTriplets(_A.begin(), _A.end());

	// Use before solve() for memory saving
	A.makeCompressed();

	b = Eigen::Map<Eigen::VectorXd>(&_b[0], _b.size());

	// 3. Solve the sparse linear system (Ax = b)
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
	if (solver.info() != Eigen::Success)
	{
		std::cerr << "decomposition failed" << std::endl;
	}
	Eigen::VectorXd x = solver.solve(b);
	if (solver.info() != Eigen::Success)
	{
		std::cerr << "solving failed" << std::endl;
	}

	// 4. Adjust the global color offset
	// Since the above linear system determines only relative pixel differences, not the absolute color level.
	// Just as determining a particular solution from a general solution, it needs to determine the global color.
	// This step restores the global color while preserving the seamless property.

	double mean_diff = 0.;
	for (int i = 0; i < h; ++i)
	{
		double *ptr = src.ptr<double>(i);
		int base = i * w;
		for (int j = 0; j < w; ++j)
		{
			mean_diff += ptr[j] - x[base + j];
			ptr[j] = x[base + j];
		}
	}
	mean_diff /= h * w;
	src += cv::Scalar(mean_diff);
}

void GenerateSeamlessImage(cv::Mat &src)
{
	// TODO: separate the process building matrix

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