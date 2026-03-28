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
	float scale_x = 1.f;
	float scale_y = 1.f;

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
	static constexpr float EPS = 1e-5;
	static constexpr float PI = 3.1415926535897932384626;
	static constexpr int iteration = 20000;
	static constexpr int dx[4]{0, -1, 0, 1};
	static constexpr int dy[4]{1, 0, -1, 0};
	// left and right boundary conditions
	for (int y = 0; y < row; ++y)
	{
		dst.at<float>(y, 0) = (src.at<float>(y, 0) + src.at<float>(y, col - 1)) / 2;
		dst.at<float>(y, col - 1) = dst.at<float>(y, 0);
	}
	// top and bottom boundary conditions
	for (int x = 0; x < col; ++x)
	{
		dst.at<float>(0, x) = (src.at<float>(0, x) + src.at<float>(row - 1, x)) / 2;
		dst.at<float>(row - 1, x) = dst.at<float>(0, x);
	}

	float rect = 0.5 * (cos(PI / src.rows) + cos(PI / src.cols));
	float omega = 2. / (1 + sqrt(1 - rect * rect));
	// optimum omega: https://www.sciencedirect.com/science/article/pii/S0893965908001523
	for (int it = 0; it < iteration; ++it)
	{
		bool ok = true;
		for (int y = 1; y < row - 1; ++y)
		{
			for (int x = 1; x < col - 1; ++x)
			{
				float fq = 0.0, vpq = 0.0;
				for (int i = 0; i < 4; ++i)
				{
					int ny = y + dy[i], nx = x + dx[i];
					fq += dst.at<float>(ny, nx);
					vpq += src.at<float>(y, x) - src.at<float>(ny, nx); // guided vector depend on what to do
				}
				float fp = (fq + vpq) / 4.;
				float err = fabs(fp - dst.at<float>(y, x));
				if (err > EPS)
				{
					ok = false;
				}
				dst.at<float>(y, x) = (1. - omega) * dst.at<float>(y, x) + omega * fp;
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
void PoissonSolver_SimplicialLDLT(Eigen::SparseMatrix<float> &A, cv::Mat &_b, cv::Mat &x)
{
	int h = x.rows;
	int w = x.cols;
	int n = w * h;
	Eigen::Map<Eigen::VectorXf> b(_b.ptr<float>(), _b.total());
	// 3. Solve the sparse linear system (Ax = b)
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver(A);
	if (solver.info() != Eigen::Success)
	{
		std::cerr << "decomposition failed" << std::endl;
	}
	Eigen::VectorXf _x = solver.solve(b);
	if (solver.info() != Eigen::Success)
	{
		std::cerr << "solving failed" << std::endl;
	}

	// 4. Adjust the global color offset
	// Since the above linear system determines only relative pixel differences, not the absolute color level.
	// Just as determining a particular solution from a general solution, it needs to determine the global color.
	// This step restores the global color while preserving the seamless property.

	float mean_diff = 0.;
	for (int i = 0; i < h; ++i)
	{
		float *ptr = x.ptr<float>(i);
		int base = i * w;
		for (int j = 0; j < w; ++j)
		{
			mean_diff += ptr[j] - _x[base + j];
			ptr[j] = _x[base + j];
		}
	}
	mean_diff /= h * w;
	x += cv::Scalar(mean_diff);
}

void PoissonSolver_FFT(cv::Mat &_b, cv::Mat &x)
{
	CV_Assert(x.type() == CV_32F);

	cv::Mat b = _b.reshape(0, x.rows);

	int h = b.rows;
	int w = b.cols;

	// --- FFT ---
	cv::Mat b_complex;
	cv::dft(b, b_complex, cv::DFT_COMPLEX_OUTPUT);

	// --- Solve the Poisson equation in Fourier Space ---
	// Memo:
	// Δx{i,j}​= 4x{i,j} - x{i+1,j} ​- x{i−1,j}​ - x{i,j+1} ​- x{i,j−1}
	// と、Aの構築時には出していた
	// x{i,j}=ei^(kx*​i + ky*​j)
	// これはフーリエ空間上のx{i,j}
	// Δx=(4 - e^(i*kx)​ - e^(−i*kx)​ - e^(i*ky)​ - e^(−i*ky)​) * x
	// これをオイラーの公式でまとめると
	// λ(kx​,ky​)=4 - 2cos(kx​) - 2cos(ky​)

	cv::Mat x_complex = cv::Mat::zeros(h, w, CV_32FC2);

	for (int i = 0; i < h; ++i)
	{
		float ky = 2.0f * CV_PI * i / h;

		for (int j = 0; j < w; ++j)
		{
			float kx = 2.0f * CV_PI * j / w;

			float lambda = 4.f - 2.f * cos(kx) - 2.f * cos(ky);

			cv::Vec2f b_val = b_complex.at<cv::Vec2f>(i, j);

			if (fabs(lambda) > 1e-5f) // Limit the lower lambda to prevent divergence
			{
				x_complex.at<cv::Vec2f>(i, j)[0] = b_val[0] / lambda;
				x_complex.at<cv::Vec2f>(i, j)[1] = b_val[1] / lambda;
			}
		}
	}

	// --- Inverse FFT ---
	cv::Mat x_real;
	cv::dft(x_complex, x_real, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

	// --- Adjust mean value ---
	float mean_diff = 0.;
	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			mean_diff += x.at<float>(i, j) - x_real.at<float>(i, j);
			x.at<float>(i, j) = x_real.at<float>(i, j);
		}
	}
	mean_diff /= h * w;
	x += cv::Scalar(mean_diff);
}

void MakeLaplacian(cv::Mat &src, Eigen::SparseMatrix<float> &A, cv::Mat &b)
{
	int w = src.cols;
	int h = src.rows;
	int n = w * h;
	// 1. Compute the gradient-based Laplacian from the input image
	// 2. Construct a sparse Laplacian matrix with periodic boundary conditions
	// this is a sparse matrix since this process only see 4-adjacent
	static constexpr int numDir = 4;
	static constexpr int dx[numDir]{0, -1, 1, 0};
	static constexpr int dy[numDir]{-1, 0, 0, 1};

	std::vector<Eigen::Triplet<float>> _A;
	_A.reserve(n * (numDir + 1)); // each pixel, 4 directions + themselves = number of nun-zero elements
	b = cv::Mat(1, n, CV_32F);

	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			int id = y * w + x;
			_A.push_back(Eigen::Triplet<float>(id, id, 4.));

			float fq = 0.;
			float vpq = 0.;
			for (int i = 0; i < numDir; ++i)
			{
				int ny = y + dy[i];
				int nx = x + dx[i];

				if (0 <= ny && ny < h && 0 <= nx && nx < w)
				{
					vpq += src.at<float>(y, x) - src.at<float>(ny, nx);
				}
				else
				{
					// periodic boundary conditions
					// and let the gradient between (y, x) and (ny, nx) be 0
					ny = (ny + h) % h;
					nx = (nx + w) % w;
				}
				int nid = ny * w + nx;
				_A.push_back(Eigen::Triplet<float>(id, nid, -1.));

				b.at<float>(0, id) = vpq;
			}
		}
	}
	// Load all the data into a matrix at once (maybe auto sorting)
	A = Eigen::SparseMatrix<float>(n, n);
	A.setFromTriplets(_A.begin(), _A.end());

	// Use before solve() for memory saving
	A.makeCompressed();
}

void GenerateSeamlessImage(cv::Mat &src)
{
	// Solve Ax = b
	Eigen::SparseMatrix<float> A;
	cv::Mat b;

	cv::Mat channels[3];
	cv::split(src, channels);
	for (cv::Mat &ch : channels)
	{
		ch.convertTo(ch, CV_32F);

		MakeLaplacian(ch, A, b);
		// PoissonSolver_SOR(ch);
		// PoissonSolver_SimplicialLDLT(A, b, ch);
		PoissonSolver_FFT(b, ch);
		ch.convertTo(ch, CV_8U);
	}
	cv::merge(channels, 3, src);
}