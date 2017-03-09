#include <iostream> // for standard I/O
#include <fstream>
#include <string>   // for strings
#include <vector>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write

using namespace std;
using namespace cv;

enum TYPE_THRESHOLDING {ENERGY_TH, MAX_TH, MEAN_TH, CUMULATION_TH, SSQ_CUMULATION_TH} ;

void fwt97(double* x, int n)
{
	double a;
	int i;

	// Predict 1
	a = -1.586134342;
	for (i = 1; i<n - 2; i += 2) {
		x[i] += a*(x[i - 1] + x[i + 1]);
	}
	x[n - 1] += 2 * a*x[n - 2];

	// Update 1
	a = -0.05298011854;
	for (i = 2; i<n; i += 2) {
		x[i] += a*(x[i - 1] + x[i + 1]);
	}
	x[0] += 2 * a*x[1];

	// Predict 2
	a = 0.8829110762;
	for (i = 1; i<n - 2; i += 2) {
		x[i] += a*(x[i - 1] + x[i + 1]);
	}
	x[n - 1] += 2 * a*x[n - 2];

	// Update 2
	a = 0.4435068522;
	for (i = 2; i<n; i += 2) {
		x[i] += a*(x[i - 1] + x[i + 1]);
	}
	x[0] += 2 * a*x[1];

	// Scale
	a = 1 / 1.149604398;
	for (i = 0; i<n; i++) {
		if (i % 2) x[i] *= a;
		else x[i] /= a;
	}

	// Pack
	/*if (tempbank == 0) tempbank = (double *) malloc(n*sizeof(double));
	for (i = 0; i<n; i++) {
	if (i % 2 == 0) tempbank[i / 2] = x[i];
	else tempbank[n / 2 + i / 2] = x[i];
	}
	for (i = 0; i<n; i++) x[i] = tempbank[i];*/
}

/**
*  iwt97 - Inverse biorthogonal 9/7 wavelet transform
*
*  This is the inverse of fwt97 so that iwt97(fwt97(x,n),n)=x for every signal x of length n.
*
*  See also fwt97.
*/
void iwt97(double* x, int n)
{
	double a;
	int i;

	// Unpack
	/*if (tempbank == 0) tempbank = (double *) malloc(n*sizeof(double));
	for (i = 0; i<n / 2; i++) {
	tempbank[i * 2] = x[i];
	tempbank[i * 2 + 1] = x[i + n / 2];
	}
	for (i = 0; i<n; i++) x[i] = tempbank[i];*/

	// Undo scale
	a = 1.149604398;
	for (i = 0; i<n; i++) {
		if (i % 2) x[i] *= a;
		else x[i] /= a;
	}

	// Undo update 2
	a = -0.4435068522;
	for (i = 2; i<n; i += 2) {
		x[i] += a*(x[i - 1] + x[i + 1]);
	}
	x[0] += 2 * a*x[1];

	// Undo predict 2
	a = -0.8829110762;
	for (i = 1; i<n - 2; i += 2) {
		x[i] += a*(x[i - 1] + x[i + 1]);
	}
	x[n - 1] += 2 * a*x[n - 2];

	// Undo update 1
	a = 0.05298011854;
	for (i = 2; i<n; i += 2) {
		x[i] += a*(x[i - 1] + x[i + 1]);
	}
	x[0] += 2 * a*x[1];

	// Undo predict 1
	a = 1.586134342;
	for (i = 1; i<n - 2; i += 2) {
		x[i] += a*(x[i - 1] + x[i + 1]);
	}
	x[n - 1] += 2 * a*x[n - 2];
}

void wt97_in_place(const vector<double> &c, vector<double> &x, int Ls)
{
	x = c;
	int n = x.size();
	double alpha = -1.586134342, beta = -0.05298011854, gamma = 0.8829110762, delta = 0.4435068522, dzeta = 1 / 1.149604398;
	int i;

	for (size_t k = 0; k < Ls; k++){

		// Predict 1
		int s = (1 << k), s1 = (2 << k);
		for (i = s; i < n; i += s1) {
			if (i + s < n)x[i] += alpha*(x[i - s] + x[i + s]);
			else x[i] += 2. * alpha*(x[i - s] + 0);
		}

		// Update 1
		for (i = s1; i < n; i += s1) {
			if (i + s < n)x[i] += beta*(x[i - s] + x[i + s]);
			else x[i] += 2. * beta * x[i - s];
		}
		x[0] += 2. * beta*x[s];

		// Predict 2
		for (i = s; i < n; i += s1) {
			if (i + s < n)x[i] += gamma*(x[i - s] + x[i + s]);
			else x[i] += 2. * gamma*x[i - s];
		}

		// Update 2
		for (i = s1; i<n; i += s1) {
			if (i + s < n)x[i] += delta*(x[i - s] + x[i + s]);
			else x[i] += 2. * delta*x[i - s];
		}
		x[0] += 2. * delta*x[s];

		// Scale
		for (i = 0; i<n; i += s) {
			if ((i >> k) % 2) x[i] *= dzeta;
			else x[i] /= dzeta;
		}
	}
}

void iwt97_in_place(const vector<double> &c, vector<double> &x, int Ls)
{
	x = c;
	int n = x.size();
	double alpha = 1.586134342, beta = 0.05298011854, gamma = -0.8829110762, delta = -0.4435068522, dzeta = 1.149604398;
	int i;

	for (int k = Ls - 1; k >= 0; k--){

		int s = (1 << k), s1 = (2 << k);
		// undo Scale
		for (i = 0; i<n; i += s) {
			if ((i >> k) % 2) x[i] *= dzeta;
			else x[i] /= dzeta;
		}

		// undo Update 2
		for (i = s1; i<n; i += s1) {
			if (i + s < n) x[i] += delta*(x[i - s] + x[i + s]);
			else x[i] += 2. * delta*(x[i - s]);
		}
		x[0] += 2. * delta*x[s];

		// undo Predict 2
		for (i = s; i<n; i += s1) {
			if (i + s < n)x[i] += gamma*(x[i - s] + x[i + s]);
			else x[i] += 2. * gamma*(x[i - s] + 0);
		}

		// undo Update 1
		for (i = s1; i<n; i += s1) {
			if (i + s < n)x[i] += beta*(x[i - s] + x[i + s]);
			else x[i] += 2. * beta * x[i - s];
		}
		x[0] += 2. *  beta*x[s];

		// undo Predict 1
		for (i = s; i < n; i += s1) {
			if (i + s < n)x[i] += alpha*(x[i - s] + x[i + s]);
			else x[i] += 2. * alpha * x[i - s];
		}
	}
}

// wavelet 1D transform with filters (P, U) coefficients p, u for Ls (levels)
void wt1D(const vector<double> &c, vector<double> &wt, vector<double> p, int ip, vector<double> u, int iu, int Ls)
{
	int n = c.size();
	wt = c;
	for (int j = 0; j < Ls; j++){
		int s = (2 << j), s1 = (1 << j);
		// detail
		for (int i = s1; i < n; i += s){
			double wc = 0;

			for (int l = 0, pos = i - s1 + ip * s; l < p.size(); l++, pos += s){
				if (pos >= 0 && pos < n)
					wc += p[l] * wt[pos];
			}
			wt[i] = 0.5 * (wt[i] - wc);
		}
		// smooth
		for (int i = 0; i < n; i += s){
			for (int l = 0, pos = s1 + i + iu * s; l < u.size(); l++, pos += s){
				if (pos >= 0 && pos < n)
					wt[i] += u[l] * wt[pos];
			}
		}
	}
}

// inverse  wavelet 1D transform
void iwt1D(const vector<double> &c, vector<double> &iwt, vector<double> p, int ip, vector<double> u, int iu, int Ls)
{
	int n = c.size();
	iwt = c;
	for (int j = Ls - 1; j >= 0; j--){
		int s = (2 << j), s1 = (1 << j);
		// smooth
		for (int i = 0; i < n; i += s){
			double wc = 0;
			for (int l = 0, pos = s1 + i + iu * s; l < u.size(); l++, pos += s){
				if (pos >= 0 && pos < n)
					wc += u[l] * iwt[pos];
			}
			iwt[i] -= wc;
		}

		// detail
		for (int i = s1; i < n; i += s){
			double wc = 0;

			for (int l = 0, pos = i - s1 + ip * s; l < p.size(); l++, pos += s){
				if (pos >= 0 && pos < n)
					wc += p[l] * iwt[pos];
			}

			iwt[i] = 2 * iwt[i] + wc;
		}
	}
}

void wavelet_by_region_rgb(vector<Mat> &src, int num_frames, int reg_w, int reg_h,
	vector<Mat> &wt, vector<Mat> &rec, vector<vector<int>> &keep_regions_by_frames, int &regs_x, int &regs_y, vector<double> p, int ip,
	vector<double> u, int iu, int Ls, double th, int typeThresholding, bool isGlobalTh = true, bool isCDF97 = false, bool testInverse = true)
{

	int rows = src[0].rows, cols = src[0].cols;
	regs_x = cols / reg_w + ((cols % reg_w) > 0);
	regs_y = rows / reg_h + ((rows % reg_h) > 0);

	int n_pixels = rows*cols, n_regions = regs_x * regs_y;

	// initial rgb energy
	vector<double> rgb_energy(n_regions, 0.0);

	// copy data 
	vector<vector<double>> dataR(n_pixels, vector<double>(num_frames, 0.0));
	vector<vector<double>> dataG(n_pixels, vector<double>(num_frames, 0.0));
	vector<vector<double>> dataB(n_pixels, vector<double>(num_frames, 0.0));
	for (int idx = 0; idx < n_pixels; idx++){
		int y = idx / cols, x = idx % cols;
		for (size_t i = 0; i < num_frames; i++){
			dataR[idx][i] = src[i].at<Vec3b>(y, x)[2];
			dataG[idx][i] = src[i].at<Vec3b>(y, x)[1];
			dataB[idx][i] = src[i].at<Vec3b>(y, x)[0];
		}

		int _y = y / reg_h, _x = x / reg_w;
		int _id = _y * regs_x + _x;
		rgb_energy[_id] += dataR[idx][0] * dataR[idx][0] + dataG[idx][0] * dataG[idx][0] + dataB[idx][0] * dataB[idx][0];
	}

	// DWT 
	vector<vector<double>> dwtR(n_pixels, vector<double>(num_frames, 0.0));
	vector<vector<double>> dwtG(n_pixels, vector<double>(num_frames, 0.0));
	vector<vector<double>> dwtB(n_pixels, vector<double>(num_frames, 0.0));
	for (int idx = 0; idx < n_pixels; idx++){
		if (isCDF97){
			wt97_in_place(dataR[idx], dwtR[idx], Ls);
			wt97_in_place(dataG[idx], dwtG[idx], Ls);
			wt97_in_place(dataB[idx], dwtB[idx], Ls);
		}
		else {
			wt1D(dataR[idx], dwtR[idx], p, ip, u, iu, Ls);
			wt1D(dataG[idx], dwtG[idx], p, ip, u, iu, Ls);
			wt1D(dataB[idx], dwtB[idx], p, ip, u, iu, Ls);
		}
		int y = idx / cols;
		int x = idx % cols;

		for (size_t i = 0; i < num_frames; i++){
			wt[i].at<Vec3d>(y, x)[2] = dwtR[idx][i];
			wt[i].at<Vec3d>(y, x)[1] = dwtG[idx][i];
			wt[i].at<Vec3d>(y, x)[0] = dwtB[idx][i];
		}

		if (!testInverse)continue;
		// inverse
		if (isCDF97){
			iwt97_in_place(dwtR[idx], dataR[idx], Ls);
			iwt97_in_place(dwtG[idx], dataG[idx], Ls);
			iwt97_in_place(dwtB[idx], dataB[idx], Ls);
		}
		else {
			iwt1D(dwtR[idx], dataR[idx], p, ip, u, iu, Ls);
			iwt1D(dwtG[idx], dataG[idx], p, ip, u, iu, Ls);
			iwt1D(dwtB[idx], dataB[idx], p, ip, u, iu, Ls);
		}
		
		for (size_t i = 0; i < num_frames; i++){
			rec[i].at<Vec3b>(y, x)[2] = dataR[idx][i];
			rec[i].at<Vec3b>(y, x)[1] = dataG[idx][i];
			rec[i].at<Vec3b>(y, x)[0] = dataB[idx][i]; 
		}
	}


	double minV, maxV;
	vector<Mat> sp;

	/*for (size_t i = 0; i < num_frames; i++){
	split(wt[i], sp);
	minMaxIdx(sp[0], &minV, &maxV);
	sp[0] = (sp[0] - minV) / (maxV - minV);
	minMaxIdx(sp[1], &minV, &maxV);
	sp[1] = (sp[1] - minV) / (maxV - minV);
	minMaxIdx(sp[2], &minV, &maxV);
	sp[2] = (sp[2] - minV) / (maxV - minV);

	merge(sp, wt[i]);
	}*/

	for (size_t i = 0; i < num_frames; i++){
		split(wt[i], sp);
		minMaxIdx(sp[0], &minV, &maxV);
		maxV = max(abs(minV), maxV);
		sp[0] = abs(sp[0]) / maxV;

		minMaxIdx(sp[1], &minV, &maxV);
		maxV = max(abs(minV), maxV);
		sp[1] = abs(sp[1]) / maxV;

		minMaxIdx(sp[2], &minV, &maxV);
		maxV = max(abs(minV), maxV);
		sp[2] = abs(sp[2]) / maxV;

		merge(sp, wt[i]);
	}


	// take into account only detail coefficients in dwt
	vector<vector<double>> energies(num_frames, vector<double>(n_regions, 0.0));
	vector<double> max_energy(n_regions, 0);
	vector<double> mean_energy(n_regions, 0);
	vector<double> cumul_energy(n_regions, 0);
	double global_max_energy = 0;
	for (size_t i = 1; i < num_frames; i++){

		for (int idx = 0; idx < n_pixels; idx++){
			int _y = (idx / cols) / reg_h, _x = (idx % cols) / reg_w;
			int _id = _y * regs_x + _x;
			energies[i][_id] += dwtR[idx][i] * dwtR[idx][i];
			energies[i][_id] += dwtG[idx][i] * dwtG[idx][i];
			energies[i][_id] += dwtB[idx][i] * dwtB[idx][i];
		}

		for (size_t j = 0; j < n_regions; j++){
			mean_energy[j] += energies[i][j];
			if (energies[i][j] > max_energy[j])
				max_energy[j] = energies[i][j];
			if (energies[i][j] > global_max_energy)
				global_max_energy = energies[i][j];
		}
	}

	double global_mean_energy = 0;
	for (size_t j = 0; j < n_regions; j++){
		global_mean_energy += mean_energy[j];
	}
	global_mean_energy /= ((num_frames - 1) * n_regions);
	
	for (size_t j = 0; j < n_regions; j++){
		mean_energy[j] /= (num_frames - 1);
	}

	// Find kept regions of frames
	keep_regions_by_frames.resize(num_frames);
	for (size_t i = 0; i < n_regions; i++){
		keep_regions_by_frames[0].push_back(i);
	}
	int saved_regions = num_frames;
	int whole_skip = 0;
	for (size_t i = 1; i < num_frames; i++){
		bool isSkip = true;
		for (size_t j = 0; j < n_regions; j++){
			switch (typeThresholding){
				case ENERGY_TH:
				if (energies[i][j] / rgb_energy[j] > th){
					keep_regions_by_frames[i].push_back(j);
					saved_regions++;
					isSkip = false;
				}
				break;
				case MAX_TH:
					if (energies[i][j] / max_energy[j] > th && !isGlobalTh ||
						energies[i][j] / global_max_energy > th && isGlobalTh){
						keep_regions_by_frames[i].push_back(j);
						saved_regions++;
						isSkip = false;
					}
					break;

				case MEAN_TH: 
					if (energies[i][j] / global_mean_energy > th && isGlobalTh ||
						energies[i][j] / mean_energy[j] > th && !isGlobalTh){
						keep_regions_by_frames[i].push_back(j);
						saved_regions++;
						isSkip = false;
					}
						
					break;

				case CUMULATION_TH:
					cumul_energy[j] += energies[i][j];
					if (cumul_energy[j] / rgb_energy[j] > th){
						keep_regions_by_frames[i].push_back(j);
						saved_regions++;
						isSkip = false;
						cumul_energy[j] = 0.0;
						rgb_energy[j] = 0.0;
						int reg_x = j % regs_x, reg_y = j / regs_x;
						int end_x = ((reg_x + 1) * reg_w < cols ? (reg_x + 1) * reg_w : cols);
						int end_y = ((reg_y + 1) * reg_h < rows ? (reg_y + 1) * reg_h : rows);
						for (size_t iy = reg_y * reg_h; iy < end_y; iy++){
							for (size_t ix = reg_x * reg_w; ix < end_x; ix++){
								int id = iy * cols + ix;
								rgb_energy[j] += dataR[id][i] * dataR[id][i] + dataG[id][i] * dataG[id][i] + dataB[id][i] * dataB[id][i];
							}
						}
					}
					break;
			}
		}
		
		if (isSkip)whole_skip++;
	}

	double saved_percent = double(saved_regions) / (num_frames * n_regions);
	cout << "number of frames = " << num_frames << endl;
	cout << "skipped frames = " << whole_skip << endl;
	cout << "mean_skip_with_0_frame = " << 1.0 - saved_percent << endl;
	cout << "mean_skipped = " << 1.0 - saved_percent + 1.0 / num_frames << endl;
}

void wavelet_by_region_gray(vector<Mat> &src, int num_frames, int reg_w, int reg_h,
	vector<Mat> &wt, vector<Mat> &rec, vector<vector<int>> &keep_regions_by_frames, int &regs_x, int &regs_y, vector<double> p, int ip,
	vector<double> u, int iu, int Ls, double th, int typeThresholding, bool isGlobalTh = true, bool isCDF97 = false, bool testInverse = true)
{

	int rows = src[0].rows, cols = src[0].cols;
	regs_x = cols / reg_w + ((cols % reg_w) > 0);
	regs_y = rows / reg_h + ((rows % reg_h) > 0);

	int n_pixels = rows*cols, n_regions = regs_x * regs_y;
	int saved_regions = 0;
	// initial rgb energy
	vector<double> gray_energy(n_regions, 0.0);

	// copy data 
	vector<vector<double>> data(n_pixels, vector<double>(num_frames, 0.0));
	
	for (int idx = 0; idx < n_pixels; idx++){
		int y = idx / cols, x = idx % cols;
		for (size_t i = 0; i < num_frames; i++){
			data[idx][i] = src[i].at<uchar>(y, x);
		}

		int _y = y / reg_h, _x = x / reg_w;
		int _id = _y * regs_x + _x;
		gray_energy[_id] += data[idx][0] * data[idx][0];
	}

	// DWT 
	vector<vector<double>> dwt(n_pixels, vector<double>(num_frames, 0.0));
	for (int idx = 0; idx < n_pixels; idx++){
		if (isCDF97){
			wt97_in_place(data[idx], dwt[idx], Ls);
		}
		else {
			wt1D(data[idx], dwt[idx], p, ip, u, iu, Ls);
		}
		int y = idx / cols;
		int x = idx % cols;

		for (size_t i = 0; i < num_frames; i++){
			wt[i].at<uchar>(y, x) = dwt[idx][i];
		}

		if (!testInverse)continue;
		// inverse
		if (isCDF97){
			iwt97_in_place(dwt[idx], data[idx], Ls);
		}
		else {
			iwt1D(dwt[idx], data[idx], p, ip, u, iu, Ls);
		}

		for (size_t i = 0; i < num_frames; i++){
			rec[i].at<uchar>(y, x) = data[idx][i];
		}
	}


	double minV, maxV;

	for (size_t i = 0; i < num_frames; i++){
		minMaxIdx(wt[i], &minV, &maxV);
		maxV = max(abs(minV), maxV);
		wt[i] = abs(wt[i]) / maxV;
	}


	// take into account only detail coefficients in dwt
	vector<vector<double>> energies(num_frames, vector<double>(n_regions, 0.0));
	vector<double> max_energy(n_regions, 0);
	vector<double> mean_energy(n_regions, 0);
	vector<double> cumul_energy(n_regions, 0);
	vector<double> ssq_cumul(n_pixels, 0);
	double global_max_energy = 0;
	for (size_t i = 1; i < num_frames; i++){

		for (int idx = 0; idx < n_pixels; idx++){
			int _y = (idx / cols) / reg_h, _x = (idx % cols) / reg_w;
			int _id = _y * regs_x + _x;
			energies[i][_id] += dwt[idx][i] * dwt[idx][i];
		}

		for (size_t j = 0; j < n_regions; j++){
			mean_energy[j] += energies[i][j];
			if (energies[i][j] > max_energy[j])
				max_energy[j] = energies[i][j];
			if (energies[i][j] > global_max_energy)
				global_max_energy = energies[i][j];
		}
	}

	double global_mean_energy = 0;
	for (size_t j = 0; j < n_regions; j++){
		global_mean_energy += mean_energy[j];
	}
	global_mean_energy /= ((num_frames - 1) * n_regions);

	for (size_t j = 0; j < n_regions; j++){
		mean_energy[j] /= (num_frames - 1);
	}

	// Find kept regions of frames
	keep_regions_by_frames.resize(num_frames);
	for (size_t i = 0; i < n_regions; i++){
		keep_regions_by_frames[0].push_back(i);
	}
	saved_regions += n_regions;
	int whole_skip = 0;
	for (size_t i = 1; i < num_frames; i++){
		bool isSkip = true;
		
		for (size_t j = 0; j < n_regions; j++){
			switch (typeThresholding){

			case ENERGY_TH:
				if (energies[i][j] / gray_energy[j] > th){
					keep_regions_by_frames[i].push_back(j);
					saved_regions++;
					isSkip = false;
				}
				break;
			case MAX_TH:
				if (energies[i][j] / max_energy[j] > th && !isGlobalTh ||
					energies[i][j] / global_max_energy > th && isGlobalTh){
					keep_regions_by_frames[i].push_back(j);
					saved_regions++;
					isSkip = false;
				}
				break;

			case MEAN_TH:
				if (energies[i][j] / global_mean_energy > th && isGlobalTh ||
					energies[i][j] / mean_energy[j] > th && !isGlobalTh){
					keep_regions_by_frames[i].push_back(j);
					saved_regions++;
					isSkip = false;
				}
					
				break;

			case CUMULATION_TH:
				cumul_energy[j] = cumul_energy[j] + energies[i][j];
				if (cumul_energy[j] / gray_energy[j] > th){
					keep_regions_by_frames[i].push_back(j);
					saved_regions++;
					isSkip = false;

					cumul_energy[j] = 0.0;
					gray_energy[j] = 0.0;
					int reg_x = j % regs_x, reg_y = j / regs_x;
					int end_x = ((reg_x + 1) * reg_w < cols ? (reg_x + 1) * reg_w : cols);
					int end_y = ((reg_y + 1) * reg_h < rows ? (reg_y + 1) * reg_h : rows);
					for (size_t iy = reg_y * reg_h; iy < end_y; iy++){
						for (size_t ix = reg_x * reg_w; ix < end_x; ix++){
							int id = iy * cols + ix;
							gray_energy[j] += data[id][i] * data[id][i];
						}
					}
				}
				break;
			case SSQ_CUMULATION_TH:
				
				int reg_x = j % regs_x, reg_y = j / regs_x;
				int end_x = ((reg_x + 1) * reg_w < cols ? (reg_x + 1) * reg_w : cols);
				int end_y = ((reg_y + 1) * reg_h < rows ? (reg_y + 1) * reg_h : rows);
				double ssq_cumul_reg = 0;
				for (size_t iy = reg_y * reg_h; iy < end_y; iy++){
					for (size_t ix = reg_x * reg_w; ix < end_x; ix++){
						int id = iy * cols + ix;
						ssq_cumul[id] += dwt[id][i];
						ssq_cumul_reg += ssq_cumul[id] * ssq_cumul[id];
					}
				}

				if (ssq_cumul_reg / gray_energy[j] > th){
					keep_regions_by_frames[i].push_back(j);
					saved_regions++;
					isSkip = false;
					gray_energy[j] = 0.0;
					for (size_t iy = reg_y * reg_h; iy < end_y; iy++){
						for (size_t ix = reg_x * reg_w; ix < end_x; ix++){
							int id = iy * cols + ix;
							ssq_cumul[id] = 0.0;
							gray_energy[j] += data[id][i] * data[id][i];
						}
					}
				}

				break;
			}
		}

		if (isSkip)whole_skip++;
	}

	double saved_percent = double(saved_regions) / (num_frames * n_regions);
	cout << "number of frames = " << num_frames << endl;
	cout << "skipped frames = " << whole_skip << endl;
	cout << "mean_skip_with_0_frame = "<<1.0 - saved_percent<<endl;
	cout << "mean_skipped = " << 1.0 - saved_percent + 1.0 / num_frames<< endl;
}

void dark_regions(int reg_w, int reg_h, int regs_x, int regs_y, Mat &src, vector<int> sign_region, Mat &dst)
{
	dst = 0.5 * src.clone();
	for (int i = 0; i < sign_region.size(); i++){
		int reg_x = sign_region[i] % regs_x, reg_y = sign_region[i] / regs_x;

		int end_x = min((reg_x + 1) * reg_w, src.cols);
		int end_y = min((reg_y + 1) * reg_h, src.rows);
		for (size_t iy = reg_y * reg_h; iy < end_y; iy++){
			for (size_t ix = reg_x * reg_w; ix < end_x; ix++){
				dst.at<Vec3b>(iy, ix) = src.at<Vec3b>(iy, ix);
			}
		}
	}
}

void alpha_regions(int reg_w, int reg_h, int regs_x, int regs_y, Mat &src, vector<int> sign_region, Mat &dst)
{
	dst = 0.95 * dst;
	for (int i = 0; i < sign_region.size(); i++){
		int reg_x = sign_region[i] % regs_x, reg_y = sign_region[i] / regs_x;

		int end_x = min((reg_x + 1) * reg_w, src.cols);
		int end_y = min((reg_y + 1) * reg_h, src.rows);
		for (size_t iy = reg_y * reg_h; iy < end_y; iy++){
			for (size_t ix = reg_x * reg_w; ix < end_x; ix++){
				dst.at<Vec3b>(iy, ix) = src.at<Vec3b>(iy, ix);
			}
		}
	}
}

int max_dwt_level(int batch_size){
	int ret = 0, res = 1;
	while (res < batch_size){
		ret++;
		res <<= 1;
	}

	return ret;
}

void write_blocks_file(string file_name, int num_frames, int block_size, vector<vector<int>> saved_blocks)
{
	ofstream out(file_name);
	for (size_t i = 0; i < num_frames; i++){
		out << i << ";" << block_size << ";";
		for (size_t j = 0; j < saved_blocks[i].size();  j++){
			out << saved_blocks[i][j] << " ";
		}
		out << endl;
	}
}

vector<vector<int>> blocks_from_wavelets_rgb(vector<Mat> &src, int block_width, int block_height, double threshold, bool isCDF97 = false)
{
	int num_frames = src.size();
	int Ls = max_dwt_level(num_frames);
	int rows = src[0].rows, cols = src[0].cols;
	int blocks_x = cols / block_width + ((cols % block_width) > 0);
	int blocks_y = rows / block_height + ((rows % block_height) > 0);
	int n_pixels = rows*cols, n_blocks = blocks_x * blocks_y;

	// initial rgb energy
	vector<double> rgb_energy(n_blocks, 0.0);

	// copy data 
	vector<vector<double>> dataR(n_pixels, vector<double>(num_frames, 0.0));
	vector<vector<double>> dataG(n_pixels, vector<double>(num_frames, 0.0));
	vector<vector<double>> dataB(n_pixels, vector<double>(num_frames, 0.0));
	for (int idx = 0; idx < n_pixels; idx++){
		int y = idx / cols, x = idx % cols;
		for (size_t i = 0; i < num_frames; i++){
			dataR[idx][i] = src[i].at<Vec3b>(y, x)[2];
			dataG[idx][i] = src[i].at<Vec3b>(y, x)[1];
			dataB[idx][i] = src[i].at<Vec3b>(y, x)[0];
		}

		int _y = y / block_height, _x = x / block_width;
		int _id = _y * blocks_x + _x;
		rgb_energy[_id] += dataR[idx][0] * dataR[idx][0] + dataG[idx][0] * dataG[idx][0] + dataB[idx][0] * dataB[idx][0];
	}

	// DWT 
	vector<vector<double>> dwtR(n_pixels, vector<double>(num_frames, 0.0));
	vector<vector<double>> dwtG(n_pixels, vector<double>(num_frames, 0.0));
	vector<vector<double>> dwtB(n_pixels, vector<double>(num_frames, 0.0));

	// helpful data for interpolating wavelets
	double p6[6] = { 3. / 256, -25. / 256, 75. / 128, 75. / 128, -25. / 256, 3. / 256 };
	int ip6 = -2;

	for (int idx = 0; idx < n_pixels; idx++){
		if (isCDF97){
			wt97_in_place(dataR[idx], dwtR[idx], Ls);
			wt97_in_place(dataG[idx], dwtG[idx], Ls);
			wt97_in_place(dataB[idx], dwtB[idx], Ls);
		}
		else {
			wt1D(dataR[idx], dwtR[idx], vector<double>(p6, p6 + 6), ip6, vector<double>(p6, p6 + 6), ip6, Ls);
			wt1D(dataG[idx], dwtG[idx], vector<double>(p6, p6 + 6), ip6, vector<double>(p6, p6 + 6), ip6, Ls);
			wt1D(dataB[idx], dwtB[idx], vector<double>(p6, p6 + 6), ip6, vector<double>(p6, p6 + 6), ip6, Ls);
		}
	}

	// take into account only detail coefficients in dwt
	vector<vector<double>> energies(num_frames, vector<double>(n_blocks, 0.0));
	for (size_t i = 1; i < num_frames; i++){
		for (int idx = 0; idx < n_pixels; idx++){
			int _y = (idx / cols) / block_height, _x = (idx % cols) / block_width;
			int _id = _y * blocks_x + _x;
			energies[i][_id] += dwtR[idx][i] * dwtR[idx][i];
			energies[i][_id] += dwtG[idx][i] * dwtG[idx][i];
			energies[i][_id] += dwtB[idx][i] * dwtB[idx][i];
		}
	}

	vector<vector<int>> keep_blocks(num_frames);
	// Find kept regions of frames
	for (size_t i = 0; i < n_blocks; i++){
		keep_blocks[0].push_back(i);
	}
	for (size_t i = 1; i < num_frames; i++){
		for (size_t j = 0; j < n_blocks; j++){
			if (energies[i][j] / rgb_energy[j] > threshold){
				keep_blocks[i].push_back(j);
			}
		}
	}

	return keep_blocks;
}

vector<vector<int>> blocks_from_wavelets_gray(vector<Mat> &src, int block_width, int block_height, double threshold, bool isCDF97 = false)
{
	int num_frames = src.size();
	int Ls = max_dwt_level(num_frames);
	int rows = src[0].rows, cols = src[0].cols;
	int blocks_x = cols / block_width + ((cols % block_width) > 0);
	int blocks_y = rows / block_height + ((rows % block_height) > 0);
	int n_pixels = rows*cols, n_blocks = blocks_x * blocks_y;

	// initial gray energy
	vector<double> gray_energy(n_blocks, 0.0);

	// copy data 
	vector<vector<double>> data(n_pixels, vector<double>(num_frames, 0.0));
	for (int idx = 0; idx < n_pixels; idx++){
		int y = idx / cols, x = idx % cols;
		for (size_t i = 0; i < num_frames; i++){
			data[idx][i] = src[i].at<uchar>(y, x);
		}

		int _y = y / block_height, _x = x / block_width;
		int _id = _y * blocks_x + _x;
		gray_energy[_id] += data[idx][0] * data[idx][0];
	}

	// DWT 
	vector<vector<double>> dwt(n_pixels, vector<double>(num_frames, 0.0));

	// helpful data for interpolating wavelets (use 6 point 5-degree polynom)
	double p6[6] = { 3. / 256, -25. / 256, 75. / 128, 75. / 128, -25. / 256, 3. / 256 };
	int ip6 = -2;

	for (int idx = 0; idx < n_pixels; idx++){
		if (isCDF97){
			wt97_in_place(data[idx], dwt[idx], Ls);
		}
		else {
			wt1D(data[idx], dwt[idx], vector<double>(p6, p6 + 6), ip6, vector<double>(p6, p6 + 6), ip6, Ls);
		}
	}

	// take into account only detail coefficients in dwt
	vector<vector<double>> energies(num_frames, vector<double>(n_blocks, 0.0));
	for (size_t i = 1; i < num_frames; i++){
		for (int idx = 0; idx < n_pixels; idx++){
			int _y = (idx / cols) / block_height, _x = (idx % cols) / block_width;
			int _id = _y * blocks_x + _x;
			energies[i][_id] += dwt[idx][i] * dwt[idx][i];
		}
	}


	// Find kept blocks of frames
	vector<vector<int>> keep_blocks(num_frames);
	for (size_t i = 0; i < n_blocks; i++){
		keep_blocks[0].push_back(i);
	}
	for (size_t i = 1; i < num_frames; i++){
		for (size_t j = 0; j < n_blocks; j++){
			if (energies[i][j] / gray_energy[j] > threshold){
				keep_blocks[i].push_back(j);
			}
		}
	}

	return keep_blocks;
}

int main(int argc, char *argv [])
{
	string name_video = string("bike");//bike.mp4 the source file name
	const string source = name_video + string(".mp4");

	VideoCapture inputVideo(source);//source             
	if (!inputVideo.isOpened()){
		cout << "Could not open the input video: " << source << endl;
		return -1;
	}

	double fps = inputVideo.get(CV_CAP_PROP_FPS);
	cout << "input video fps = " << fps << endl;

	Size S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
		(int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
	cout <<S.height<< "x" << S.width << endl;
	namedWindow("video", 1);

	// size for yolo input
	Size S_new(416, 416);

	// if unused CDF97 wavelets => use interpolating wavelets in below lines
	// cubic interpolation filter coefficients
	double p4[4] = { -1. / 16, 9. / 16, 9. / 16, -1. / 16 };
	int ip4 = -1;

	// 5-degree polynomial interpolation coefficients
	double p6[6] = { 3. / 256, -25. / 256, 75. / 128, 75. / 128, -25. / 256, 3. / 256 };
	int ip6 = -2;

	int batch_size = (int) inputVideo.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "number of frames = " << batch_size << endl;
	int Ls = max_dwt_level(batch_size);

	// Main parameters 
	int tile_w = 8, tile_h = 8;
	double threshold = 0.05;
	int typeTh = ENERGY_TH;//SSQ_CUMULATION_TH, MAX_TH, MEAN_TH, SSQ_
	bool isGlobal = false, isRGB = true;
	bool isUseCDF97 = true;//Cohen-Daubechies-Feauveau (CDF)9/7 wavelets
	
	// forming name of output file
	string strTypeTh;
	switch (typeTh){
	case ENERGY_TH:
		strTypeTh = string("energy_");
		break;
	case MAX_TH:
		strTypeTh = string("max_");
		break;
	case MEAN_TH:
		strTypeTh = string("mean_");
		break;
	case CUMULATION_TH:
		strTypeTh = string("cumul_");
		break;
	case SSQ_CUMULATION_TH:
		strTypeTh = string("ssq_cum_");
		break;
	}
	string file_reduce = name_video + string("_") + 
		(isUseCDF97 ? string("cdf97_"): string("p6p6_")) + 
		to_string(tile_w) + string("x") + to_string(tile_h) + string("_th") + 
		to_string(threshold) + string("_") +
		strTypeTh +
		(isRGB ? string("rgb"): string("gray")) + string(".avi");
	
	string file_saved_blocks = name_video + string("_blocks_") +
		(isUseCDF97 ? string("cdf97_") : string("p6p6_")) +
		to_string(tile_w) + string("x") + to_string(tile_h) + string("_th") +
		to_string(threshold) + string("_") + strTypeTh + 
		(isRGB ? string("rgb") : string("gray")) + string(".txt");

	VideoWriter outputVideo(file_reduce, CV_FOURCC('M', 'P', 'E', 'G'), fps, S_new, true);
	VideoWriter outputVideo2("reduce_flash.avi", CV_FOURCC('M', 'P', 'E', 'G'), fps, S_new, true);

	if (!outputVideo.isOpened()){
		cout << "Could not open the output video for write: " << source << endl;
		waitKey(1000);
		return -1;
	}
	if (!outputVideo2.isOpened()){
		cout << "Could not open the output video for write: " << source << endl;
		waitKey(1000);
		return -1;
	}
	VideoWriter outputVideo3("dwt.avi", CV_FOURCC('M', 'P', 'E', 'G'), fps, S_new, isRGB); //HFYU - lossless  
	VideoWriter outputVideo4("rec.avi", CV_FOURCC('M', 'P', 'E', 'G'), fps, S_new, isRGB); //HFYU - lossless 
	if (!outputVideo3.isOpened()){
		cout << "Could not open the output video for write: " << source << endl;
		waitKey(1000);
		return -1;
	}
	if (!outputVideo4.isOpened()){
		cout << "Could not open the output video for write: " << source << endl;
		waitKey(1000);
		return -1;
	}

	/* coefficients for non-lifting scheme
	Taps	Low Pass Filter		Taps	High Pass Filter
	0, 8	0.026748757410		0, 6	0.091271763114
	1, 7	–0.016864118442		1, 5	–0.057543526228
	2, 6	–0.078223266528		2, 4	–0.591271763114
	3, 5	0.266864118442		3		1.111508705245
	4		0.602949018326
	*/

	// ---- check dwt
	const int _N = 23, _L = 5;
	double test[_N] = { 1.0, 1.5, 2.0, 1.5, 2.0, 3.0, 4.0, 5.5, 6.0, 3.5, 4.5, 1.0, 1.5, 3.5, 6.0, 9.0, -2., -4, -3, -1, -6.9, -9.0, -8.0 };

	vector<double> x(test, test + _N), y(_N);
	wt97_in_place(x, y, _L);
	//wt1D(x, y, vector<double>(p6, p6 + sizeof(p6) / sizeof(double)), ip6, vector<double>(p6, p6 + sizeof(p6) / sizeof(double)), ip6, _L);
	cout << "dwt 9/7" << endl;
	for (size_t i = 0; i < _N; i++){
		cout << y[i] << " ";
	}
	cout << endl;

	iwt97_in_place(y, x, _L);
	//iwt1D(y, x, vector<double>(p6, p6 + sizeof(p6) / sizeof(double)), ip6, vector<double>(p6, p6 + sizeof(p6) / sizeof(double)), ip6, _L);
	for (size_t i = 0; i < _N; i++){
		cout << x[i] << " ";
	}
	cout << endl;
	// ---- end test dwt

	int counter = 0;
	vector<vector<int>> regions;
	vector<Mat> src(batch_size); 
	vector<Mat> rec(batch_size); 
	vector<Mat> res(batch_size);
	vector<Mat> gray(batch_size);
	for (int i = 0; i < batch_size; i++){
		src[i].create(S_new, CV_8UC3);
		
		if (!isRGB){
			gray[i].create(S_new, CV_8U);
			res[i].create(S_new, CV_64F);
			rec[i].create(S_new, CV_8U);
		}
		else {
			res[i].create(S_new, CV_64FC3);
			rec[i].create(S_new, CV_8UC3);
		}
	}

	bool isStop = false;
	Mat accum_frame(S_new, CV_8UC3);
	int n_frames = 0;
	for (;;) //Show the image captured in the window and repeat
	{
		Mat frame;
		inputVideo >> frame;
		resize(frame, frame, S_new);
		// check if at end
		if (!frame.empty()){
			src[counter] = frame.clone();
			if (!isRGB)cvtColor(frame, gray[counter], CV_BGR2GRAY);

			imshow("video", frame);
			counter++;
			n_frames++;
		}
		else {
			isStop = true;
		}

		if ((counter == batch_size) || isStop){
			if (isStop){
				batch_size = (batch_size < counter ? batch_size : counter);
				Ls = max_dwt_level(batch_size);
			}

			int regs_x, regs_y;

			// find saved regions of frames
			if (isRGB)wavelet_by_region_rgb(src, batch_size, tile_w, tile_h, res, rec, regions, regs_x, regs_y,
				vector<double>(p6, p6 + sizeof(p6) / sizeof(double)), ip6, vector<double>(p6, p6 + sizeof(p6) / sizeof(double)), ip6,
				Ls, threshold, typeTh, isGlobal, isUseCDF97, true);
			else {
				wavelet_by_region_gray(gray, batch_size, tile_w, tile_h, res, rec, regions, regs_x, regs_y,
					vector<double>(p6, p6 + sizeof(p6) / sizeof(double)), ip6, vector<double>(p6, p6 + sizeof(p6) / sizeof(double)), ip6,
					Ls, threshold, typeTh, isGlobal, isUseCDF97, true);
			}

			// reduced video s
			accum_frame = src[0].clone();
			outputVideo << accum_frame;
			for (size_t i = 1; i < batch_size; i++){
				alpha_regions(tile_w, tile_h, regs_x, regs_y, src[i], regions[i], accum_frame);
				outputVideo << accum_frame;
			}

			/*   Test/Debug output videos*/
			
			for (size_t i = 0; i < batch_size; i++){
				Mat tmp_frame(S_new, CV_8UC3);
				dark_regions(tile_w, tile_h, regs_x, regs_y, src[i], regions[i], tmp_frame);
				outputVideo2 << tmp_frame;
			}

			for (size_t i = 0; i < batch_size; i++){
				if (isRGB){
					Mat tmp_frame(S_new, CV_8UC3);
					res[i].convertTo(tmp_frame, CV_8UC3, 255.);
					outputVideo3 << tmp_frame;
				}
				else{
					Mat tmp_frame(S_new, CV_8U);
					res[i].convertTo(tmp_frame, CV_8U, 255.);
					outputVideo3 << tmp_frame;
				}
			}

			// reconstruct video
			for (int i = 0; i < batch_size; i++){
				outputVideo4 << rec[i];
			}
			
			// write file of saved blocks
			write_blocks_file(file_saved_blocks, batch_size, tile_w, regions);

			break;
		}
		if (waitKey(1) >= 0)break;
	}

	cout << "num capture frames = "<< n_frames<< endl;
	cout << "Finished writing" << endl;
	waitKey();
	return 0;
}