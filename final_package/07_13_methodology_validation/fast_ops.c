#include <math.h>
#include <stdlib.h>

#define RAD(deg) ((deg) * M_PI / 180.0)

// 1. Fast CUSUM Drift Detection
int detect_cusum_c(const double* values, int n, double threshold, double drift) {
    if (n <= 0) return 0;
    
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        mean += values[i];
    }
    mean /= n;
    
    double pos = 0.0;
    double neg = 0.0;
    int flags = 0;
    
    for (int i = 0; i < n; i++) {
        pos = fmax(0.0, pos + (values[i] - mean - drift));
        neg = fmax(0.0, neg + (mean - values[i] - drift));
        
        if (pos > threshold || neg > threshold) {
            flags++;
            pos = 0.0;
            neg = 0.0;
        }
    }
    return flags;
}

// 2. Fast EWMA Volatility Detection
int detect_ewma_c(const double* values, int n, double alpha, double control_limit) {
    if (n <= 0) return 0;
    
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        mean += values[i];
    }
    mean /= n;
    
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = values[i] - mean;
        sum_sq += diff * diff;
    }
    double std = sqrt(sum_sq / n);
    if (std < 1e-6) return 0;
    
    double ewma_std = std * sqrt(alpha / (2.0 - alpha));
    double upper = mean + control_limit * ewma_std;
    double lower = mean - control_limit * ewma_std;
    
    double s = values[0];
    int flags = 0;
    for (int i = 0; i < n; i++) {
        s = alpha * values[i] + (1.0 - alpha) * s;
        if (s > upper || s < lower) {
            flags++;
        }
    }
    return flags;
}

// 3. Fast Pairwise Haversine Distance Matrix (meters)
void haversine_matrix_c(const double* lats, const double* lons, int n, double* dist_matrix) {
    double R = 6371000.0; // Earth radius in meters
    for (int i = 0; i < n; i++) {
        double phi1 = RAD(lats[i]);
        double lam1 = RAD(lons[i]);
        for (int j = 0; j < n; j++) {
            if (i == j) {
                dist_matrix[i * n + j] = 0.0;
                continue;
            }
            double phi2 = RAD(lats[j]);
            double lam2 = RAD(lons[j]);
            
            double dphi = phi2 - phi1;
            double dlam = lam2 - lam1;
            
            double a = sin(dphi / 2.0) * sin(dphi / 2.0) +
                       cos(phi1) * cos(phi2) * sin(dlam / 2.0) * sin(dlam / 2.0);
            double c = 2.0 * atan2(sqrt(a), sqrt(fmax(0.0, 1.0 - a)));
            dist_matrix[i * n + j] = R * c;
        }
    }
}

// 4. Fast Gaussian Adjacency Matrix
void gaussian_adjacency_c(const double* dist_matrix, int n, double sigma, double threshold, double* W_out) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                W_out[i * n + j] = 0.0;
            } else {
                double d = dist_matrix[i * n + j];
                if (isinf(d)) {
                    W_out[i * n + j] = 0.0;
                } else {
                    double w = exp(-pow(d / sigma, 2.0));
                    W_out[i * n + j] = (w < threshold) ? 0.0 : w;
                }
            }
        }
    }
}
