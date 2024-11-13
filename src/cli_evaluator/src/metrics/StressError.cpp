#include "StressError.hpp"

#include "GraphAlgorithms.hpp"
#include "Macros.hpp"

std::vector<std::string> StressError::getMetricValues(const OptionValues& options, const Graph& g, const VecList& coords) {
    std::vector<std::vector<int>> allDists = GraphAlgo::calculateAllPairShortestPaths(g);

    double fullStressScale = calculateOptimalFullStressScale(options, g, coords, allDists);
    LOG_DEBUG( "Using full stress scale factor of " << fullStressScale);

    double maxentStressScale = calculateOptimalMaxentStressScale(options, g, coords);
    LOG_DEBUG( "Using maxent stress scale factor of " << maxentStressScale);

    std::vector<std::string> result = {
        std::to_string(calculateFullStress(options, g, coords, allDists, fullStressScale)),  // full stress
        // std::to_string(calculateFullStress(g, coords, allDists, 1.0)),              // full stress no scale
        std::to_string(calculateMaxEntStress(options, g, coords, maxentStressScale)),  // maxent stress
        // std::to_string(calculateMaxEntStress(g, coords, 1.0)),                      // maxent stress no scale
    };

    return result;
}

std::vector<std::string> StressError::getMetricNames() {
    std::vector<std::string> result = {
        "full_stress",
        //"full_stress_no_scale",
        "maxent_stress",
        //"maxent_stress_no_scale",
    };
    return result;
}

double StressError::calculateOptimalFullStressScale(const OptionValues& options, const Graph& g, const VecList& coords, const std::vector<std::vector<int>>& allDist) {
    VecBuffer<1> buffer(options.embDimension);
    TmpVec<0> tmpVec(buffer);

    const int N = g.getNumVertices();

    // determine the scaling parameter to make embedding fair
    double dividend = 0.0;
    double divisor = 0.0;

    for (int u = 0; u < N; u++) {
        for (int v = u + 1; v < N; v++) {
            tmpVec = coords[u] - coords[v];
            double duvr = tmpVec.norm();
            double duvi = allDist[u][v];
            double wuv = 1.0 / (duvi * duvi);

            dividend += wuv * 2.0 * duvi * duvr;
            divisor += wuv * 2.0 * duvr * duvr;
        }
    }
    double fullStressScale = dividend / divisor;
    return fullStressScale;
}

double StressError::calculateOptimalMaxentStressScale(const OptionValues& options, const Graph& g, const VecList& coords) {
    VecBuffer<1> buffer(options.embDimension);
    TmpVec<0> tmpVec(buffer);

    const int N = g.getNumVertices();
    // calculate optimal scaling s, such that f(s) = min
    // mini s has to be determined through a quadratic function as^2+bs+c=0
    // calulate a,b,c first
    double a = 0;
    double b = 0;
    double c = 0;
    for (NodeId v = 0; v < N; v++) {
        for (NodeId u : g.getNeighbors(v)) {
            tmpVec = coords[u] - coords[v];
            double duvr = tmpVec.norm();
            double duvi = 1.0;
            double wuv = 1.0 / (duvi * duvi);

            a += 2 * wuv * duvr * duvr;
            b -= 2 * wuv * duvr * duvi;
        }
    }
    for (int u = 0; u < N; u++) {
        std::vector<int> neighbours = g.getNeighbors(u);
        for (int v = u + 1; v < N; v++) {
            if (std::find(neighbours.begin(), neighbours.end(), v) != neighbours.end()) {
                // skip neighbours
                continue;
            }
            c -= 1;
        }
    }
    c *= maxentMinAlpha;
    // scale everything sucht that a = 1;
    b /= a;
    c /= a;
    // pq-formula
    double x1 = (-b / 2.0) + std::sqrt((b / 2.0) * (b / 2.0) - c);
    double x2 = (-b / 2.0) - std::sqrt((b / 2.0) * (b / 2.0) - c);
    if ((x1 < 0.0 && x2 < 0.0) || (x1 > 0.0 && x2 > 0.0)) {
        LOG_ERROR( "Expected positiv and negative scale: " << x1 << ", " << x2);
    }
    double maxentStressScale = x1 < 0.0 ? x2 : x1;
    LOG_DEBUG( "Optimal scales are " << x1 << " and " << x2 << " chose: " << maxentStressScale);
    return maxentStressScale;
}

double StressError::calculateFullStress(const OptionValues& options, const Graph& g, const VecList& coords, const std::vector<std::vector<int>>& allDist, double scale) {
    VecBuffer<1> buffer(options.embDimension);
    TmpVec<0> tmpVec(buffer);

    const int N = g.getNumVertices();
    double fullStress = 0;
    for (int u = 0; u < N; u++) {
        for (int v = u + 1; v < N; v++) {
            tmpVec = coords[u] - coords[v];
            double duvr = tmpVec.norm();
            double duvi = allDist[u][v];
            double wuv = 1.0 / (duvi * duvi);

            fullStress += wuv * ((scale * duvr) - duvi) * ((scale * duvr) - duvi);
        }
    }

    return fullStress;
}

double StressError::calculateMaxEntStress(const OptionValues& options, const Graph& g, const VecList& coords, double scale) {
    VecBuffer<1> buffer(options.embDimension);
    TmpVec<0> tmpVec(buffer);

    const int N = g.getNumVertices();
    double maxentStress = 0.0;
    for (int u = 0; u < N; u++) {
        for (int v : g.getNeighbors(u)) {
            if (v <= u) continue;
            tmpVec = coords[u] - coords[v];
            double duvr = tmpVec.norm();
            double duvi = 1;
            double wuv = 1.0 / (duvi * duvi);

            maxentStress += wuv * (scale * duvr - duvi) * (scale * duvr - duvi);
        }
    }
    for (int u = 0; u < N; u++) {
        std::vector<int> neighbours = g.getNeighbors(u);
        for (int v = u + 1; v < N; v++) {
            if (std::find(neighbours.begin(), neighbours.end(), v) != neighbours.end()) {
                // skip neighbours
                continue;
            }

            tmpVec = coords[u] - coords[v];
            double duvr = tmpVec.norm();
            maxentStress -= maxentMinAlpha * std::log(scale * duvr);
        }
    }
    return maxentStress;
}
