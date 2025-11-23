/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <glm/detail/qualifier.hpp>
#include <glm/glm.hpp>
#include <mutex>
#include <numeric>
#include <print>
#include <stdexcept>
#include <vector>

// A tiny over-engineered solver to improve progress bar accuracy
// Disclaimer: AI written
template <class T, glm::length_t N, class Func>
std::pair<T, glm::vec<N, T>> fitNonlinear(std::span<const T>    Xs,
                                          std::span<const T>    Yx,
                                          const glm::vec<N, T>& initialGuess,
                                          Func&&                func)
{
  using Variables = glm::vec<N, T>;
  using MatN = std::array<Variables, N>;  // sadly glm is not generic enough to allow glm::mat<N, N, T>
  assert(Xs.size() == Yx.size());
  assert(Xs.size() >= N);  // Need at least N data points for N parameters

  Variables    params        = initialGuess;
  const size_t numPoints     = Xs.size();
  const int    maxIterations = 100;
  const T      epsilon       = T(1e-6);  // Convergence threshold

  // Compute typical scale of X values for adaptive step sizing
  T xScale = T(0);
  for(size_t i = 0; i < numPoints; ++i)
    xScale += std::abs(Xs[i]);
  xScale /= T(numPoints);
  xScale = std::max(xScale, T(1));  // Avoid zero scale

  // Adaptive parameters based on data scale and fp32 precision
  const T sqrtEps = std::sqrt(std::numeric_limits<T>::epsilon());  // ~3e-4 for fp32
  const T baseLambda = T(0.1);

  std::vector<T> residuals(numPoints);
  std::vector<T> jacobian(numPoints * N);

  T prevLeastSquares = std::numeric_limits<T>::max();

  for(int iteration = 0; iteration < maxIterations; ++iteration)
  {
    // Compute residuals and current least squares error
    T leastSquares = T(0);
    for(size_t i = 0; i < numPoints; ++i)
    {
      T predicted  = func(params, Xs[i]);
      residuals[i] = Yx[i] - predicted;
      leastSquares += residuals[i] * residuals[i];
    }

    // Check for convergence
    if(std::abs(prevLeastSquares - leastSquares) < epsilon)
    {
      return {leastSquares, params};
    }
    prevLeastSquares = leastSquares;

    // Compute Jacobian matrix using numerical differentiation with adaptive step size
    for(glm::length_t paramIdx = 0; paramIdx < N; ++paramIdx)
    {
      // Adaptive delta: scale by parameter magnitude and sqrt of machine epsilon
      // For large X values (e.g. 10^6), this ensures delta is large enough to avoid fp32 precision loss
      T paramScale = std::max(std::abs(params[paramIdx]), xScale);
      T delta      = sqrtEps * paramScale;

      Variables paramsOffset = params;
      paramsOffset[paramIdx] += delta;

      for(size_t i = 0; i < numPoints; ++i)
      {
        T predicted       = func(params, Xs[i]);
        T predictedOffset = func(paramsOffset, Xs[i]);
        jacobian[i * N + size_t(paramIdx)] = -(predictedOffset - predicted) / delta;
      }
    }

    // Solve normal equations: (J^T * J + λI) * deltaParams = -J^T * residuals
    // Build J^T * J
    MatN JtJ{};
    for(glm::length_t i = 0; i < N; ++i)
    {
      for(glm::length_t j = 0; j < N; ++j)
      {
        JtJ[size_t(i)][j] = T(0);
        for(size_t k = 0; k < numPoints; ++k)
        {
          JtJ[size_t(i)][j] +=
              jacobian[k * N + size_t(i)] * jacobian[k * N + size_t(j)];
        }
      }
    }

    // Add damping (Levenberg-Marquardt) scaled by diagonal elements
    // This makes lambda adaptive to the scale of the problem
    for(glm::length_t i = 0; i < N; ++i)
    {
      T diagonalScale = std::max(JtJ[size_t(i)][i], T(1e-6));  // Avoid division by zero
      JtJ[size_t(i)][i] += baseLambda * diagonalScale;
    }

    // Build -J^T * residuals (negative sign needed when J = ∂residual/∂params)
    Variables Jtr{};
    for(glm::length_t i = 0; i < N; ++i)
    {
      T sum = T(0);
      for(size_t k = 0; k < numPoints; ++k)
      {
        sum -= jacobian[k * N + size_t(i)] * residuals[k];
      }
      Jtr[i] = sum;
    }

    // Solve for parameter update using Gauss elimination
    MatN      A = JtJ;
    Variables b = Jtr;

    // Gaussian elimination with partial pivoting
    for(glm::length_t i = 0; i < N; ++i)
    {
      // Find pivot (only needed for N > 1), silencing coverity's dead code
      // warning
      if constexpr(N > 1)
      {
        glm::length_t maxRow = i;
        for(glm::length_t k = i + 1; k < N; ++k)
        {
          if(std::abs(A[size_t(i)][k]) > std::abs(A[size_t(i)][maxRow]))
            maxRow = k;
        }

        // Swap rows
        if(maxRow != i)
        {
          for(glm::length_t k = 0; k < N; ++k)
            std::swap(A[size_t(k)][i], A[size_t(k)][maxRow]);
          std::swap(b[i], b[maxRow]);
        }
      }

      // Check for singular matrix (shouldn't happen with proper damping)
      if(std::abs(A[size_t(i)][i]) < std::numeric_limits<T>::epsilon())
        break;  // Skip remaining iterations if matrix is singular

      // Forward elimination
      if constexpr(N > 1)
      {
        for(glm::length_t k = i + 1; k < N; ++k)
        {
          T factor = A[size_t(i)][k] / A[size_t(i)][i];
          b[k] -= factor * b[i];
          for(glm::length_t j = i; j < N; ++j)
          {
            A[size_t(j)][k] -= factor * A[size_t(j)][i];
          }
        }
      }
    }

    // Back substitution
    Variables deltaParams{};
    for(glm::length_t ii = 0; ii < N; ++ii)
    {
      glm::length_t i   = N - 1 - ii;
      T             sum = b[i];
      if constexpr(N > 1)
      {
        for(glm::length_t j = i + 1; j < N; ++j)
        {
          sum -= A[size_t(j)][i] * deltaParams[j];
        }
      }
      // Avoid division by zero (matrix might be singular if damping failed)
      if(std::abs(A[size_t(i)][i]) > std::numeric_limits<T>::epsilon())
        deltaParams[i] = sum / A[size_t(i)][i];
      else
        deltaParams[i] = T(0);
    }

    // Update parameters
    params += deltaParams;
  }

  // Final least squares calculation
  T leastSquares = T(0);
  for(size_t i = 0; i < numPoints; ++i)
  {
    T predicted = func(params, Xs[i]);
    T residual  = Yx[i] - predicted;
    leastSquares += residual * residual;
  }

  return {leastSquares, params};
}

// Tries to fit a few different functions to the given data and prints the
// simplest with the least error.
template <class T>
void printBestFit(std::span<const T> Xs, std::span<const T> Ys)
{
#if 0
  for(size_t i = 0; i < Xs.size(); ++i)
    std::print("{}\t{}\n", Xs[i], Ys[i]);
#endif
  auto constant = fitNonlinear(Xs, Ys, glm::vec1(0.0f),
                               [](const glm::vec1& v, T /* x */) { return v.x; });

  if(Xs.size() < 3)
  {
    std::print("Not enough data. Use constant fit: time = complexity + {}\n",
               constant.second.x);
    return;
  }
  auto linear =
      fitNonlinear(Xs, Ys, glm::vec2(1.0f, 0.0f),
                   [](const glm::vec2& v, T x) { return x * v.x + v.y; });
  auto log = fitNonlinear(Xs, Ys, glm::vec2(1.0f, 0.0f), [](const glm::vec2& v, T x) {
    // Transition to linear for small/negative x, match gradient at xmin
    const float xmin = 1e-3f;
    float       lx;
    if(x > xmin)
    {
      lx = v.x * logf(x) + v.y;
    }
    else
    {
      // log'(x) = v.x / x
      float slope     = v.x / xmin;
      float intercept = v.x * logf(xmin) + v.y - slope * xmin;
      lx              = slope * x + intercept;
    }
    return lx;
  });
  auto logLinear =
      fitNonlinear(Xs, Ys, glm::vec2(1.0f, 0.0f), [](const glm::vec2& v, T x) {
        // x * ln(x) - handle small/negative x similar to log
        const float xmin = 1e-3f;
        float       xlnx;
        if(x > xmin)
        {
          xlnx = v.x * x * logf(x) + v.y;
        }
        else
        {
          // (x*ln(x))' = ln(x) + 1
          float slope     = v.x * (logf(xmin) + 1.0f);
          float intercept = v.x * xmin * logf(xmin) + v.y - slope * xmin;
          xlnx            = slope * x + intercept;
        }
        return xlnx;
      });
  auto quadratic =
      fitNonlinear(Xs, Ys, glm::vec3(0.0f, 1.0f, 0.0f), [](const glm::vec3& v, T x) {
        return x * x * v.x + x * v.y + v.z;
      });
  auto exponential =
      fitNonlinear(Xs, Ys, glm::vec3(1.0f, 1.0f, 0.0f), [](const glm::vec3& v, T x) {
        return v.x * expf(v.y * x) + v.z;
      });
  auto power = fitNonlinear(Xs, Ys, glm::vec2(1.0f, 2.0f), [](const glm::vec2& v, T x) {
    // y = a * x^b - transition to linear for small/negative x to avoid zero gradient
    const float xmin = 1e-3f;
    if(x > xmin)
    {
      return v.x * powf(x, v.y);
    }
    else
    {
      // (a * x^b)' = a * b * x^(b-1)
      float slope     = v.x * v.y * powf(xmin, v.y - 1.0f);
      float intercept = v.x * powf(xmin, v.y) - slope * xmin;
      return slope * x + intercept;
    }
  });
  T best = std::min({constant.first, linear.first, log.first, logLinear.first,
                     quadratic.first, exponential.first, power.first});
  T preferSimpler{1.1f};
  if(constant.first <= best * preferSimpler)
    std::print("Fits constant best: time = {}\n", constant.second.x);
  else if(linear.first <= best * preferSimpler)
    std::print("Fits linear best: time = {} complexity + {}\n", linear.second.x,
               linear.second.y);
  else if(quadratic.first <= best * preferSimpler)
    std::print("Fits quadratic best: time = {} complexity^2 + {} complexity + {}\n",
               quadratic.second.x, quadratic.second.y, quadratic.second.z);
  else if(log.first <= best * preferSimpler)
    std::print("Fits log best: time = {} ln(complexity) + {}\n", log.second.x,
               log.second.y);
  else if(logLinear.first <= best * preferSimpler)
    std::print("Fits logLinear best: time = {} complexity * ln(complexity) + {}\n",
               logLinear.second.x, logLinear.second.y);
  else if(exponential.first <= best * preferSimpler)
    std::print("Fits exponential best: time = {} exp({} * complexity) + {}\n",
               exponential.second.x, exponential.second.y, exponential.second.z);
  else if(power.first <= best * preferSimpler)
    std::print("Fits power best: time = {} complexity^{}\n", power.second.x,
               power.second.y);
  else
    assert(false);
}

#if 0
// Test cases for fitNonlinear
class FitNonlinearTester
{
public:
  FitNonlinearTester()
  {
    // Test 1: Constant offset fit (y = x + c)
    {
      std::vector<float> Xs = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
      std::vector<float> Ys = {6.0f, 7.0f, 8.0f, 9.0f, 10.0f};  // y = x + 5
      auto result = fitNonlinear<float, 1>(Xs, Ys, glm::vec1(0.0f), [](const glm::vec1& v, float x) { return x + v.x; });
      assert(std::abs(result.second.x - 5.0f) < 0.01f);
      printf("Test 1 (constant offset): fitted c=%.3f (expected 5.0), error=%.6f\n", result.second.x, result.first);
    }
    
    // Test 2: Linear fit (y = ax + b)
    {
      std::vector<float> Xs = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
      std::vector<float> Ys = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};  // y = 2x + 1
      auto result = fitNonlinear<float, 2>(Xs, Ys, glm::vec2(1.0f, 0.0f), [](const glm::vec2& v, float x) { 
        return v.x * x + v.y; 
      });
      assert(std::abs(result.second.x - 2.0f) < 0.01f);
      assert(std::abs(result.second.y - 1.0f) < 0.01f);
      printf("Test 2 (linear): fitted y=%.3fx + %.3f (expected 2x + 1), error=%.6f\n", 
             result.second.x, result.second.y, result.first);
    }
    
    // Test 3: Quadratic fit (y = ax^2 + bx + c)
    {
      std::vector<float> Xs = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
      std::vector<float> Ys = {4.0f, 9.0f, 16.0f, 25.0f, 36.0f};  // y = x^2 + 2x + 1 = (x+1)^2
      for(size_t i = 0; i < Xs.size(); ++i)
        Ys[i] = Xs[i] * Xs[i] + 2.0f * Xs[i] + 1.0f;
      auto result = fitNonlinear<float, 3>(Xs, Ys, glm::vec3(1.0f, 2.0f, 1.0f), [](const glm::vec3& v, float x) { 
        return v.x * x * x + v.y * x + v.z; 
      });
      assert(std::abs(result.second.x - 1.0f) < 0.01f);
      assert(std::abs(result.second.y - 2.0f) < 0.01f);
      assert(std::abs(result.second.z - 1.0f) < 0.01f);
      printf("Test 3 (quadratic): fitted y=%.3fx^2 + %.3fx + %.3f (expected x^2 + 2x + 1), error=%.6f\n", 
             result.second.x, result.second.y, result.second.z, result.first);
    }
    
    // Test 4: Noisy data
    {
      std::vector<float> Xs = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
      std::vector<float> Ys = {3.1f, 5.2f, 6.9f, 9.1f, 10.8f, 13.2f, 14.9f, 17.1f};  // roughly y = 2x + 1
      auto result = fitNonlinear<float, 2>(Xs, Ys, glm::vec2(1.0f, 0.0f), [](const glm::vec2& v, float x) { 
        return v.x * x + v.y; 
      });
      printf("Test 4 (noisy data): fitted y=%.3fx + %.3f (expected ~2x + 1), error=%.6f\n", 
             result.second.x, result.second.y, result.first);
      assert(std::abs(result.second.x - 2.0f) < 0.2f);  // More tolerance for noisy data
      assert(std::abs(result.second.y - 1.0f) < 0.5f);
    }
    
    // Test 5: Large X values (10^6 range) - tests adaptive delta for fp32 precision
    {
      std::vector<float> Xs = {1e6f, 2e6f, 3e6f, 4e6f, 5e6f};
      std::vector<float> Ys;
      for(float x : Xs)
        Ys.push_back(x + 1000.0f);  // y = x + 1000
      auto result = fitNonlinear<float, 1>(Xs, Ys, glm::vec1(0.0f), [](const glm::vec1& v, float x) { 
        return x + v.x; 
      });
      printf("Test 5 (large X ~10^6): fitted c=%.1f (expected 1000), error=%.6f\n", 
             result.second.x, result.first);
      assert(std::abs(result.second.x - 1000.0f) < 10.0f);  // Some tolerance due to fp32 precision
    }
    
    // Test 6: Large X values with linear fit
    {
      std::vector<float> Xs = {1e6f, 2e6f, 3e6f, 4e6f, 5e6f};
      std::vector<float> Ys;
      for(float x : Xs)
        Ys.push_back(0.5f * x + 1000.0f);  // y = 0.5x + 1000
      auto result = fitNonlinear<float, 2>(Xs, Ys, glm::vec2(1.0f, 0.0f), [](const glm::vec2& v, float x) { 
        return v.x * x + v.y; 
      });
      printf("Test 6 (large X linear): fitted y=%.6fx + %.1f (expected 0.5x + 1000), error=%.6f\n", 
             result.second.x, result.second.y, result.first);
      assert(std::abs(result.second.x - 0.5f) < 0.01f);
      assert(std::abs(result.second.y - 1000.0f) < 50.0f);
    }
    
    printf("All fitNonlinear tests passed!\n");
  }
};
static inline FitNonlinearTester g_fitNonlinearTest;
#endif

// Utility to hold time-proportional progress predictions and measurements after
// work completion.
struct ProgressTimeline
{
  using Clock = std::chrono::steady_clock;
  std::vector<float> estimate;
  std::vector<float> measure;
  Clock::time_point  start             = Clock::now();
  Clock::time_point  lastProgress      = Clock::now();
  float              completedEstimate = 0.0f;
  float              promisedEstimate  = 0.0f;
  ProgressTimeline(float estimatedTotalSeconds, Clock::time_point now = Clock::now())
      : start(now)
      , lastProgress(now)
      , promisedEstimate(estimatedTotalSeconds)
  {
  }
  void append(float partialPromisedEstimate, Clock::time_point now = Clock::now())
  {
    estimate.push_back(partialPromisedEstimate);
    measure.push_back(std::chrono::duration<float>(now - lastProgress).count());
    lastProgress = now;
    completedEstimate += partialPromisedEstimate;
  }
  void printBestFit() const { ::printBestFit<float>(estimate, measure); }
  void printRatios() const
  {
    float totalMeasured = std::accumulate(measure.begin(), measure.end(), 0.0f);
    printf("Measured subtask time ratios:");
    for(float measured : measure)
    {
      printf(" %.2f", measured * float(measure.size()) / totalMeasured);
    }
    printf("\n");
  }
};

// Progress may consists of multiple subtasks. Subtasks do not have a known
// absolute time estimate, and are instead estimated based on ratios of
// complexity between them. E.g. "the second task takes 2x as long as the
// first". When a subtask starts, a time estimate is required. Progress is then
// incrementally added. The estimate is linearly compensated for, so it just
// needs to be a linear relationship to the subtask complexity. The overall
// progress estimate is based on the remaining subtask ratios. Values for both
// ratios and fitting user complexities to time estimates are computed and
// printed at the end.
class TaskProgress
{
public:
  using Clock = typename ProgressTimeline::Clock;
  TaskProgress() {}
  ~TaskProgress()
  {
    if(started() && !m_cancel)
    {
      try
      {
        endSubtask(Clock::now());
        assert(currentSubtask() == m_subtaskRatios.size());
        assert(m_subtaskTimeline.value().completedEstimate
                       / m_subtaskTimeline.value().promisedEstimate
                   - 1.0f
               < 1e-6f);
        m_taskTimeline.value().printRatios();
      }
      catch(const std::exception& e)
      {
        // Being explicit about std::terminate for coverity. Shouldn't get here
        // unless there's a bug.
        fprintf(stderr, "Error in TaskProgress destructor: %s\n", e.what());
        assert(false);
        std::terminate();
      }
    }
  }
  TaskProgress(const TaskProgress& other)            = delete;
  TaskProgress& operator=(const TaskProgress& other) = delete;
  void defineSubtasks(std::vector<std::pair<std::string_view, float>>&& subtaskTimeRatios)
  {
    std::lock_guard lock(m_mutex);
    for(auto [name, ratio] : subtaskTimeRatios)
    {
      m_subtaskNames.emplace_back(name);
      m_subtaskRatios.push_back(ratio);
    }
  }
  void startSubtask(float estimateSeconds)
  {
    std::lock_guard lock(m_mutex);
    if(m_cancel)
      throw std::runtime_error("Task cancelled");  // DANGER: calling code better be exception safe
    auto now = Clock::now();
    if(started())
      endSubtask(now);
    else
    {
      m_taskTimeline.emplace(
          std::accumulate(m_subtaskRatios.begin(), m_subtaskRatios.end(), 0.0f), now);
    }
    m_subtaskTimeline.emplace(estimateSeconds, now);
    assert(currentSubtask() < m_subtaskRatios.size());  // not enough items given to estimateSubtaskComplexities()
  }
  void setSubtaskWorkName(std::string_view workName)
  {
    std::lock_guard lock(m_mutex);
    m_currentSubtaskWorkName.emplace(workName);
  }
  void makeProgress(float promisedEstimateSeconds)
  {
    std::lock_guard lock(m_mutex);
    if(m_cancel)
      throw std::runtime_error("Task cancelled");  // DANGER: calling code better be exception safe
    m_subtaskTimeline.value().append(promisedEstimateSeconds);
  }
  void progressTo(float completedEstimateSeconds, std::string_view nextWorkName)
  {
    makeProgress(completedEstimateSeconds);
    setSubtaskWorkName(nextWorkName);
  }
  void cancel()
  {
    std::lock_guard lock(m_mutex);
    m_cancel = true;
  }

  struct Progress
  {
    std::optional<std::string>     subtaskName;
    std::optional<std::string>     subtaskWorkName;
    float                          ratio;
    Clock::duration                elapsed;
    std::optional<Clock::duration> remaining;
    std::string                    currentWork() const
    {
      return subtaskName.value_or("not started")
             + (subtaskWorkName ? std::string(" -> ") + *subtaskWorkName : "");
    }
    std::string toString() const
    {
      auto durationString = [](auto duration) -> std::string {
        auto seconds = std::chrono::floor<std::chrono::seconds>(duration);
        return seconds > std::chrono::hours(1) ?
                   std::format("{:%H:%M:%S}", std::chrono::hh_mm_ss(seconds)) :
                   std::format("{:%M:%S}", std::chrono::hh_mm_ss(seconds));
      };
      return std::format("Progress: {:.1f}%, elapsed {}, remaining {} (current: {})",
                         ratio * 100.0f, durationString(elapsed),
                         remaining ? durationString(*remaining) : "??", currentWork());
    }
  };

  Progress progress() const
  {
    std::lock_guard lock(m_mutex);
    if(!started() || (currentSubtask() == 0 && m_subtaskTimeline.value().completedEstimate == 0.0f))
      return {std::nullopt, std::nullopt, 0.0f, Clock::duration(0), std::nullopt};

    auto  now             = Clock::now();
    float subtaskProgress = m_subtaskTimeline.value().completedEstimate
                            / m_subtaskTimeline.value().promisedEstimate;
    float progress = (m_taskTimeline.value().completedEstimate
                      + m_subtaskRatios.at(currentSubtask()) * subtaskProgress)
                     / m_taskTimeline.value().promisedEstimate;
    auto elapsed = now - m_taskTimeline.value().start;
    auto estimatedTotalDuration =
        (m_subtaskTimeline.value().lastProgress - m_taskTimeline.value().start) / progress;
    auto remaining =
        std::chrono::duration_cast<Clock::duration>(estimatedTotalDuration) - elapsed;
    remaining = std::max(remaining, Clock::duration(0));
    return {m_subtaskNames.at(currentSubtask()), m_currentSubtaskWorkName,
            progress, elapsed, remaining};
  }

private:
  bool   started() const { return m_taskTimeline.has_value(); }
  size_t currentSubtask() const
  {
    return m_taskTimeline.value().measure.size();
  }
  void endSubtask(Clock::time_point now)
  {
    assert(std::abs(m_subtaskTimeline.value().completedEstimate
                        / m_subtaskTimeline.value().promisedEstimate
                    - 1.0f)
           < 1e-6f);  // calls to makeProgress() didn't add up
    printf("Subtask '%s' completed. Fitting estimate time to measured:\n",
           m_subtaskNames.at(currentSubtask()).c_str());
    m_subtaskTimeline.value().printBestFit();
    m_taskTimeline.value().append(m_subtaskRatios.at(currentSubtask()), now);
  }

  mutable std::mutex m_mutex;

  std::vector<std::string> m_subtaskNames;
  std::vector<float>       m_subtaskRatios;

  std::optional<ProgressTimeline> m_taskTimeline;
  std::optional<ProgressTimeline> m_subtaskTimeline;
  std::optional<std::string>      m_currentSubtaskWorkName;

  // Complexity progress of the current subtask
  bool m_cancel = false;
};

// Utility class to poll and print progress in a background thread. It starts
// out printing frequently and slows down if progress doesn't change each time.
class ProgressPrinter
{
public:
  ProgressPrinter(TaskProgress& taskProgress)
      : m_taskProgress(taskProgress)
  {
    m_thread = std::thread([this]() {
      float                lastProgressRatio = m_taskProgress.progress().ratio;
      std::chrono::seconds sleepTime(1);
      for(;;)
      {
        std::unique_lock lock(m_mutex);
        m_condition.wait_for(lock, sleepTime, [this]() { return m_stop; });
        if(m_stop)
          return;
        auto progress = m_taskProgress.progress();
        std::print("{}\n", progress.toString());
        if(progress.ratio != lastProgressRatio)
          lastProgressRatio = progress.ratio;
        else
          sleepTime++;
      }
    });
  }
  ~ProgressPrinter()
  {
    {
      std::unique_lock lock(m_mutex);
      m_stop = true;
      m_condition.notify_all();
    }
    if(m_thread.joinable())
      m_thread.join();
  }
  ProgressPrinter(const ProgressPrinter& other)            = delete;
  ProgressPrinter& operator=(const ProgressPrinter& other) = delete;

private:
  std::thread             m_thread;
  std::mutex              m_mutex;
  std::condition_variable m_condition;
  bool                    m_stop = false;
  TaskProgress&           m_taskProgress;
};

#if 0
class TaskProgressTester
{
public:
  TaskProgressTester()
  {
    TaskProgress tp;
    tp.defineSubtasks({{"subtask 1", 1.0f}, {"subtask 2", 1.0f}, {"subtask 3", 1.0f}});
    tp.startSubtask(5);
    assert(tp.progress().ratio == 0.0f);
    tp.makeProgress(2);
    tp.makeProgress(3);
    assert(std::abs(tp.progress().ratio - 1.0f / 3.0f) < 0.0001f);
    tp.startSubtask(2);
    assert(std::abs(tp.progress().ratio - 1.0f / 3.0f) < 0.0001f);
    tp.makeProgress(2);
    assert(std::abs(tp.progress().ratio - 2.0f / 3.0f) < 0.0001f);
    tp.startSubtask(3);
    assert(std::abs(tp.progress().ratio - 2.0f / 3.0f) < 0.0001f);
    tp.makeProgress(1);
    tp.makeProgress(1);
    tp.makeProgress(1);
  }
};
static inline TaskProgressTester g_taskProgressTest;
#endif
