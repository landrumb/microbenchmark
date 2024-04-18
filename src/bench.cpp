#include <benchmark/benchmark.h>
#include "threadlocal.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/random.h"
#include "utils/NSGDist.h"
#include "utils/avx_dist.h"

#include <atomic>
#include <random>
#include <math.h>

// int numbers[1000000000];

#define DIM 100
#define N 10000000

// constexpr size_t N = 1000000000;
// constexpr size_t DIM = 100;

parlay::sequence<int> numbers;

alignas(64) float floats[N];

const int N_THREADS = 192;

template <size_t pad_bytes>
static void BM_Accumulator(benchmark::State& state) {
  threadlocal::accumulator<int, pad_bytes, N_THREADS> acc;

  for (auto _ : state) {
    parlay::parallel_for(0, 1000000000, [&] (size_t i) {
    acc.add(numbers[i]);
  });
  }
}
// BENCHMARK(BM_Accumulator<4>);
// BENCHMARK(BM_Accumulator<8>);
// BENCHMARK(BM_Accumulator<16>);
// BENCHMARK(BM_Accumulator<32>);
// BENCHMARK(BM_Accumulator<64>);
// BENCHMARK(BM_Accumulator<128>);
// BENCHMARK(BM_Accumulator<256>);
// BENCHMARK(BM_Accumulator<512>);
// BENCHMARK(BM_Accumulator<1024>);

template <size_t pad_bytes>
static void BM_Accumulator_Blocks(benchmark::State& state) {
  threadlocal::accumulator<int, pad_bytes, N_THREADS> acc;

  for (auto _ : state) {
    // size_t N = numbers.size();
    size_t block_size = 1000;
        size_t num_blocks = N / block_size;
    parlay::parallel_for(0, num_blocks, [&] (size_t i) {
      size_t ctr = 0;
      size_t offset = i*block_size;
      for (size_t j=0; j<block_size; ++j) {
        ctr += numbers[offset + j];
      }
      acc.add(ctr);
    }, 1);
  }
}
// BENCHMARK(BM_Accumulator_Blocks<4>);
// BENCHMARK(BM_Accumulator_Blocks<8>);
// BENCHMARK(BM_Accumulator_Blocks<16>);
// BENCHMARK(BM_Accumulator_Blocks<32>);
// BENCHMARK(BM_Accumulator_Blocks<64>);
// BENCHMARK(BM_Accumulator_Blocks<128>);
// BENCHMARK(BM_Accumulator_Blocks<256>);
// BENCHMARK(BM_Accumulator_Blocks<512>);
// BENCHMARK(BM_Accumulator_Blocks<1024>);


static void BM_SeqSum(benchmark::State& state) {
    int sum = 0;
  for (auto _ : state) {
    for (int i = 0; i < 1000000000; i++) {
        sum += numbers[i];
    }
  }
  if (sum == 0) {
      std::cout << "sum is zero" << std::endl;
  }
}
// BENCHMARK(BM_SeqSum);

static void BM_AtomicSum(benchmark::State& state) {
    std::atomic<int> sum = 0;
  for (auto _ : state) {
    parlay::parallel_for(0, 1000000000, [&] (size_t i) {
    sum += numbers[i];
  });
  }
}
// BENCHMARK(BM_AtomicSum);

inline float* select_point(std::mt19937& gen) {
    return floats + (gen() % ((N - DIM) / 64)) * 64;
}


static void BM_FloatNSGDist(benchmark::State& state) {
  std::random_device rd;
  std::mt19937 gen(rd());
  efanna2e::DistanceL2 distfunc;
  volatile size_t dim = DIM;
  for (auto _ : state) {
    float *p = select_point(gen);
    float *q = select_point(gen);
    float x = distfunc.compare(p, q, dim);
    if (x == 12345678) {
        std::cout << "x is zero" << std::endl;
    }
  }
}
BENCHMARK(BM_FloatNSGDist)->MinWarmUpTime(3)->MinTime(3);

float distance(const float *p, const float *q, size_t dim) {
    float sum = 0;
    for (size_t i=0; i<dim; ++i) {
        float diff = p[i] - q[i];
        sum += diff * diff;
    }
    return sum;
}

template<size_t dim>
float distance_const(const float *p, const float *q) {
    float sum = 0;
    for (size_t i=0; i<dim; ++i) {
        float diff = p[i] - q[i];
        sum += diff * diff;
    }
    return sum;
}

float distance_sqrt(const float *p, const float *q, size_t dim) {
    float sum = 0;
    for (size_t i=0; i<dim; ++i) {
        float diff = p[i] - q[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

template<size_t dim>
float distance_sqrt_const(const float *p, const float *q) {
    float sum = 0;
    for (size_t i=0; i<dim; ++i) {
        float diff = p[i] - q[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

static void BM_FloatVarLoop(benchmark::State& state) {
  std::random_device rd;
  std::mt19937 gen(rd());
  
  volatile size_t dim = 100;
  for (auto _ : state) {
    float *p = select_point(gen);
    float *q = select_point(gen);
    float x = distance(p, q, dim);
    if (x == 12345678) {
        std::cout << "x is zero" << std::endl;
    }
  }
}
BENCHMARK(BM_FloatVarLoop)->MinWarmUpTime(3)->MinTime(3);

static void BM_FloatSqrtVarLoop(benchmark::State& state) {
  std::random_device rd;
  std::mt19937 gen(rd());
  
  volatile size_t dim = DIM;
  for (auto _ : state) {
    float *p = select_point(gen);
    float *q = select_point(gen);
    float x = distance_sqrt(p, q, dim);
    if (x == 12345678) {
        std::cout << "x is zero" << std::endl;
    }
  }
}
BENCHMARK(BM_FloatSqrtVarLoop)->MinWarmUpTime(3)->MinTime(3);

static void BM_FloatSqrtConstLoop(benchmark::State& state) {
  std::random_device rd;
  std::mt19937 gen(rd());

  for (auto _ : state) {
    float *p = select_point(gen);
    float *q = select_point(gen);
    float x = distance_sqrt_const<DIM>(p, q);
    if (x == 12345678) {
        std::cout << "x is zero" << std::endl;
    }
  }
}
BENCHMARK(BM_FloatSqrtConstLoop)->MinWarmUpTime(3)->MinTime(3);

static void BM_FloatAVX512(benchmark::State& state) {
  std::random_device rd;
  std::mt19937 gen(rd());
  
  for (auto _ : state) {
    float *p = select_point(gen);
    float *q = select_point(gen);
    float x = sq_euclidean_aligned_100(p, q);
    if (x == 12345678) {
        std::cout << "x is zero" << std::endl;
    }
  }
}
BENCHMARK(BM_FloatAVX512)->MinWarmUpTime(3)->MinTime(3);

static void BM_FloatAVX512Pipeline2(benchmark::State& state) {
  std::random_device rd;
  std::mt19937 gen(rd());
  
  for (auto _ : state) {
    float *p = select_point(gen);
    float *q = select_point(gen);
    float x = sq_euclidean_aligned_pipeline2(p, q);
    if (x == 12345678) {
        std::cout << "x is zero" << std::endl;
    }
  }
}
BENCHMARK(BM_FloatAVX512Pipeline2)->MinWarmUpTime(3)->MinTime(3);

static void BM_JoinSortedArrays(benchmark::State& state) {
  std::random_device rd;
  std::mt19937 gen(rd());
  size_t n = 20000;
  parlay::sequence<int> a = parlay::sequence<int>::uninitialized(n);
  parlay::sequence<int> b = parlay::sequence<int>::uninitialized(n);
  

  for (auto _ : state) {
    int *p = numbers.begin() + (gen() % (N - 20'000));
    int *q = numbers.begin() + (gen() % (N - 20'000));
    for (size_t i=0; i<n; i++) {
        a[i] = p[i];
        b[i] = q[i];
    }
    // std::cout<<a[0]<<std::endl;

    parlay::sort_inplace(a);
    parlay::sort_inplace(b);

    

  }
}
// BENCHMARK(BM_JoinSortedArrays)->MinWarmUpTime(3)->MinTime(3);

// static void BM_FloatConstLoop(benchmark::State& state) {
//   std::random_device rd;
//   std::mt19937 gen(rd());

//   for (auto _ : state) {
//     float *p = floats + (gen() % (N - 192));
//     float *q = floats + (gen() % (N - 192));
//     float x = distance_const<DIM>(p, q);
//     if (x == 12345678) {
//         std::cout << "x is zero" << std::endl;
//     }
//   }
// }
// BENCHMARK(BM_FloatConstLoop)->MinWarmUpTime(3)->MinTime(3);


int main(int argc, char** argv) {
    numbers = parlay::sequence<int>::uninitialized(N);

    // parlay::random r;
    // fill numbers with normally distributed random numbers
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<float> d(0, 1);
    
    parlay::parallel_for(0, N, [&] (int i) {
        numbers[i] = static_cast<int>(gen()) % 10'000'000;
    });

    parlay::parallel_for(0, N, [&] (int i) {
        floats[i] = d(gen);
    });

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}