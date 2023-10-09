#include <benchmark/benchmark.h>
#include "threadlocal.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/random.h"

#include <atomic>

int numbers[1000000];

template <size_t pad_bytes>
static void BM_Accumulator(benchmark::State& state) {
  threadlocal::accumulator<int, pad_bytes, 192> acc;

  for (auto _ : state) {
    parlay::parallel_for(0, 1000000, [&] (size_t i) {
    acc.add(numbers[i]);
  });
  }
}
BENCHMARK(BM_Accumulator<4>);
BENCHMARK(BM_Accumulator<8>);
BENCHMARK(BM_Accumulator<16>);
BENCHMARK(BM_Accumulator<32>);
BENCHMARK(BM_Accumulator<64>);
BENCHMARK(BM_Accumulator<128>);
BENCHMARK(BM_Accumulator<256>);
BENCHMARK(BM_Accumulator<512>);
BENCHMARK(BM_Accumulator<1024>);


static void BM_SeqSum(benchmark::State& state) {
    int sum = 0;
  for (auto _ : state) {
    for (int i = 0; i < 1000000; i++) {
        sum += numbers[i];
    }
  }
  if (sum == 0) {
      std::cout << "sum is zero" << std::endl;
  }
}
BENCHMARK(BM_SeqSum);

static void BM_AtomicSum(benchmark::State& state) {
    std::atomic<int> sum = 0;
  for (auto _ : state) {
    parlay::parallel_for(0, 1000000, [&] (size_t i) {
    sum += numbers[i];
  });
  }
}
BENCHMARK(BM_AtomicSum);

int main(int argc, char** argv) {
    parlay::random r;
    for (int i = 0; i < 1000000; i++) {
        numbers[i] = r.ith_rand(i) % 100;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}