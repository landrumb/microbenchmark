#include <benchmark/benchmark.h>
#include "threadlocal.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/random.h"

#include <atomic>


static void BM_Accumulator128(benchmark::State& state) {
  threadlocal::accumulator<int, 128, 192> acc;

  for (auto _ : state) {
    parlay::parallel_for(0, 1000000, [&] (size_t i) {
    acc.add(i % 100);
  });
  }
}
BENCHMARK(BM_Accumulator128);

static void BM_Accumulator64(benchmark::State& state) {
  threadlocal::accumulator<int, 64, 192> acc;

  for (auto _ : state) {
    parlay::parallel_for(0, 1000000, [&] (size_t i) {
    acc.add(i % 100);
  });
  }
}
BENCHMARK(BM_Accumulator64);

static void BM_Accumulator32(benchmark::State& state) {
  threadlocal::accumulator<int, 32, 192> acc;

    for (auto _ : state) {
        parlay::parallel_for(0, 1000000, [&] (size_t i) {
        acc.add(i % 100);
    });
    }
}
BENCHMARK(BM_Accumulator32);

static void BM_Accumulator16(benchmark::State& state) {
  threadlocal::accumulator<int, 16, 192> acc;

    for (auto _ : state) {
        parlay::parallel_for(0, 1000000, [&] (size_t i) {
        acc.add(i % 100);
    });
    }
}
BENCHMARK(BM_Accumulator16);

static void BM_SeqSum(benchmark::State& state) {
    int sum = 0;
  for (auto _ : state) {
    for (int i = 0; i < 1000000; i++) {
        sum += i % 100;
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
    sum += i % 100;
  });
  }
}
BENCHMARK(BM_AtomicSum);

BENCHMARK_MAIN();