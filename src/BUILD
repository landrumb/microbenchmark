# cc_library(
#     name = "parlay",
#     hdrs = ["parlay/sequence.h",
#             "parlay/parallel.h",
#             "parlay/primitives.h",
#             "parlay/random.h",
#             "parlay/monoid.h",
#             "parlay/io.h",
#             "parlay/alloc.h",
#             "parlay/internal/get_time.h",
#             "parlay/slice.h",
#             "parlay/delayed_sequence.h"],
#     alwayslink = False,
#     includes = ["parlaylib/include"],
#     include_prefix = "parlaylib/include",
# )

cc_binary(
    name = "bench",
    srcs = ["bench.cpp", "threadlocal.h"],
    deps = ["@benchmark//:benchmark",
            "@parlaylib//parlay:sequence",
            "@parlaylib//parlay:parallel",
            "@parlaylib//parlay:primitives",
            "@parlaylib//parlay:random",
            "@parlaylib//parlay:monoid",],
)