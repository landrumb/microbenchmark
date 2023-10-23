cc_binary(
    name = "bench",
    srcs = ["bench.cpp", "threadlocal.h"],
    deps = ["@benchmark//:benchmark",
            "@parlaylib//parlay:sequence",
            "@parlaylib//parlay:parallel",
            "@parlaylib//parlay:primitives",
            "@parlaylib//parlay:random",
            "@parlaylib//parlay:monoid",
            "//src/utils:NSGDist"],
)