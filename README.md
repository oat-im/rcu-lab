# RCU-Lab  
**A NUMA-aware microbenchmark suite for real-world RCU analysis**  
*"Because `shared_mutex` is for amateurs."*

## What Is This?

RCU-Lab is a high-performance benchmark suite built to expose the **true behavior** of Read-Copy-Update (RCU) under realistic, high-concurrency, NUMA-aware workloads.

Forget toy loops with `std::mutex`. This exposes:
- ✅ Real-world **cache coherency hell**
- ✅ **NUMA-local vs remote** access impact
- ✅ **P99.9 latency** under aggressive writers
- ✅ Grace period visibility
- ✅ **Microsecond-level** contention analysis

It's built for people who care about:
- Kernel scalability
- Trading system latency
- Lock-free data structures
- Multisocket memory bottlenecks

## 🔧 Features

### 🔍 Precision Measurement
- Per-core read latency tracking in **nanoseconds**
- **CPU pinning** to avoid scheduler randomness
- NUMA-aware memory allocation per-core
- `alignas(128)` to kill false sharing dead

### 💣 Real Contention Scenarios
- Writers updating at **1K → 100K ops/sec**
- NUMA-aware writer partitioning
- Readers focused on **local or remote slots**
- **Cross-NUMA latency explosions** fully visible

### Actual Analysis Tools
- Full latency histograms: **P50, P99, P99.9, P99.99**
- Optional **CSV dump** for plotting
- Reader-side cache miss correlation via `last_update` timestamp diffing
- **Grace period aging** (optional)

## Quick Start

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install liburcu-dev libnuma-dev cmake build-essential

# Fedora/RHEL
sudo dnf install userspace-rcu-devel numactl-devel cmake gcc-c++

# macOS (builds, but NUMA is a joke)
brew install userspace-rcu cmake
```

## Build & Run

```bash
git clone https://github.com/oat-im/rcu-lab.git
cd rcu-lab
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run with sudo for CPU affinity
```bash
sudo ./rcu-lab
```

It will:
 - Detect NUMA topology
 - Pin threads to physical cores
 - Launch readers and writers across sockets
 - Run multiple write rates (1K → 100K/sec)
 - Output latency stats per core

## Output Sample

```
=== Per-Core Read Latency Analysis ===
Core | NUMA | Reads    | Median(ns) | P99(ns) | P99.9(ns) | Cache Misses
-----|------|----------|------------|---------|-----------|-------------
  42 |    1 | 80528025 |         34 |     498 |       958 |     122119
  44 |    1 |149080063 |         30 |     491 |       941 |    1562720
  55 |    1 | 16445651 |        488 |    4093 |      6412 |     150441
```

### Interpreting the Data

 - Median: Best-case L1-local access
 - P99/P99.9: Tail latency under contention / remote NUMA / cache invalidation
 - Cache Misses: Proxy for invalidated lines from recent writes

##  What You'll Learn
 1. RCU reads are free - until NUMA or frequent writers show up.
 2. NUMA locality matters. Cross-socket reads explode latency.
 3. Frequent writes torch P99.9 latency, even if P50 stays fine.
 4. Grace period cleanup can lag under read pressure.


## Advanced Usage

### Custom Access Patterns

Want readers to hammer remote slots?

```cpp
lab.set_access_pattern(RCULab::AccessPattern::REMOTE_HEAVY);
```

### Plug In Your Own Data

```cpp
struct MyThing {
    struct rcu_head rcu_head;
    int field1;
    char padding[128 - sizeof(rcu_head) - sizeof(int)];
};
```

Keep it aligned. Keep it realistic.

## Visualizing Output

Optional CSV export can be added for:
 - Tail latency histograms
 - Per-core NUMA effects
 - Cache miss correlation

Then use Python, R, Excel, whatever. Just don't pretend average latency means anything.

## Benchmarking Philosophy

RCU-Lab exists because too many people:
 - Benchmark wrong things (avg latency, mutex loops)
 - Run on unstable setups (no CPU pinning, no isolation)
 - Don't account for NUMA, cache coherence, or reality

This tool fixes that. You get real answers, or you don't ship.


## Contributing

Yes, you can send PRs. Just follow these rules:
 - No std::cout in hot paths
 - No sleep() to simulate load
 - No half-assed benchmarks

If you can write a low-latency allocator, port to Rust or analyze grace periods - great.

## Citation

```
@software{rcu-lab,
  title = {RCU-Lab: A NUMA-aware microbenchmark suite for real-world RCU analysis},
  author = {Eric Christian},
  year = {2025},
  url = {https://github.com/oat-im/rcu-lab}
}
```

## License

MIT - because good tooling should be free and fast.

## Acknowledgments
 - liburcu - production-grade RCU for userspace
 - Paul McKenney - RCU godfather
 - All the engineers who don't accept lock contention as "inevitable"

"Stop benchmarking like a CS student. Start benchmarking like a kernel engineer."
