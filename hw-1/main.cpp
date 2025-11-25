#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <map>

using namespace std::chrono;

const uint32_t ARRAY_READS_COUNT = 1'000'000;
const uint32_t WARMUP_READS_COUNT = 1'000'000;
const uint32_t BATCHES_COUNT = 1;
const uint32_t PAGE_SIZE = (1 << 16); // 64 KB
const uint32_t TIME_DIV_FACTOR = 10'000;
bool debug = false;
std::mt19937 gen(239);

using timetype = std::chrono::nanoseconds;

uint32_t *array = nullptr;
uint32_t arrayLen = 0;

void rassert(bool expr, uint32_t id) {
  if (!expr) {
    std::cerr << "Assertion failed: " << id << std::endl;
    std::exit(1);
  }
}

std::ostream &log() {
  if (debug) {
    return std::cout;
  }
  static std::ostream noopStream = std::ostream(nullptr);
  return noopStream;
}

std::string bytesToString(uint32_t bytes) {
  std::vector<std::string> parts;
  if (bytes >= (1 << 20)) {
    parts.push_back(std::to_string(bytes / (1 << 20)) + "MB");
    bytes = bytes % (1 << 20);
  }
  if (bytes >= (1 << 10)) {
    parts.push_back(std::to_string(bytes / (1 << 10)) + "KB");
    bytes = bytes % (1 << 10);
  }
  if (bytes > 0) {
    parts.push_back(std::to_string(bytes) + "B");
  }

  std::string result;
  size_t i = 0;
  for (const auto &p : parts) {
    result += p;
    i++;
    if (i < parts.size())
      result += " ";
  }
  return result;
}

uint32_t calcLog2(uint32_t n) {
  rassert(n != 0, 2);
  uint32_t log = 0;
  while (n >>= 1)
    ++log;
  return log;
}

void prettyPrint(uint32_t maxAssoc, uint32_t minStride, uint32_t stridePow,
                 const std::vector<std::vector<timetype>> &times,
                 const std::vector<std::vector<bool>> &jumps, uint32_t width,
                 uint32_t timeFactor) {
  // printing results
  log() << std::setw(width) << "s/e";
  for (uint32_t p = 0; p < stridePow; p++) {
    uint32_t bytes = (1 << p) * minStride;
    log() << std::setw(width) << bytesToString(bytes);
  }
  log() << std::endl;

  for (uint32_t s = 1; s < maxAssoc; s++) {
    log() << std::setw(width) << s;
    for (uint32_t p = 0; p < stridePow; p++) {
      auto time = times[s][p].count() / timeFactor;
      std::string timeWithJump =
          (jumps[s][p] ? "[+]" : "") + std::to_string(time);
      log() << std::setw(width) << timeWithJump;
    }
    log() << std::endl;
  }
}

void fillDirectIndexes(uint32_t stride, uint32_t elems) {
  // direct indexes
  for (uint32_t i = 0; i <= stride * (elems - 1); i += stride) {
    if (i == stride * (elems - 1)) {
      // loop back
      array[i] = 0;
      break;
    } else {
      array[i] = (i + stride);
    }
  }
}

void fillReverseIndexes(uint32_t stride, uint32_t elems) {
  // reverse indexes
  // 0  i1 i2 i3 ... in
  // in 0  i1 i2 ... in-1
  for (uint32_t i = stride * (elems - 1), cnt = 0; cnt < elems;
       i -= stride, cnt++) {
    if (i == 0) {
      // loop back
      array[i] = stride * (elems - 1);
      break;
    } else {
      array[i] = (i - stride);
    }
  }
}

void fillShuffledIndexes(uint32_t stride, uint32_t elems) {
  // shuffled indexes
  std::vector<uint32_t> indexes(elems);
  for (uint32_t i = 0; i < elems; i++) {
    indexes[i] = i;
  }

  // simple shuffle
  std::shuffle(indexes.begin(), indexes.end(), gen);

  for (uint32_t i = 0; i < elems; i++) {
    if (i == elems - 1) {
      array[indexes[i] * stride] = indexes[0] * stride;
    } else {
      array[indexes[i] * stride] = indexes[i + 1] * stride;
    }
  }
}

timetype timeOfArrayRead(uint32_t stride, uint32_t elems, uint32_t readsCount,
                         uint32_t warmupReadsCount, uint32_t batchesCount) {
  volatile uint32_t sink = 0; // prevents optimization
  timetype diff = timetype::zero();
  uint32_t iterationsPerBatch = readsCount;
  for (uint32_t batch = 0; batch < batchesCount; ++batch) {
    // fill indexes
    fillShuffledIndexes(stride, elems);

    // some reads to warm up cache
    for (uint32_t i = 0, idx = 0; i < warmupReadsCount; i++) {
      idx = array[idx];
      sink = idx;
    }

    // measure
    steady_clock::time_point start = steady_clock::now();
    for (uint32_t i = 0, idx = 0; i < iterationsPerBatch; i++) {
      idx = array[idx];
      sink = idx;
    }
    steady_clock::time_point end = steady_clock::now();
    diff += duration_cast<timetype>(end - start);
  }
  diff /= batchesCount;
  return diff;
}

bool isSufficientIncrease(timetype currentTime, timetype prevTime,
                          double fraction) {
  auto delta = currentTime - prevTime;
  auto div = static_cast<double>(delta.count()) / prevTime.count();
  return delta > timetype::zero() && div > fraction;
}

bool isSuffientDiff(timetype currentTime, timetype prevTime, double fraction) {
  double delta = std::abs((currentTime - prevTime).count());
  auto div = delta / prevTime.count();
  return div > fraction;
}

timetype getAveragedDelta(const std::vector<std::vector<timetype>> &times,
                          uint32_t fromElem, uint32_t toElem,
                          uint32_t stridePow) {
  rassert(fromElem < toElem, 3);
  timetype total = timetype::zero();
  for (uint32_t elem = fromElem; elem < toElem; elem++) {
    total += times[elem][stridePow];
  }
  return total / (toElem - fromElem);
}

std::tuple<uint32_t, uint32_t, bool>
capacityAndAssociativity(uint32_t maxAssoc, uint32_t minStride,
                         uint32_t maxStride) {
  uint32_t stride = minStride / sizeof(uint32_t); // elements
  uint32_t stridePow = 0;
  // matrix elems x strides
  std::vector<std::vector<timetype>> times(
      maxAssoc, std::vector<timetype>(maxStride, timetype{}));
  std::vector<std::vector<bool>> jumps(maxAssoc,
                                       std::vector<bool>(maxStride, false));

  // calculate averaged execution times for each (elems, stride)
  while (stride * sizeof(uint32_t) <= maxStride) {
    uint32_t elems = 1;
    // calculate time jumps
    while (elems < maxAssoc) {
      timetype currentTime = timeOfArrayRead(stride, elems, ARRAY_READS_COUNT,
                                             WARMUP_READS_COUNT, BATCHES_COUNT);
      times[elems][stridePow] = currentTime;
      elems++;
    }
    stride *= 2;
    stridePow++;
  }

  // calculate cumulative jumps
  uint32_t span = 4;
  double fraction = 0.3;
  for (uint32_t p = 0; p < stridePow; p++) {
    for (uint32_t elem = span + 1; elem + span <= maxAssoc; elem++) {
      // find a jump between averaged time[(elem - span)..elem) and
      // time[elem..(elem + span)) for some fixed p
      timetype prevDelta = getAveragedDelta(times, elem - span, elem, p);
      timetype currentDelta = getAveragedDelta(times, elem, elem + span, p);
      bool isJump = isSufficientIncrease(currentDelta, prevDelta, fraction);
      jumps[elem - 1][p] = isJump; // write at elem - 1, so that we see jumps at
                                   // elems 2^t and not 2^t + 1 indexes
    }
  }

  // since we only target the cpu L1 cache,
  // for its calculation we need to find the first movement sequence:
  // jump at
  // 2^(t+k) -> 2^(t+k-1) -> ... ->    2^t    ->   2^t      (no jump at the end)
  //   s          2 * s            2^(k-1) * s   2^k * s
  // then the cache associativity will be 2^t and the cache size will be
  // (2^(k-1) * s) * associativity
  uint32_t cacheAssociativity = 0;
  uint32_t cacheSize = 0;
  bool found = false;

  for (uint32_t p = 0; p < stridePow && !found; p++) {
    for (uint32_t elem = 1; elem < maxAssoc && !found; elem *= 2) {
      if (!jumps[elem][p])
        continue;

      uint32_t assoc = elem;
      uint32_t stride = p;
      while (assoc >= 1 && stride + 2 < stridePow) {
        bool jumpMoved = jumps[assoc / 2][stride + 1];
        bool jumpThenStayed = jumps[assoc / 2][stride + 2];

        if (!jumpMoved)
          break;

        if (jumpMoved && jumpThenStayed) {
          // found the end of the sequence
          cacheAssociativity = assoc / 2;
          uint32_t strideShift = minStride / sizeof(uint32_t);
          cacheSize = (1 << ((stride + 1) + strideShift)) * cacheAssociativity;
          found = true;
          break;
        }

        assoc /= 2;
        stride++;
      }
    }
  }

  // print a table
  prettyPrint(maxAssoc, minStride, stridePow, times, jumps, 8, TIME_DIV_FACTOR);

  return {cacheAssociativity, cacheSize, found};
}

timetype averageTime(uint32_t higherStride, uint32_t lowerStride,
                     uint32_t maxSpots) {
  uint32_t stride = higherStride + lowerStride;
  timetype totalTime = timetype::zero();
  for (uint32_t spots = 1; spots <= maxSpots; spots++) {
    totalTime += timeOfArrayRead(stride, spots, ARRAY_READS_COUNT,
                                 WARMUP_READS_COUNT, BATCHES_COUNT);
  }
  return totalTime / maxSpots;
}

uint32_t lineSize(uint32_t minStride, uint32_t maxStride, uint32_t cacheSize) {
  log() << "Min stride: " << bytesToString(minStride) << std::endl;
  log() << "Max stride: " << bytesToString(maxStride) << std::endl;
  uint32_t maxSpots = 1024;
  double fraction = 0.11;
  enum Trend : int { DEC = 0, STABLE = 1, INC = 2 };
  std::map<uint32_t, Trend> trends;

  for (uint32_t higherStride = minStride; higherStride <= maxStride;
       higherStride *= 2) {
    timetype higherTime =
        averageTime(higherStride / sizeof(uint32_t), 0, maxSpots);
    log() << "Higher stride: " << bytesToString(higherStride)
          << ", time: " << higherTime.count() / TIME_DIV_FACTOR << std::endl;

    uint32_t increases = 0, decreases = 0;
    // Start from 4 values below the higher stride
    for (uint32_t lowerStride = higherStride / 16; lowerStride < higherStride;
         lowerStride *= 2) {
      timetype combinedTime =
          averageTime(higherStride / sizeof(uint32_t),
                      lowerStride / sizeof(uint32_t), maxSpots);
      log() << "  Lower stride: " << bytesToString(lowerStride)
            << ", time: " << combinedTime.count() / TIME_DIV_FACTOR
            << std::endl;

      if (isSuffientDiff(combinedTime, higherTime, fraction)) {
        if (combinedTime > higherTime) {
          increases++;
        } else {
          decreases++;
        }
      }
    }
    log() << "Increases: " << increases << ", decreases: " << decreases
          << std::endl;

    trends[higherStride] = Trend::STABLE;
    if (increases > decreases) {
      trends[higherStride] = Trend::INC;
    } else if (increases < decreases) {
      trends[higherStride] = Trend::DEC;
    }
  }

  uint32_t lineSize = 0;
  bool candidateFound = false;

  // trend sequence from which we can determine cache line looks like this:
  // [INC|STABLE]* [DEC|STABLE]*
  for (auto [stride, trend] : trends) {
    log() << "Stride: " << bytesToString(stride) << ", trend: " << trend
          << std::endl;

    if (trend == Trend::DEC && !candidateFound) {
      // found first decrease
      lineSize = stride / 2; // previous value
      candidateFound = true;
    } else if (trend == Trend::INC && candidateFound) {
      // found increase after decrease -> we cannot determine the cache line
      return 0;
    }
  }

  if (!candidateFound) {
    // if no decrease found, assume the line size is the maximum we tested
    lineSize = maxStride;
  }

  return lineSize;
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::string(argv[1]) == "--debug") {
    debug = true;
  }

  uint32_t maxMemory = (1 << 30); // 1 GB
  uint32_t maxAssoc = 32;
  uint32_t maxStride = (1 << 23); // 8 MB
  uint32_t minStride = 16;

  rassert(maxAssoc * maxStride <= maxMemory, 1);

  array = static_cast<uint32_t *>(aligned_alloc(PAGE_SIZE, maxMemory));
  arrayLen = maxMemory / sizeof(uint32_t);

  std::cout << "array: " << array << std::endl;
  std::cout << "len: " << arrayLen << std::endl;

  auto [cacheAssociativity, cacheSize, detected] =
      capacityAndAssociativity(maxAssoc + 1, minStride, maxStride);

  if (detected) {
    std::cout << "Detected L1 cache associativity: " << cacheAssociativity
              << std::endl;
    std::cout << "Detected L1 cache size: " << bytesToString(cacheSize)
              << std::endl;
  } else {
    std::cout << "Could not detect L1 cache associativity and size."
              << std::endl;
  }

  auto lineSizeBytes = lineSize(minStride, 512, cacheSize);
  if (lineSizeBytes != 0) {
    std::cout << "Detected cache line size: " << bytesToString(lineSizeBytes)
              << std::endl;
  } else {
    std::cout << "Could not detect cache line size." << std::endl;
  }

  free(array);
  return 0;
}