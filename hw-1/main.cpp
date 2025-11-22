#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std::chrono;

const uint32_t ARRAY_READS_COUNT = 1'000'000;
const uint32_t WARMUP_READS_COUNT = 5000;
const uint32_t BATCHES_COUNT = 5;
const uint32_t PAGE_SIZE = (1 << 14); // 16 KB
const uint32_t TIME_DIV_FACTOR = 10'000;
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

uint32_t log2(uint32_t n) {
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
  std::cout << std::setw(width) << "s/e";
  for (uint32_t p = 0; p < stridePow; p++) {
    uint32_t bytes = (1 << p) * minStride;
    std::cout << std::setw(width) << bytesToString(bytes);
  }
  std::cout << std::endl;

  for (uint32_t s = 1; s < maxAssoc; s++) {
    std::cout << std::setw(width) << s;
    for (uint32_t p = 0; p < stridePow; p++) {
      auto time = times[s][p].count() / timeFactor;
      std::string timeWithJump =
          (jumps[s][p] ? "[+]" : "") + std::to_string(time);
      std::cout << std::setw(width) << timeWithJump;
    }
    std::cout << std::endl;
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

std::tuple<uint32_t, bool> lineSize(uint32_t minStride, uint32_t cacheSize) {
  uint32_t minStridePow = log2(minStride / sizeof(uint32_t));
  uint32_t maxStridePow = log2(cacheSize / sizeof(uint32_t));
  uint32_t offsetFactor = 4;

  for (uint32_t p = minStridePow; p < maxStridePow; p++) {
    std::vector<int> trend; // 1 - increasing, -1 - decreasing, 0 - stable

    uint32_t higherStride = (1 << p);
    uint32_t baseSpots = cacheSize / (higherStride * sizeof(uint32_t));
    timetype baseTime =
        timeOfArrayRead(higherStride, baseSpots, ARRAY_READS_COUNT,
                        WARMUP_READS_COUNT, BATCHES_COUNT);

    uint32_t baseLeftSpots = baseSpots - baseSpots / offsetFactor;
    timetype baseLeftTime =
        timeOfArrayRead(higherStride, baseLeftSpots, ARRAY_READS_COUNT,
                        WARMUP_READS_COUNT, BATCHES_COUNT);
    uint32_t baseRightSpots = baseSpots + baseSpots / offsetFactor;
    timetype baseRightTime =
        timeOfArrayRead(higherStride, baseRightSpots, ARRAY_READS_COUNT,
                        WARMUP_READS_COUNT, BATCHES_COUNT);

    std::cout << "Base stride: "
              << bytesToString(higherStride * sizeof(uint32_t))
              << ", Spots: " << baseSpots << ", AMAT: "
              << static_cast<double>(baseTime.count()) / ARRAY_READS_COUNT
              << std::endl;
    std::cout << "  Left AMAT: "
              << static_cast<double>(baseLeftTime.count()) / ARRAY_READS_COUNT
              << " (spots: " << baseLeftSpots << ")"
              << ", Right AMAT: "
              << static_cast<double>(baseRightTime.count()) / ARRAY_READS_COUNT
              << " (spots: " << baseRightSpots << ")" << std::endl;

    uint32_t jumpsCount = 0;
    for (uint32_t lp = 1; lp < p; lp++) { // lower stride pow
      // find the elems count where the jump occurs for stride (p + lp)
      // save
      uint32_t lowerStride = (1 << lp);
      uint32_t spotsWhenHigherStrideIsLessThanLineSize =
          cacheSize / ((higherStride + lowerStride) * sizeof(uint32_t));
      timetype currentTime = timeOfArrayRead(
          higherStride + lowerStride, spotsWhenHigherStrideIsLessThanLineSize,
          ARRAY_READS_COUNT, WARMUP_READS_COUNT, BATCHES_COUNT);

      std::cout << "Lower stride: "
                << bytesToString(lowerStride * sizeof(uint32_t))
                << ", Spots: " << spotsWhenHigherStrideIsLessThanLineSize
                << ", AMAT: "
                << static_cast<double>(currentTime.count()) / ARRAY_READS_COUNT
                << std::endl;

      uint32_t leftSpots =
          spotsWhenHigherStrideIsLessThanLineSize -
          spotsWhenHigherStrideIsLessThanLineSize / offsetFactor;
      timetype leftTime =
          timeOfArrayRead(higherStride + lowerStride, leftSpots,
                          ARRAY_READS_COUNT, WARMUP_READS_COUNT, BATCHES_COUNT);
      uint32_t rightSpots = baseSpots + baseSpots / offsetFactor;
      // spotsWhenHigherStrideIsLessThanLineSize +
      // spotsWhenHigherStrideIsLessThanLineSize / offsetFactor;
      timetype rightTime =
          timeOfArrayRead(higherStride + lowerStride, rightSpots,
                          ARRAY_READS_COUNT, WARMUP_READS_COUNT, BATCHES_COUNT);
      std::cout << "  Left AMAT: "
                << static_cast<double>(leftTime.count()) / ARRAY_READS_COUNT
                << " (spots: " << leftSpots << ")"
                << ", Right AMAT: "
                << static_cast<double>(rightTime.count()) / ARRAY_READS_COUNT
                << " (spots: " << rightSpots << ")";

      bool isJump = isSufficientIncrease(rightTime, leftTime, 0.3);
      std::cout << ", isJump: " << isJump << std::endl;
      jumpsCount += isJump;
    }

    // at least half of the pairs (p + lp) do not have
    // jumps, then we are now greater than the cache size
    if (jumpsCount < p / 2) {
      uint32_t lineSize = (1 << p) * sizeof(uint32_t);
      return {lineSize, true};
    }

    std::cout << "==========" << std::endl;
  }

  return {0, false};
}

int main() {
  uint32_t maxMemory = (1 << 30); // 1 GB
  uint32_t maxAssoc = 32;
  uint32_t maxStride = (1 << 23); // 8 MB
  uint32_t minStride = 16;        // 16 B

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

  auto [lineSizeBytes, lineSizeDetected] = lineSize(minStride, cacheSize);
  if (lineSizeDetected) {
    std::cout << "Detected cache line size: " << lineSizeBytes << " bytes"
              << std::endl;
  } else {
    std::cout << "Could not detect cache line size." << std::endl;
  }

  free(array);
  return 0;
}