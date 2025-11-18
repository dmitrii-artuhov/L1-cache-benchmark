#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <ratio>
#include <string>
#include <vector>

using namespace std::chrono;

const uint32_t ARRAY_READS_COUNT = 1'000'000;
const uint32_t WARNUP_READS_COUNT = 5000;
const uint32_t BATCHES_COUNT = 5;
const uint32_t PAGE_SIZE = (1 << 14); // 16 KB
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
  if (bytes >= (1 << 20))
    return std::to_string(bytes / (1 << 20)) + "MB";
  else if (bytes >= (1 << 10))
    return std::to_string(bytes / (1 << 10)) + "KB";
  return std::to_string(bytes) + "B";
}

uint32_t log2(uint32_t n) {
  rassert(n != 0, 2);
  uint32_t log = 0;
  while (n >>= 1)
    ++log;
  return log;
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

void fillShuffledIndexes(uint32_t stride, uint32_t elems, uint32_t seed = 42) {
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
                         uint32_t warnupReadsCount, uint32_t batchesCount) {
  volatile uint32_t sink = 0; // prevents optimization

  timetype diff = timetype::zero();
  uint32_t iterationsPerBatch = readsCount;
  for (uint32_t batch = 0; batch < batchesCount; ++batch) {
    // fill indexes
    fillShuffledIndexes(stride, elems);

    // some reads to warm up cache
    for (uint32_t i = 0, idx = 0; i < warnupReadsCount; i++) {
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

bool deltaDiff(timetype currentTime, timetype prevTime, double fraction) {
  auto delta = currentTime - prevTime;
  return currentTime > prevTime &&
         static_cast<double>(delta.count()) / prevTime.count() > fraction;
}

void capacityAndAssociativity(uint32_t maxMemory, uint32_t maxAssoc,
                              uint32_t maxStride, uint32_t minStride) {
  uint32_t timeFactor = 10'000;
  uint32_t stride = minStride / sizeof(uint32_t); // elements
  uint32_t stridePow = 0;
  timetype currentTime;
  timetype prevTime;
  // matrix elems x strides
  std::vector<std::vector<timetype>> times(
      maxAssoc, std::vector<timetype>(maxStride, timetype{}));
  std::vector<std::vector<bool>> jumps(maxAssoc,
                                       std::vector<bool>(maxStride, false));

  while (stride * sizeof(uint32_t) <= maxStride) {
    uint32_t elems = 1;
    // calculate time jumps
    while (elems < maxAssoc) {
      currentTime = timeOfArrayRead(stride, elems, ARRAY_READS_COUNT,
                                    WARNUP_READS_COUNT, BATCHES_COUNT);
      times[elems][stridePow] = currentTime;

      if (elems > 1) {
        bool isJump = false; // deltaDiff(currentTime, prevTime, 0.0);
        if (isJump) {
          jumps[elems][stridePow] = true;
          // std::cout << "Jump detected at elems=" << elems
          //           << " stride=" << bytesToString(stride * sizeof(uint32_t))
          //           << " time=" << currentTime.count() / timeFactor
          //           << " prevTime=" << prevTime.count() / timeFactor
          //           << std::endl;
        }
      }

      elems++;
      prevTime = currentTime;
    }

    // TODO: check for movement, then double the stride, otherwise break
    stride *= 2;
    stridePow++;
  }

  // printing results
  int width = 10;
  std::cout << std::setw(width) << "s/e";
  for (uint32_t p = 0; p < stridePow; p++) {
    uint32_t bytes = (1 << p) * minStride;
    std::cout << std::setw(width) << bytesToString(bytes);
  }
  std::cout << std::endl;

  for (uint32_t s = 1; s < maxAssoc; s *= 2) {
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

int main() {
  uint32_t maxMemory = (1 << 30); // 1 GB
  uint32_t maxAssoc = 32;
  uint32_t maxStride = (1 << 25); // 32 MB
  uint32_t minStride = 16;        // 16 B

  rassert(maxAssoc * maxStride <= maxMemory, 1);

  array = static_cast<uint32_t *>(aligned_alloc(PAGE_SIZE, maxMemory));
  arrayLen = maxMemory / sizeof(uint32_t);

  std::cout << "array: " << array << std::endl;
  std::cout << "len: " << arrayLen << std::endl;

  capacityAndAssociativity(maxMemory, maxAssoc + 1, maxStride, minStride);

  free(array);
  return 0;
}