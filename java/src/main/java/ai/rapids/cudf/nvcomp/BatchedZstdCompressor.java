/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.nvcomp;

import ai.rapids.cudf.Cuda;

/** Multi-buffer ZSTD compressor */
public class BatchedZstdCompressor extends BatchedCompressor {
  /**
   * Construct a batched ZSTD compressor instance
   * @param chunkSize maximum amount of uncompressed data to compress as a single chunk.
   *                  Inputs larger than this will be compressed in multiple chunks.
   * @param maxIntermediateBufferSize desired maximum size of intermediate device buffers
   *                                  used during compression.
   */
  public BatchedZstdCompressor(long chunkSize, long maxIntermediateBufferSize) {
    super(chunkSize, NvcompJni.batchedZstdCompressGetMaxOutputChunkSize(chunkSize),
        maxIntermediateBufferSize);
  }

  @Override
  protected long batchedCompressGetTempSize(long batchSize, long maxChunkSize, long totalSize,
      Cuda.Stream stream) {
    return NvcompJni.batchedZstdCompressGetTempSize(batchSize, maxChunkSize, totalSize,
        stream.getStream());
  }

  @Override
  protected void batchedCompressAsync(long devInPtrs, long devInSizes, long chunkSize,
      long batchSize, long tempPtr, long tempSize, long devOutPtrs,
      long compressedSizesOutPtr, long stream) {
    NvcompJni.batchedZstdCompressAsync(devInPtrs, devInSizes, chunkSize, batchSize,
        tempPtr, tempSize, devOutPtrs, compressedSizesOutPtr, stream);
  }
}
