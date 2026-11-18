import { expect, test, describe } from "vitest";
import png from "@jimp/js-png";
import { createJimp } from "@jimp/core";

import { methods } from "./index.js";

const jimp = createJimp({ formats: [png], plugins: [methods] });

/**
 * Reference implementation of the generic convolution() path, kept verbatim
 * so the optimized 3x3 fast path can be checked against it byte for byte.
 * Only RGB are touched by convolution(); alpha is left alone.
 */
function referenceConvolution(
  data: Uint8Array,
  width: number,
  height: number,
  kernel: number[][],
) {
  const newData = Buffer.from(data);
  const kRows = kernel.length;
  const kCols = kernel[0]!.length;
  const rowEnd = Math.floor(kRows / 2);
  const colEnd = Math.floor(kCols / 2);
  const rowIni = -rowEnd;
  const colIni = -colEnd;

  const pixelIndex = (x: number, y: number) => {
    let xi = x;
    let yi = y;

    if (x < 0) xi = 0;
    if (x >= width) xi = width - 1;
    if (y < 0) yi = 0;
    if (y >= height) yi = height - 1;

    return (width * yi + xi) << 2;
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let rSum = 0;
      let gSum = 0;
      let bSum = 0;

      for (let row = rowIni; row <= rowEnd; row++) {
        for (let col = colIni; col <= colEnd; col++) {
          const weight = kernel[row + rowEnd]![col + colEnd]!;
          const idxi = pixelIndex(x + col, y + row);

          rSum += weight * data[idxi]!;
          gSum += weight * data[idxi + 1]!;
          bSum += weight * data[idxi + 2]!;
        }
      }

      const idx = (y * width + x) << 2;
      newData[idx] = rSum < 0 ? 0 : rSum > 255 ? 255 : rSum;
      newData[idx + 1] = gSum < 0 ? 0 : gSum > 255 ? 255 : gSum;
      newData[idx + 2] = bSum < 0 ? 0 : bSum > 255 ? 255 : bSum;
    }
  }

  return newData;
}

// Deterministic pseudo-random RGBA bytes so failures stay reproducible.
function randomBytes(length: number, seed: number) {
  let state = seed;
  return Uint8Array.from({ length }, () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state & 0xff;
  });
}

function makeImage(width: number, height: number, seed: number) {
  return jimp.fromBitmap({
    width,
    height,
    data: Buffer.from(randomBytes(width * height * 4, seed)),
  });
}

const integerKernels = [
  [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ],
  [
    [-1, -1, 0],
    [-1, 1, 1],
    [0, 1, 1],
  ],
  [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
  ],
  [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
  ],
  [
    [-7, 3, -2],
    [4, 9, -5],
    [1, -3, 6],
  ],
];

describe("convolution fast path", () => {
  test.each(integerKernels.map((kernel) => ({ kernel })))(
    "matches the generic implementation byte for byte with kernel $kernel",
    ({ kernel }) => {
      // Dimensions are deliberately large enough that Jimp.fromBitmap()
      // allocates a standalone buffer instead of Node's shared 8 KiB pool.
      const sizes = [
        [120, 120],
        [137, 91],
        [160, 128],
      ] as const;

      sizes.forEach(([width, height], caseIndex) => {
        const seed = 1000 + caseIndex;

        // Both images start from identical bytes; the reference runs the
        // generic EXTEND algorithm while the plugin takes its fast path.
        const reference = makeImage(width, height, seed);
        const optimized = makeImage(width, height, seed);
        const expected = referenceConvolution(
          reference.bitmap.data,
          width,
          height,
          kernel,
        );

        optimized.convolution(kernel);

        expect(Buffer.compare(optimized.bitmap.data, expected)).toBe(0);
      });
    },
  );

  test("leaves the alpha channel untouched", () => {
    const before = makeImage(120, 96, 42);
    const alphaBefore = Buffer.from(before.bitmap.data);
    const image = makeImage(120, 96, 42);

    image.convolution([
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ]);

    for (let i = 3; i < alphaBefore.length; i += 4) {
      expect(image.bitmap.data[i]).toBe(alphaBefore[i]);
    }
  });

  test("fractional kernels keep using the generic path", () => {
    const width = 120;
    const height = 96;
    const kernel = [
      [1 / 9, 1 / 9, 1 / 9],
      [1 / 9, 1 / 9, 1 / 9],
      [1 / 9, 1 / 9, 1 / 9],
    ];

    const reference = makeImage(width, height, 7);
    const optimized = makeImage(width, height, 7);
    const expected = referenceConvolution(
      reference.bitmap.data,
      width,
      height,
      kernel,
    );

    optimized.convolution(kernel);

    expect(Buffer.compare(optimized.bitmap.data, expected)).toBe(0);
  });

  test.each([
    [1, 1],
    [2, 3],
    [3, 2],
  ] as const)(
    "edge clamping matches the generic path on a %ix%i bitmap",
    (width, height) => {
      const kernel = [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
      ];

      // Build each image with its own exact-sized backing store so the
      // shared 8 KiB buffer pool cannot bleed bytes into these tiny bitmaps.
      const build = (seed: number) => {
        const backing = new Uint8Array(width * height * 4);
        let state = seed;

        for (let i = 0; i < backing.length; i++) {
          state = (state * 1103515245 + 12345) & 0x7fffffff;
          backing[i] = state & 0xff;
        }

        return jimp.fromBitmap({
          width,
          height,
          data: Buffer.from(backing.buffer, 0, backing.length),
        });
      };

      const reference = build(99);
      const optimized = build(99);
      const expected = referenceConvolution(
        reference.bitmap.data,
        width,
        height,
        kernel,
      );

      optimized.convolution(kernel);

      expect(Buffer.compare(optimized.bitmap.data, expected)).toBe(0);
    },
  );
});
