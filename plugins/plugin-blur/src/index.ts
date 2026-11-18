import { mulTable, shgTable } from "./blur-tables.js";
import { JimpClass } from "@jimp/types";

/*
    Superfast Blur (0.5)
    http://www.quasimondo.com/BoxBlurForCanvas/FastBlur.js

    Copyright (c) 2011 Mario Klingemann

    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use,
    copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following
    conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.
*/

export const methods = {
  /**
   * A fast blur algorithm that produces similar effect to a Gaussian blur - but MUCH quicker
   * @param r the pixel radius of the blur
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.blur(5);
   * ```
   */
  blur<I extends JimpClass>(image: I, r: number) {
    if (typeof r !== "number") {
      throw new Error("r must be a number");
    }

    if (r < 1) {
      throw new Error("r must be greater than 0");
    }

    let rsum: number;
    let gsum: number;
    let bsum: number;
    let asum: number;
    let i: number;
    let p: number;
    let p1: number;
    let p2: number;
    let yp: number;
    let yi: number;
    let yw: number;

    const { width, height } = image.bitmap;
    const data = image.bitmap.data;
    const wm = width - 1;
    const hm = height - 1;
    const rowBytes = width << 2;
    const rad1 = r + 1;

    const mulSum = mulTable[r]!;
    const shgSum = shgTable[r]!;

    // Byte offsets of the pixel that enters / leaves the horizontal
    // sliding window at each x position (independent of y).
    const hIn = new Int32Array(width);
    const hOut = new Int32Array(width);

    for (let x = 0; x < width; x++) {
      p = x + rad1;
      hIn[x] = (p < wm ? p : wm) << 2;
      p = x - r;
      hOut[x] = (p > 0 ? p : 0) << 2;
    }

    // Pixel offsets (row index * width, x is added later) of the row
    // that enters / leaves the vertical sliding window at each y.
    const vIn = new Int32Array(height);
    const vOut = new Int32Array(height);

    for (let y = 0; y < height; y++) {
      p = y + rad1;
      vIn[y] = (p < hm ? p : hm) * width;
      p = y - r;
      vOut[y] = (p > 0 ? p : 0) * width;
    }

    const red = new Int32Array(width * height);
    const green = new Int32Array(width * height);
    const blue = new Int32Array(width * height);
    const alpha = new Int32Array(width * height);

    // Working copy of the RGBA bytes. Assigning an out-of-range integer
    // clamps it to 0..255 exactly like limit255() did, but in the typed
    // array setter instead of a function call with Math.max / Math.min.
    const work = new Uint8ClampedArray(data);

    let iterations = 2;

    while (iterations-- > 0) {
      yi = 0;
      yw = 0;

      for (let y = 0; y < height; y++) {
        rsum = work[yw]! * rad1;
        gsum = work[yw + 1]! * rad1;
        bsum = work[yw + 2]! * rad1;
        asum = work[yw + 3]! * rad1;

        for (i = 1; i <= r; i++) {
          p = yw + ((i > wm ? wm : i) << 2);
          rsum += work[p]!;
          gsum += work[p + 1]!;
          bsum += work[p + 2]!;
          asum += work[p + 3]!;
        }

        for (let x = 0; x < width; x++) {
          red[yi] = rsum;
          green[yi] = gsum;
          blue[yi] = bsum;
          alpha[yi] = asum;

          p1 = yw + hIn[x]!;
          p2 = yw + hOut[x]!;

          rsum += work[p1]! - work[p2]!;
          gsum += work[p1 + 1]! - work[p2 + 1]!;
          bsum += work[p1 + 2]! - work[p2 + 2]!;
          asum += work[p1 + 3]! - work[p2 + 3]!;

          yi++;
        }

        yw += rowBytes;
      }

      for (let x = 0; x < width; x++) {
        yp = x;
        rsum = red[yp]! * rad1;
        gsum = green[yp]! * rad1;
        bsum = blue[yp]! * rad1;
        asum = alpha[yp]! * rad1;

        for (i = 1; i <= r; i++) {
          yp += i > hm ? 0 : width;
          rsum += red[yp]!;
          gsum += green[yp]!;
          bsum += blue[yp]!;
          asum += alpha[yp]!;
        }

        yi = x << 2;

        for (let y = 0; y < height; y++) {
          // (x >>> shg) is always a non-negative integer, so the typed
          // array setter's built-in clamp matches limit255() byte for byte.
          work[yi] = (rsum * mulSum) >>> shgSum;
          work[yi + 1] = (gsum * mulSum) >>> shgSum;
          work[yi + 2] = (bsum * mulSum) >>> shgSum;
          work[yi + 3] = (asum * mulSum) >>> shgSum;

          p1 = x + vIn[y]!;
          p2 = x + vOut[y]!;

          rsum += red[p1]! - red[p2]!;
          gsum += green[p1]! - green[p2]!;
          bsum += blue[p1]! - blue[p2]!;
          asum += alpha[p1]! - alpha[p2]!;

          yi += rowBytes;
        }
      }
    }

    // Write the clamped results back into the original bitmap buffer,
    // preserving the reference held by the image.
    data.set(work);

    return image;
  },
  // http://blog.ivank.net/fastest-gaussian-blur.html
  /**
   * Applies a true Gaussian blur to the image (warning: this is VERY slow)
   * @param r the pixel radius of the blur
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.gaussian(15);
   * ```
   */
  gaussian<I extends JimpClass>(image: I, r: number) {
    if (typeof r !== "number") {
      throw new Error("r must be a number");
    }

    if (r < 1) {
      throw new Error("r must be greater than 0");
    }

    const rs = Math.ceil(r * 2.57); // significant radius
    const range = rs * 2 + 1;
    const rr2 = r * r * 2;
    const rr2pi = rr2 * Math.PI;

    const weights = [];

    for (let y = 0; y < range; y++) {
      const weightsRow = [];
      for (let x = 0; x < range; x++) {
        const dsq = (x - rs) ** 2 + (y - rs) ** 2;
        weightsRow[x] = Math.exp(-dsq / rr2) / rr2pi;
      }
      weights.push(weightsRow);
    }

    for (let y = 0; y < image.bitmap.height; y++) {
      for (let x = 0; x < image.bitmap.width; x++) {
        let red = 0;
        let green = 0;
        let blue = 0;
        let alpha = 0;
        let wsum = 0;

        for (let iy = 0; iy < range; iy++) {
          for (let ix = 0; ix < range; ix++) {
            const x1 = Math.min(
              image.bitmap.width - 1,
              Math.max(0, ix + x - rs)
            );
            const y1 = Math.min(
              image.bitmap.height - 1,
              Math.max(0, iy + y - rs)
            );
            const weight = weights[iy]![ix]!;
            const idx = (y1 * image.bitmap.width + x1) << 2;

            red += image.bitmap.data[idx]! * weight;
            green += image.bitmap.data[idx + 1]! * weight;
            blue += image.bitmap.data[idx + 2]! * weight;
            alpha += image.bitmap.data[idx + 3]! * weight;
            wsum += weight;
          }

          const idx = (y * image.bitmap.width + x) << 2;

          image.bitmap.data[idx] = Math.round(red / wsum);
          image.bitmap.data[idx + 1] = Math.round(green / wsum);
          image.bitmap.data[idx + 2] = Math.round(blue / wsum);
          image.bitmap.data[idx + 3] = Math.round(alpha / wsum);
        }
      }
    }

    return image;
  },
};
