import tinyColor from "tinycolor2";
import { clone, limit255, scan } from "@jimp/utils";
import { Edge, JimpClass, RGBColor } from "@jimp/types";
import { z } from "zod";

const ConvolutionMatrixSchema = z.array(z.number()).min(1).array();
const ConvolutionComplexOptionsSchema = z.object({
  /** a matrix to weight the neighbors sum */
  kernel: ConvolutionMatrixSchema,
  /**define how to sum pixels from outside the border */
  edgeHandling: z.nativeEnum(Edge).optional(),
});
const ConvolutionOptionsSchema = z.union([
  ConvolutionMatrixSchema,
  ConvolutionComplexOptionsSchema,
]);

type ConvolutionOptions = z.infer<typeof ConvolutionOptionsSchema>;

const ConvoluteComplexOptionsSchema = z.object({
  /** the convolution kernel */
  kernel: ConvolutionMatrixSchema,
  /** the x position of the region to apply convolution to */
  x: z.number().optional(),
  /** the y position of the region to apply convolution to */
  y: z.number().optional(),
  /** the width of the region to apply convolution to */
  w: z.number().optional(),
  /** the height of the region to apply convolution to */
  h: z.number().optional(),
});
const ConvoluteOptionsSchema = z.union([
  ConvolutionMatrixSchema,
  ConvoluteComplexOptionsSchema,
]);

type ConvoluteComplexOptions = z.infer<typeof ConvoluteComplexOptionsSchema>;
type ConvoluteOptions = z.infer<typeof ConvoluteOptionsSchema>;

const PixelateSize = z.number().min(1).max(Infinity);
const PixelateComplexOptionsSchema = z.object({
  /** the size of the pixels */
  size: PixelateSize,
  /** the x position of the region to pixelate */
  x: z.number().optional(),
  /** the y position of the region to pixelate */
  y: z.number().optional(),
  /** the width of the region to pixelate */
  w: z.number().optional(),
  /** the height of the region to pixelate */
  h: z.number().optional(),
});
const PixelateOptionsSchema = z.union([
  PixelateSize,
  PixelateComplexOptionsSchema,
]);

type PixelateComplexOptions = z.infer<typeof PixelateComplexOptionsSchema>;
type PixelateOptions = z.infer<typeof PixelateOptionsSchema>;

function applyKernel(
  image: JimpClass,
  kernel: number[][],
  x: number,
  y: number,
) {
  const value = [0, 0, 0, 0] as [number, number, number, number];
  const size = (kernel.length - 1) / 2;

  for (let kx = 0; kx < kernel.length; kx += 1) {
    for (let ky = 0; ky < kernel[kx]!.length; ky += 1) {
      const idx = image.getPixelIndex(x + kx - size, y + ky - size);

      value[0] += image.bitmap.data[idx]! * kernel[kx]![ky]!;
      value[1] += image.bitmap.data[idx + 1]! * kernel[kx]![ky]!;
      value[2] += image.bitmap.data[idx + 2]! * kernel[kx]![ky]!;
      value[3] += image.bitmap.data[idx + 3]! * kernel[kx]![ky]!;
    }
  }

  return value;
}

function mix(clr: RGBColor, clr2: RGBColor, p = 50) {
  return {
    r: (clr2.r - clr.r) * (p / 100) + clr.r,
    g: (clr2.g - clr.g) * (p / 100) + clr.g,
    b: (clr2.b - clr.b) * (p / 100) + clr.b,
  };
}

const HueActionSchema = z.object({
  apply: z.literal("hue"),
  params: z.tuple([z.number().min(-360).max(360)]),
});
export type HueAction = z.infer<typeof HueActionSchema>;

const SpinActionSchema = z.object({
  apply: z.literal("spin"),
  params: z.tuple([z.number().min(-360).max(360)]),
});
export type SpinAction = z.infer<typeof SpinActionSchema>;

const LightenActionSchema = z.object({
  apply: z.literal("lighten"),
  params: z.tuple([z.number().min(0).max(100)]).optional(),
});
export type LightenAction = z.infer<typeof LightenActionSchema>;

const RGBColorSchema = z.object({
  r: z.number().min(0).max(255),
  g: z.number().min(0).max(255),
  b: z.number().min(0).max(255),
});

const MixActionSchema = z.object({
  apply: z.literal("mix"),
  params: z.union([
    z.tuple([RGBColorSchema]),
    z.tuple([RGBColorSchema, z.number().min(0).max(100)]),
  ]),
});
export type MixAction = z.infer<typeof MixActionSchema>;

const TintActionSchema = z.object({
  apply: z.literal("tint"),
  params: z.tuple([z.number().min(0).max(100)]).optional(),
});
export type TintAction = z.infer<typeof TintActionSchema>;

const ShadeActionSchema = z.object({
  apply: z.literal("shade"),
  params: z.tuple([z.number().min(0).max(100)]).optional(),
});
export type ShadeAction = z.infer<typeof ShadeActionSchema>;

const XorActionSchema = z.object({
  apply: z.literal("xor"),
  params: z.tuple([RGBColorSchema]),
});
export type XorAction = z.infer<typeof XorActionSchema>;

const RedActionSchema = z.object({
  apply: z.literal("red"),
  params: z.tuple([z.number().min(-255).max(255)]),
});
export type RedAction = z.infer<typeof RedActionSchema>;

const GreenActionSchema = z.object({
  apply: z.literal("green"),
  params: z.tuple([z.number().min(-255).max(255)]),
});
export type GreenAction = z.infer<typeof GreenActionSchema>;

const BlueActionSchema = z.object({
  apply: z.literal("blue"),
  params: z.tuple([z.number().min(-255).max(255)]),
});
export type BlueAction = z.infer<typeof BlueActionSchema>;

const BrightenActionSchema = z.object({
  apply: z.literal("brighten"),
  params: z.tuple([z.number().min(0).max(100)]).optional(),
});
export type BrightenAction = z.infer<typeof BrightenActionSchema>;

const DarkenActionSchema = z.object({
  apply: z.literal("darken"),
  params: z.tuple([z.number().min(0).max(100)]).optional(),
});
export type DarkenAction = z.infer<typeof DarkenActionSchema>;

const DesaturateActionSchema = z.object({
  apply: z.literal("desaturate"),
  params: z.tuple([z.number().min(0).max(100)]).optional(),
});
export type DesaturateAction = z.infer<typeof DesaturateActionSchema>;

const SaturateActionSchema = z.object({
  apply: z.literal("saturate"),
  params: z.tuple([z.number().min(0).max(100)]).optional(),
});
export type SaturateAction = z.infer<typeof SaturateActionSchema>;

const GrayscaleActionSchema = z.object({
  apply: z.literal("greyscale"),
  params: z.tuple([]).optional(),
});
export type GrayscaleAction = z.infer<typeof GrayscaleActionSchema>;

const ColorActionNameSchema = z.union([
  HueActionSchema,
  SpinActionSchema,
  LightenActionSchema,
  MixActionSchema,
  TintActionSchema,
  ShadeActionSchema,
  XorActionSchema,
  RedActionSchema,
  GreenActionSchema,
  BlueActionSchema,
  BrightenActionSchema,
  DarkenActionSchema,
  DesaturateActionSchema,
  SaturateActionSchema,
  GrayscaleActionSchema,
]);
export type ColorAction = z.infer<typeof ColorActionNameSchema>;

export const ColorActionName = Object.freeze({
  LIGHTEN: "lighten",
  BRIGHTEN: "brighten",
  DARKEN: "darken",
  DESATURATE: "desaturate",
  SATURATE: "saturate",
  GREYSCALE: "greyscale",
  SPIN: "spin",
  HUE: "hue",
  MIX: "mix",
  TINT: "tint",
  SHADE: "shade",
  XOR: "xor",
  RED: "red",
  GREEN: "green",
  BLUE: "blue",
});

/**
 * Get an image's histogram
 * @return An object with an array of color occurrence counts for each channel (r,g,b)
 */
function histogram<I extends JimpClass>(image: I) {
  const histogram = {
    r: new Array<number>(256).fill(0),
    g: new Array<number>(256).fill(0),
    b: new Array<number>(256).fill(0),
  };

  image.scan((_, __, index) => {
    histogram.r[image.bitmap.data[index + 0]!]!++;
    histogram.g[image.bitmap.data[index + 1]!]!++;
    histogram.b[image.bitmap.data[index + 2]!]!++;
  });

  return histogram;
}

/**
 * Normalize values
 * @param  value Pixel channel value.
 * @param  min   Minimum value for channel
 * @param  max   Maximum value for channel
 */
const normalizeValue = function (value: number, min: number, max: number) {
  return ((value - min) * 255) / (max - min);
};

const getBounds = function (histogramChannel: number[]) {
  return [
    histogramChannel.findIndex((value) => value > 0),
    255 -
      histogramChannel
        .slice()
        .reverse()
        .findIndex((value) => value > 0),
  ];
};

export const methods = {
  /**
   * Normalizes the image.
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.normalize();
   * ```
   */
  normalize<I extends JimpClass>(image: I) {
    const h = histogram(image);

    // store bounds (minimum and maximum values)
    const bounds = {
      r: getBounds(h.r),
      g: getBounds(h.g),
      b: getBounds(h.b),
    };

    // apply value transformations
    image.scan((_, __, idx) => {
      const r = image.bitmap.data[idx + 0]!;
      const g = image.bitmap.data[idx + 1]!;
      const b = image.bitmap.data[idx + 2]!;

      image.bitmap.data[idx + 0] = normalizeValue(
        r,
        bounds.r[0]!,
        bounds.r[1]!,
      );
      image.bitmap.data[idx + 1] = normalizeValue(
        g,
        bounds.g[0]!,
        bounds.g[1]!,
      );
      image.bitmap.data[idx + 2] = normalizeValue(
        b,
        bounds.b[0]!,
        bounds.b[1]!,
      );
    });

    return image;
  },

  /**
   * Inverts the colors in the image.
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.invert();
   * ```
   */
  invert<I extends JimpClass>(image: I) {
    image.scan((_, __, idx) => {
      image.bitmap.data[idx] = 255 - image.bitmap.data[idx]!;
      image.bitmap.data[idx + 1] = 255 - image.bitmap.data[idx + 1]!;
      image.bitmap.data[idx + 2] = 255 - image.bitmap.data[idx + 2]!;
    });

    return image;
  },

  /**
   * Adjusts the brightness of the image
   * @param val the amount to adjust the brightness.
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.brightness(0.5);
   * ```
   */
  brightness<I extends JimpClass>(image: I, val: number) {
    if (typeof val !== "number") {
      throw new Error("val must be numbers");
    }

    image.scan((_, __, idx) => {
      image.bitmap.data[idx]! = limit255(image.bitmap.data[idx]! * val);
      image.bitmap.data[idx + 1]! = limit255(image.bitmap.data[idx + 1]! * val);
      image.bitmap.data[idx + 2]! = limit255(image.bitmap.data[idx + 2]! * val);
    });

    return image;
  },

  /**
   * Adjusts the contrast of the image
   * @param val the amount to adjust the contrast, a number between -1 and +1
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.contrast(0.75);
   * ```
   */
  contrast<I extends JimpClass>(image: I, val: number) {
    if (typeof val !== "number") {
      throw new Error("val must be numbers");
    }

    if (val < -1 || val > +1) {
      throw new Error("val must be a number between -1 and +1");
    }

    const factor = (val + 1) / (1 - val);

    function adjust(value: number) {
      value = Math.floor(factor * (value - 127) + 127);
      return value < 0 ? 0 : value > 255 ? 255 : value;
    }

    image.scan((_, __, idx) => {
      image.bitmap.data[idx] = adjust(image.bitmap.data[idx]!);
      image.bitmap.data[idx + 1] = adjust(image.bitmap.data[idx + 1]!);
      image.bitmap.data[idx + 2] = adjust(image.bitmap.data[idx + 2]!);
    });

    return image;
  },

  /**
   * Apply a posterize effect
   * @param  n the amount to adjust the contrast, minimum threshold is two
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.posterize(5);
   * ```
   */
  posterize<I extends JimpClass>(image: I, n: number) {
    if (typeof n !== "number") {
      throw new Error("n must be numbers");
    }

    // minimum of 2 levels
    if (n < 2) {
      n = 2;
    }

    image.scan((_, __, idx) => {
      const r = image.bitmap.data[idx]!;
      const g = image.bitmap.data[idx + 1]!;
      const b = image.bitmap.data[idx + 2]!;

      image.bitmap.data[idx] =
        (Math.floor((r / 255) * (n - 1)) / (n - 1)) * 255;
      image.bitmap.data[idx + 1] =
        (Math.floor((g / 255) * (n - 1)) / (n - 1)) * 255;
      image.bitmap.data[idx + 2] =
        (Math.floor((b / 255) * (n - 1)) / (n - 1)) * 255;
    });

    return image;
  },

  /**
   * Removes colour from the image using ITU Rec 709 luminance values
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.greyscale();
   * ```
   */
  greyscale<I extends JimpClass>(image: I) {
    image.scan((_, __, idx) => {
      // const grey = parseInt(
      //   0.2126 * image.bitmap.data[idx]! +
      //     0.7152 * image.bitmap.data[idx + 1]! +
      //     0.0722 * image.bitmap.data[idx + 2]!,
      //   10
      // );
      const grey =
        0.2126 * image.bitmap.data[idx]! +
        0.7152 * image.bitmap.data[idx + 1]! +
        0.0722 * image.bitmap.data[idx + 2]!;

      image.bitmap.data[idx] = grey;
      image.bitmap.data[idx + 1] = grey;
      image.bitmap.data[idx + 2] = grey;
    });

    return image;
  },

  /**
   * Multiplies the opacity of each pixel by a factor between 0 and 1
   * @param f A number, the factor by which to multiply the opacity of each pixel
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.opacity(0.5);
   * ```
   */
  opacity<I extends JimpClass>(image: I, f: number) {
    if (typeof f !== "number") {
      throw new Error("f must be a number");
    }

    if (f < 0 || f > 1) {
      throw new Error("f must be a number from 0 to 1");
    }

    image.scan((_, __, idx) => {
      const v = image.bitmap.data[idx + 3]! * f;
      image.bitmap.data[idx + 3] = v;
    });

    return image;
  },

  /**
   * Applies a sepia tone to the image.
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.sepia();
   * ```
   */
  sepia<I extends JimpClass>(image: I) {
    image.scan((_, __, idx) => {
      let red = image.bitmap.data[idx]!;
      let green = image.bitmap.data[idx + 1]!;
      let blue = image.bitmap.data[idx + 2]!;

      red = red * 0.393 + green * 0.769 + blue * 0.189;
      green = red * 0.349 + green * 0.686 + blue * 0.168;
      blue = red * 0.272 + green * 0.534 + blue * 0.131;

      image.bitmap.data[idx] = red < 255 ? red : 255;
      image.bitmap.data[idx + 1] = green < 255 ? green : 255;
      image.bitmap.data[idx + 2] = blue < 255 ? blue : 255;
    });

    return image;
  },

  /**
   * Fades each pixel by a factor between 0 and 1
   * @param f A number from 0 to 1. 0 will haven no effect. 1 will turn the image completely transparent.
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.fade(0.7);
   * ```
   */
  fade<I extends JimpClass>(image: I, f: number) {
    if (typeof f !== "number") {
      throw new Error("f must be a number");
    }

    if (f < 0 || f > 1) {
      throw new Error("f must be a number from 0 to 1");
    }

    // this method is an alternative to opacity (which may be deprecated)
    return this.opacity(image, 1 - f);
  },

  /**
   * Adds each element of the image to its local neighbors, weighted by the kernel
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.convolute([
   *   [-1, -1, 0],
   *   [-1, 1, 1],
   *   [0, 1, 1],
   * ]);
   * ```
   */
  convolution<I extends JimpClass>(image: I, options: ConvolutionOptions) {
    const parsed = ConvolutionOptionsSchema.parse(options);
    const { kernel, edgeHandling = Edge.EXTEND } =
      "kernel" in parsed ? parsed : { kernel: parsed, edgeHandling: undefined };

    if (!kernel[0]) {
      throw new Error("kernel must be a matrix");
    }

    const { width, height } = image.bitmap;
    const data = image.bitmap.data;
    const kRows = kernel.length;
    const kCols = kernel[0].length;
    const rowEnd = Math.floor(kRows / 2);
    const colEnd = Math.floor(kCols / 2);
    const rowIni = -rowEnd;
    const colIni = -colEnd;

    // Fast path: a 3x3 kernel made of integer weights with the default
    // edge handling. All arithmetic stays integer and no per-pixel helper
    // calls or closures are involved, while every output byte matches the
    // generic implementation exactly (alpha is left untouched).
    if (
      edgeHandling === Edge.EXTEND &&
      kRows === 3 &&
      kCols === 3 &&
      kernel.every((kernelRow) =>
        kernelRow.every((weight) => Number.isInteger(weight)),
      )
    ) {
      const k00 = kernel[0]![0]!;
      const k01 = kernel[0]![1]!;
      const k02 = kernel[0]![2]!;
      const k10 = kernel[1]![0]!;
      const k11 = kernel[1]![1]!;
      const k12 = kernel[1]![2]!;
      const k20 = kernel[2]![0]!;
      const k21 = kernel[2]![1]!;
      const k22 = kernel[2]![2]!;

      const newData = Buffer.from(data);

      for (let y = 0; y < height; y++) {
        const yMinus = y > 0 ? y - 1 : 0;
        const yPlus = y < height - 1 ? y + 1 : height - 1;
        const rowOff = y * width;
        const rowMinusOff = yMinus * width;
        const rowPlusOff = yPlus * width;

        for (let x = 0; x < width; x++) {
          const xMinus = x > 0 ? x - 1 : 0;
          const xPlus = x < width - 1 ? x + 1 : width - 1;

          // Byte offsets of the 3x3 neighborhood; computed once and
          // reused for every channel.
          const i00 = (rowMinusOff + xMinus) << 2;
          const i01 = (rowMinusOff + x) << 2;
          const i02 = (rowMinusOff + xPlus) << 2;
          const i10 = (rowOff + xMinus) << 2;
          const i11 = (rowOff + x) << 2;
          const i12 = (rowOff + xPlus) << 2;
          const i20 = (rowPlusOff + xMinus) << 2;
          const i21 = (rowPlusOff + x) << 2;
          const i22 = (rowPlusOff + xPlus) << 2;

          const rSum =
            k00 * data[i00]! +
            k01 * data[i01]! +
            k02 * data[i02]! +
            k10 * data[i10]! +
            k11 * data[i11]! +
            k12 * data[i12]! +
            k20 * data[i20]! +
            k21 * data[i21]! +
            k22 * data[i22]!;

          const gSum =
            k00 * data[i00 + 1]! +
            k01 * data[i01 + 1]! +
            k02 * data[i02 + 1]! +
            k10 * data[i10 + 1]! +
            k11 * data[i11 + 1]! +
            k12 * data[i12 + 1]! +
            k20 * data[i20 + 1]! +
            k21 * data[i21 + 1]! +
            k22 * data[i22 + 1]!;

          const bSum =
            k00 * data[i00 + 2]! +
            k01 * data[i01 + 2]! +
            k02 * data[i02 + 2]! +
            k10 * data[i10 + 2]! +
            k11 * data[i11 + 2]! +
            k12 * data[i12 + 2]! +
            k20 * data[i20 + 2]! +
            k21 * data[i21 + 2]! +
            k22 * data[i22 + 2]!;

          newData[i11] = rSum < 0 ? 0 : rSum > 255 ? 255 : rSum;
          newData[i11 + 1] = gSum < 0 ? 0 : gSum > 255 ? 255 : gSum;
          newData[i11 + 2] = bSum < 0 ? 0 : bSum > 255 ? 255 : bSum;
        }
      }

      image.bitmap.data = newData;

      return image;
    }

    const newData = Buffer.from(data);
    let weight: number;
    let rSum: number;
    let gSum: number;
    let bSum: number;
    let ri: number;
    let gi: number;
    let bi: number;
    let xi: number;
    let yi: number;
    let idxi: number;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        bSum = 0;
        gSum = 0;
        rSum = 0;

        for (let row = rowIni; row <= rowEnd; row++) {
          for (let col = colIni; col <= colEnd; col++) {
            xi = x + col;
            yi = y + row;
            weight = kernel[row + rowEnd]![col + colEnd]!;
            idxi = image.getPixelIndex(xi, yi, edgeHandling);

            if (idxi === -1) {
              bi = 0;
              gi = 0;
              ri = 0;
            } else {
              ri = data[idxi + 0]!;
              gi = data[idxi + 1]!;
              bi = data[idxi + 2]!;
            }

            rSum += weight * ri;
            gSum += weight * gi;
            bSum += weight * bi;
          }
        }

        if (rSum < 0) {
          rSum = 0;
        }

        if (gSum < 0) {
          gSum = 0;
        }

        if (bSum < 0) {
          bSum = 0;
        }

        if (rSum > 255) {
          rSum = 255;
        }

        if (gSum > 255) {
          gSum = 255;
        }

        if (bSum > 255) {
          bSum = 255;
        }

        const idx = (y * width + x) << 2;
        newData[idx + 0] = rSum;
        newData[idx + 1] = gSum;
        newData[idx + 2] = bSum;
      }
    }

    image.bitmap.data = newData;

    return image;
  },

  /**
   * Set the alpha channel on every pixel to fully opaque.
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.opaque();
   * ```
   */
  opaque<I extends JimpClass>(image: I) {
    image.scan((_, __, idx) => {
      image.bitmap.data[idx + 3] = 255;
    });

    return image;
  },

  /**
   * Pixelates the image or a region
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * // pixelate the whole image
   * image.pixelate(10);
   *
   * // pixelate a region
   * image.pixelate(10, 10, 10, 20, 20);
   * ```
   */
  pixelate<I extends JimpClass>(image: I, options: PixelateOptions) {
    const parsed = PixelateOptionsSchema.parse(options);
    const {
      size,
      x = 0,
      y = 0,
      w = image.bitmap.width - x,
      h = image.bitmap.height - y,
    } = typeof parsed === "number"
      ? ({ size: parsed } as PixelateComplexOptions)
      : parsed;

    const kernel = [
      [1 / 16, 2 / 16, 1 / 16],
      [2 / 16, 4 / 16, 2 / 16],
      [1 / 16, 2 / 16, 1 / 16],
    ];

    const source = clone(image);

    scan(source, x, y, w, h, (xx, yx, idx) => {
      xx = size * Math.floor(xx / size);
      yx = size * Math.floor(yx / size);

      const value = applyKernel(source, kernel, xx, yx);

      image.bitmap.data[idx] = value[0]!;
      image.bitmap.data[idx + 1] = value[1]!;
      image.bitmap.data[idx + 2] = value[2]!;
      image.bitmap.data[idx + 3] = value[3]!;
    });

    return image;
  },

  /**
   * Applies a convolution kernel to the image or a region
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * // apply a convolution kernel to the whole image
   * image.convolution([
   *   [-1, -1, 0],
   *   [-1, 1, 1],
   *   [0, 1, 1],
   * ]);
   *
   * // apply a convolution kernel to a region
   * image.convolution([
   *   [-1, -1, 0],
   *   [-1, 1, 1],
   *   [0, 1, 1],
   * ], 10, 10, 10, 20);
   * ```
   */
  convolute<I extends JimpClass>(image: I, options: ConvoluteOptions) {
    const parsed = ConvoluteOptionsSchema.parse(options);
    const {
      kernel,
      x = 0,
      y = 0,
      w = image.bitmap.width - x,
      h = image.bitmap.height - y,
    } = "kernel" in parsed
      ? parsed
      : ({ kernel: parsed } as ConvoluteComplexOptions);

    const source = clone(image);
    const srcData = source.bitmap.data;
    const { width, height } = source.bitmap;
    const size = (kernel.length - 1) / 2;
    const data = image.bitmap.data;

    // scan() rounds the region parameters before iterating; keep the
    // same contract for the direct loop below.
    const rx = Math.round(x);
    const ry = Math.round(y);
    const rw = Math.round(w);
    const rh = Math.round(h);

    for (let yy = ry; yy < ry + rh; yy++) {
      for (let xx = rx; xx < rx + rw; xx++) {
        let rSum = 0;
        let gSum = 0;
        let bSum = 0;
        let aSum = 0;

        for (let kx = 0; kx < kernel.length; kx += 1) {
          // applyKernel() clamps sample coordinates to the bitmap edges,
          // matching the default EXTEND edge handling of getPixelIndex().
          const sampleY = Math.min(height - 1, Math.max(0, yy + kx - size));

          for (let ky = 0; ky < kernel[kx]!.length; ky += 1) {
            const sampleX = Math.min(width - 1, Math.max(0, xx + ky - size));
            const weight = kernel[kx]![ky]!;
            const idx = (sampleY * width + sampleX) << 2;

            rSum += srcData[idx]! * weight;
            gSum += srcData[idx + 1]! * weight;
            bSum += srcData[idx + 2]! * weight;
            aSum += srcData[idx + 3]! * weight;
          }
        }

        const idx = (yy * width + xx) << 2;
        data[idx] = limit255(rSum);
        data[idx + 1] = limit255(gSum);
        data[idx + 2] = limit255(bSum);
        data[idx + 3] = limit255(aSum);
      }
    }

    return image;
  },

  /**
   * Apply multiple color modification rules
   * @param  actions list of color modification rules, in following format: { apply: '<rule-name>', params: [ <rule-parameters> ]  }
   * @example
   * ```ts
   * import { Jimp } from "jimp";
   *
   * const image = await Jimp.read("test/image.png");
   *
   * image.color([
   *   { apply: "hue", params: [-90] },
   *   { apply: "lighten", params: [50] },
   *   { apply: "xor", params: ["#06D"] },
   * ]);
   * ```
   */
  color<I extends JimpClass>(image: I, actions: ColorAction[]) {
    if (!actions || !Array.isArray(actions)) {
      throw new Error("actions must be an array");
    }

    actions.forEach((action) => ColorActionNameSchema.parse(action));

    actions = actions.map((action) => {
      if (action.apply === "xor" || action.apply === "mix") {
        action.params[0] = tinyColor(action.params[0]).toRgb();
      }

      return action;
    });

    image.scan((_, __, idx) => {
      let clr: RGBColor = {
        r: image.bitmap.data[idx]!,
        g: image.bitmap.data[idx + 1]!,
        b: image.bitmap.data[idx + 2]!,
      };

      const colorModifier = (i: "r" | "g" | "b", amount: number) =>
        limit255(clr[i] + amount);

      actions.forEach((action) => {
        if (action.apply === "mix") {
          clr = mix(clr, action.params[0], action.params[1]);
        } else if (action.apply === "tint") {
          clr = mix(clr, { r: 255, g: 255, b: 255 }, action.params?.[0]);
        } else if (action.apply === "shade") {
          clr = mix(clr, { r: 0, g: 0, b: 0 }, action.params?.[0]);
        } else if (action.apply === "xor") {
          clr = {
            r: clr.r! ^ action.params[0].r,
            g: clr.g! ^ action.params[0].g,
            b: clr.b! ^ action.params[0].b,
          };
        } else if (action.apply === "red") {
          clr.r = colorModifier("r", action.params[0]);
        } else if (action.apply === "green") {
          clr.g = colorModifier("g", action.params[0]);
        } else if (action.apply === "blue") {
          clr.b = colorModifier("b", action.params[0]);
        } else {
          if (action.apply === "hue") {
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-ignore
            action.apply = "spin";
          }

          const tnyClr = tinyColor(clr);
          const fn = tnyClr[action.apply].bind(tnyClr);

          if (!fn) {
            throw new Error("action " + action.apply + " not supported");
          }

          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          clr = (fn as any)(...(action.params || [])).toRgb();
        }
      });

      image.bitmap.data[idx] = clr.r;
      image.bitmap.data[idx + 1] = clr.g;
      image.bitmap.data[idx + 2] = clr.b;
    });

    return image;
  },
};
