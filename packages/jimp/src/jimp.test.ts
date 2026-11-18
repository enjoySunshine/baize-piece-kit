import { describe, expect, test } from "vitest";
import { Jimp } from "./index.js";
import { getTestImagePath } from "@jimp/test-utils";

const sharpenKernel = [
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0],
];

function pixelHex(image: Jimp, x: number, y: number) {
  const p = (y * image.bitmap.width + x) * 4;
  return [0, 1, 2, 3]
    .map((o) => image.bitmap.data[p + o]!.toString(16).padStart(2, "0"))
    .join("");
}

describe("hasAlpha", () => {
  test("image with no alpha", async () => {
    const image = await Jimp.read(getTestImagePath("cops.jpg"));
    expect(image.hasAlpha()).toBe(false);
  });

  test("image with alpha", async () => {
    const image = await Jimp.read(getTestImagePath("dice.png"));
    expect(image.hasAlpha()).toBe(true);
  });
});

describe("sharpen then radius-3 soft edge pipeline", () => {
  test("produces the exact dice pixels and survives a png round trip", async () => {
    const original = await Jimp.read(getTestImagePath("dice.png"));
    expect(pixelHex(original, 200, 150)).toBe("4c4c8ee8");
    expect(pixelHex(original, 400, 300)).toBe("fff2f2ff");

    const onlySharpen = await Jimp.read(getTestImagePath("dice.png"));
    onlySharpen.convolution(sharpenKernel);
    expect(pixelHex(onlySharpen, 200, 150)).toBe("4d4d8ce8");

    const onlyBlur = await Jimp.read(getTestImagePath("dice.png"));
    onlyBlur.blur(3);
    expect(pixelHex(onlyBlur, 200, 150)).toBe("4e4e92ec");
    expect(pixelHex(onlyBlur, 400, 300)).toBe("c69898f9");

    const image = await Jimp.read(getTestImagePath("dice.png"));
    image.convolution(sharpenKernel);
    image.blur(3);

    expect(pixelHex(image, 200, 150)).toBe("4d4d92ec");
    expect(pixelHex(image, 400, 300)).toBe("be9393f9");

    // The transparent margin kept for the board stays fully transparent.
    const { width, height } = image.bitmap;
    expect(pixelHex(image, 0, 0)).toBe("00000000");
    expect(pixelHex(image, width - 1, 0)).toBe("00000000");
    expect(pixelHex(image, 0, height - 1)).toBe("00000000");
    expect(pixelHex(image, width - 1, height - 1)).toBe("00000000");

    const roundTrip = await Jimp.read(await image.getBuffer("image/png"));
    expect(pixelHex(roundTrip, 200, 150)).toBe("4d4d92ec");
    expect(pixelHex(roundTrip, 400, 300)).toBe("be9393f9");
  });

  test("leaves the mask.png selection ring transparent corners intact", async () => {
    const image = await Jimp.read(getTestImagePath("mask.png"));
    image.convolution(sharpenKernel);
    image.blur(3);

    const { width, height } = image.bitmap;
    expect(pixelHex(image, 0, 0)).toBe("000000ff");
    expect(pixelHex(image, width - 1, height - 1)).toBe("000000ff");
  });
});
