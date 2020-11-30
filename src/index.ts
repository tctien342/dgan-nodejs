import * as cv from 'opencv4nodejs';

import * as tf from '@tensorflow/tfjs-node';

import { DetailGAN } from './models/detailGAN';

const dGAN = new DetailGAN()

const noise = tf.randomNormal([1, 784])

console.log(dGAN.generativeModel.summary())
;(dGAN.generativeModel.predict(noise) as tf.Tensor)
  .toFloat()
  .array()
  .then((data: any) => {
    // plt.title('Hello GAN')
    // plt.plot(data[0], data[1])
    // plt.show()
    let out = new cv.Mat([...data[0]], 1)
    cv.imshow('Hello world', out)
    cv.waitKey()
    //   console.log(data)
  })
