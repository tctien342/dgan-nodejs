import * as tf from '@tensorflow/tfjs-node';

const L = tf.layers

export class DetailGAN {
  generativeModel = new tf.Sequential()
  discriminatorModel = new tf.Sequential()

  constructor () {
    this.initGenerator()
    this.initDiscriminator()
  }

  initGenerator () {
    this.generativeModel.add(
      L.dense({
        units: 7 * 7 * 256,
        useBias: false,
        inputShape: [784]
      })
    )
    this.generativeModel.add(L.batchNormalization())
    this.generativeModel.add(L.leakyReLU())
    this.generativeModel.add(L.reshape({ targetShape: [7, 7, 256] }))
    this.generativeModel.add(
      L.conv2dTranspose({
        kernelSize: 5,
        filters: 128,
        strides: 1,
        padding: 'same',
        useBias: false
      })
    )
    // Output => 7,7,128
    this.generativeModel.add(L.batchNormalization())
    this.generativeModel.add(L.leakyReLU())
    this.generativeModel.add(
      L.conv2dTranspose({
        kernelSize: 5,
        filters: 64,
        strides: 2,
        padding: 'same',
        useBias: false
      })
    )
    // Output => 14,14,64
    this.generativeModel.add(L.batchNormalization())
    this.generativeModel.add(L.leakyReLU())
    this.generativeModel.add(
      L.conv2dTranspose({
        kernelSize: 5,
        filters: 1,
        strides: 2,
        padding: 'same',
        useBias: false,
        activation: 'tanh'
      })
    )
    // Output => 28,28,1
  }
  initDiscriminator () {}
}
