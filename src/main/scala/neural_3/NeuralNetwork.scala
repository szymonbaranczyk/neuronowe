package neural_3

import java.io.{BufferedWriter, File, FileReader, FileWriter}
import java.util.{Locale, Scanner}
import javax.imageio.ImageIO

import breeze.linalg._
import breeze.plot._

import scala.collection.mutable.ListBuffer
import scala.io.StdIn
import scala.util.Random

class NeuralLayer(var weights: DenseMatrix[Double], var bestWeights: DenseMatrix[Double]) {
  var z = DenseMatrix(0.0)
  //input
  var a = DenseMatrix(0.0)

  //output
  def runOnce(input: DenseMatrix[Double]) = {
    z = input * weights
    a = z.map(d => sigmoid(d))
    a
  }

  def runBest(input: DenseMatrix[Double]) = {
    (input * bestWeights).map(d => sigmoid(d))
  }

  def sigmoid(x: Double): Double = {
    1.0 / (1.0 + math.pow(math.E, -x))
  }

}

//input is vertical vector (one column DenseMatrix)
//example size for network with 2 inputs, 3 hidden nodes and 4 outputs:
/*
input 2x1
out 4x1
deltaOut 4x1
costW2 3x4 (same as weight matrix for outputLayer)
deltaHidden 3x1
costW1 2x3 (same as weight matrix for hiddenLayer)
 */
class NeuralNetwork(inputLength: Int, hiddenLength: Int, outputLength: Int, range: Double, learning: Double, momentum: Double, dropout: Double, regularization: Double) {
  val hidInit = randomMatrix(range, inputLength, hiddenLength)
  val outInit = randomMatrix(range, hiddenLength, outputLength)
  var hidden = new NeuralLayer(hidInit, hidInit.copy)
  var output = new NeuralLayer(outInit, outInit.copy)
  var smallestError = 100000.0
  var noChangeTicks = 0
  var lastPair: Option[(DenseMatrix[Double], DenseMatrix[Double])] = None
  val errorValues = ListBuffer[Double]()

  def costFunction(input: DenseMatrix[Double], expectedOutput: DenseMatrix[Double]) = {
    val out = runOnce(input)
    0.5 * sum(breeze.numerics.pow(expectedOutput - out, 2.0))
  }

  def randomMatrix(range: Double, x: Int, y: Int): DenseMatrix[Double] = {
    val dist = breeze.stats.distributions.Uniform(range - 0.05, range + 0.05)
    DenseMatrix.rand(x, y, dist)
  }

  def train(examples: Seq[(DenseMatrix[Double], DenseMatrix[Double])], endCheck: NeuralNetwork => Boolean): Int = {
    var stop = false
    var i = 0
    val r = new Random
    while (!stop) {
      shuffle(examples).foreach(mat => {
          val pair = costFunctionPrime(mat._1, mat._2)
        pair._1.map(d => if (r.nextDouble() > dropout) {
          d
        } else {
          0
        })
        hidden.weights = (1 - learning * regularization) * hidden.weights - pair._2 * learning
        output.weights = (1 - learning * regularization) * output.weights - pair._1 * learning
          lastPair match {
            case Some(p) => hidden.weights = hidden.weights - (p._2 * learning * momentum)
              output.weights = output.weights - (p._1 * learning * momentum)
            case None =>
          }
          lastPair = Some(pair)
      })
      i += 1
      stop = endCheck(this)
    }
    i
  }

  def shuffle(seq: Seq[(DenseMatrix[Double], DenseMatrix[Double])]) = {
    val rand = new Random()
    val shuffledTrainSet: ListBuffer[(DenseMatrix[Double], DenseMatrix[Double])] = ListBuffer()
    val shuffledIds = util.Random.shuffle[Int, IndexedSeq](seq.indices)
    shuffledIds.foreach(id => shuffledTrainSet += seq(id))
    shuffledTrainSet
  }

  def costFunctionPrime(input: DenseMatrix[Double], expectedOutput: DenseMatrix[Double]) = {
    val out = runOnce(input)
    // output.z.map(d => sigmoidPrime(d)) is f'(z) - rate of how much changing 'z' will affect the output.
    // (-(expectedOutput - out)) is a value by which output should be changed - it is also derivative of used error function
    // 1/2*(expectedOutput - out)^2; by multiplying it by f'(z) we can get gradient of activation (or hiddenLayer
    // output multiplied by outputLayer weights) descent. This multiplication applies gradient, and can
    // be understood like this - if activation function would be increasing and output would be too big we would decrease
    // input; conversely, if activation function would be decreasing and output would be too big we would increase
    // input
    val deltaOut = (-(expectedOutput - out)) :* output.z.map(d => sigmoidPrime(d))
    // we multiply it by activation - this way we get value by which weights of output layer should be changed - this
    // will cause gradient descent
    val costW2 = hidden.a.t * deltaOut
    // (deltaOut * output.weights.t) is a value by which output of hidden layer should be changed
    val deltaHidden = (deltaOut * output.weights.t) :* hidden.z.map(d => sigmoidPrime(d))
    val costW1 = input.t * deltaHidden
    (costW2, costW1)
  }

  def runOnce(input: DenseMatrix[Double]): DenseMatrix[Double] = {
    output.runOnce(hidden.runOnce(input))
  }

  def runBest(input: DenseMatrix[Double]): DenseMatrix[Double] = {
    output.runBest(hidden.runBest(input))
  }

  def sigmoidPrime(x: Double): Double = {
    sigmoid(x) * (1 - sigmoid(x))
  }

  def sigmoid(x: Double): Double = {
    1.0 / (1.0 + math.pow(math.E, -x))
  }

  def trainStep(matWithIndex: (DenseMatrix[Double], DenseMatrix[Double])) = {
    val pair = costFunctionPrime(matWithIndex._1, matWithIndex._2)
    hidden.weights = hidden.weights - pair._2 * learning
    output.weights = output.weights - pair._1 * learning
  }
}

object NeuralNetwork {
  val fig = Figure()

  def plotter(errorValues: ListBuffer[Double]) = {
    fig.clear
    val plt = fig.subplot(0)
    plt.xlabel = "epoki"
    plt.ylabel = "e"
    plt.xlim(0, errorValues.length)
    plt.ylim(0, errorValues.max)
    plt += plot(errorValues.indices.map(x => x.toDouble), errorValues)
    fig.refresh()
  }
}

object Test extends App {
  val options = readOptions()
  val range = options.getOrElse("weights_range", 0.5)
  val hiddenLayerSize = options.getOrElse("hidden_layer_size", 70.0).toInt
  val momentum = options.getOrElse("momentum", 0.0)
  val learningFactor = options.getOrElse("learning_factor", 0.1)
  val dropout = options.getOrElse("dropout", 0.5)
  val regularization = options.getOrElse("regularization", 0.0)
  val nn = new NeuralNetwork(784, hiddenLayerSize, 784, range, learningFactor, momentum, dropout, regularization)
  val trainingSetSize = 5000
  val trainSet = skip(Mnist.trainDataset.imageReader.imagesAsVectors.map(v => v.map(d => d)).map(v => (v.asDenseMatrix, v.asDenseMatrix)), 100)
  val testList = skip(Mnist.testDataset.imageReader.imagesAsVectors.map(v => v.map(d => d)).map(v => (v.asDenseMatrix, v.asDenseMatrix)), 100)
  println("size: " + trainSet.length)

  val start = System.currentTimeMillis()
  val iterations = nn.train(trainSet, test(testList = testList))
  val end = System.currentTimeMillis()
  println(iterations + " iterations needed")
  var count = testList.count(t => {
    nn.runBest(t._1).toArray.zip(t._2.toArray).forall(pair => Math.abs(pair._1 - pair._2) < 0.1)
  })
  println("errors: " + count + "/" + testList.size + "=" + ((testList.size.toDouble - count.toDouble) / testList.size.toDouble))
  println("time elapsed: " + (end - start))
  persistNetwork(nn)
  runNetwork(nn)

  def runNetwork(nn: NeuralNetwork) = {
    var input = ""
    while (input != "stop") {
      println("enter file path")
      input = StdIn.readLine()
      val file = new File(input)
      val result: List[Double] = BAWImgReader.getImageList(ImageIO.read(file))
      BAWImgReader.saveImage(nn.hidden.runBest(DenseMatrix(result)).toArray.toList, "hidden.png", 10, 10)
      BAWImgReader.saveImage(nn.runBest(DenseMatrix(result)).toArray.toList, "output.png", 28, 28)
    }
  }

  def test(testList: Seq[(DenseMatrix[Double], DenseMatrix[Double])])(nn: NeuralNetwork): Boolean = {
    var cost = 0.0
    testList.foreach(tuple => cost += nn.costFunction(tuple._1, tuple._2))
    nn.errorValues += cost
    NeuralNetwork.plotter(nn.errorValues)
    if (nn.smallestError > cost) {
      nn.smallestError = cost
      nn.noChangeTicks = 0
      nn.hidden.bestWeights = nn.hidden.weights
      nn.output.bestWeights = nn.output.weights
    } else {
      nn.noChangeTicks += 1
    }
    println(cost)
    nn.noChangeTicks > 5 || cost < 1
  }

  def persistNetwork(nn: NeuralNetwork) = {
    val file = new File("C:/network.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    for (j <- 0 until nn.hidden.bestWeights.cols) {
      for (i <- 0 until nn.hidden.bestWeights.rows) {
        bw.write(nn.hidden.bestWeights.valueAt(i, j) + " ")
      }
      bw.write("\n")
    }
    val file2 = new File("C:/network2.txt")
    val bw2 = new BufferedWriter(new FileWriter(file2))
    for (j <- 0 until nn.output.bestWeights.cols) {
      for (i <- 0 until nn.output.bestWeights.rows) {
        bw2.write(nn.output.bestWeights.valueAt(i, j) + " ")
      }
      bw2.write("\n")
    }



    bw.close()
    bw2.close()
  }

  def readOptions() = {
    val fileReader = new FileReader(getClass.getResource("/params.txt").getPath)
    val sc = new Scanner(fileReader).useLocale(Locale.US)
    var options = Map.empty[String, Double]
    while (sc.hasNext) {
      val key = sc.next()
      val value = sc.nextDouble()
      options = options + (key -> value)
      sc.nextLine()
    }
    options
  }

  def skip[A](l: Seq[A], n: Int) = {
    l.zipWithIndex.collect { case (e, i) if ((i + 1) % n) == 0 => e }
  }
}