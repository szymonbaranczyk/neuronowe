package neural_2
import java.io.{File, FileReader}
import java.util.Scanner

import breeze.linalg._
import breeze.numerics._

import scala.collection.mutable.ListBuffer
import scala.util.Random

class NeuralLayer(var weights:DenseMatrix[Double]) {
  var z = DenseMatrix(0.0)    //input
  var a = DenseMatrix(0.0)    //output
  def runOnce(input: DenseMatrix[Double]) = {
    z = input * weights
    a = z.map(d => sigmoid(d))
    a
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
deltaHidden 1x3 (something's wrong with transpositions)
costW1 2x3 (same as weight matrix for hiddenLayer)
 */
class NeuralNetwork(inputLength:Int, hiddenLength:Int, outputLength:Int, range:Double, learning:Double){
  var hidden = new NeuralLayer(randomMatrix(range,inputLength,hiddenLength))
  var output = new NeuralLayer(randomMatrix(range,hiddenLength,outputLength))
  def runOnce(input:DenseMatrix[Double]): DenseMatrix[Double] = {
    output.runOnce( hidden.runOnce(input) )
  }
  def costFunctionPrime(input:DenseMatrix[Double],expectedOutput:DenseMatrix[Double]) = {
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
    (costW2,costW1)
  }
  def costFunction(input:DenseMatrix[Double],expectedOutput:DenseMatrix[Double]) = {
    val out = runOnce(input)
    0.5 * sum(breeze.numerics.pow(expectedOutput-out,2.0))
  }
  def randomMatrix(range:Double,x:Int,y:Int):DenseMatrix[Double] = {
    val dist = breeze.stats.distributions.Uniform(-range,range)
    DenseMatrix.rand(x,y,dist)
  }
  def sigmoid(x: Double): Double = {
    1.0 / (1.0 + math.pow(math.E, -x))
  }
  def sigmoidPrime(x:Double): Double ={
    sigmoid(x) * (1 - sigmoid(x))
  }
  def train(examples : Seq[(DenseMatrix[Double],DenseMatrix[Double])], endCheck: NeuralNetwork => Boolean ):Boolean = {
    var stop=false
    var i=0
    while(!stop) {
      examples.foreach(matWithIndex => {
          val pair = costFunctionPrime(matWithIndex._1, matWithIndex._2)
          hidden.weights = hidden.weights - pair._2 * learning
          output.weights = output.weights - pair._1 * learning
      })
      i+=1
      stop = endCheck(this)
    }
    stop
  }
  def trainStep(matWithIndex:(DenseMatrix[Double],DenseMatrix[Double])) = {
    val pair = costFunctionPrime(matWithIndex._1, matWithIndex._2)
    hidden.weights = hidden.weights - pair._2 * learning
    output.weights = output.weights - pair._1 * learning
  }
}
object Test extends App{
  val nn = new NeuralNetwork(70,70,10,0.5,0.1)
  val file = new File(getClass.getResource("/result.txt").getPath)
  val sc = new Scanner(new FileReader(file))
  var trainSet: ListBuffer[(DenseMatrix[Double], DenseMatrix[Double])] = ListBuffer()
  while(sc.hasNext){
    val line = sc.nextLine()
    val tokens = line.split(" ")
    val out = tokens.head.toInt
    val outMatrix = DenseMatrix((List.fill(out)(0.0) :+ 1.0) ::: List.fill(9-out)(0.0))
    val img = DenseMatrix(tokens.tail.map( z => z.toDouble))
    trainSet += ((img,outMatrix))
  }
  val rand = new Random()
  var i=0
  var testList:List[(DenseMatrix[Double], DenseMatrix[Double])] = List()
  while(i<60) {
    val random_index = rand.nextInt(trainSet.length)
    testList = testList :+ trainSet(random_index)
    trainSet.remove(random_index)
    i+=1
  }
  val shuffledIds = util.Random.shuffle[Int, IndexedSeq](trainSet.indices)
  val shuffledTrainSet: ListBuffer[(DenseMatrix[Double], DenseMatrix[Double])] = ListBuffer()
  shuffledIds.foreach(id => shuffledTrainSet += trainSet(id))
  nn.train(trainSet, test(testList = testList))



  var smallestError = 10000.0
  var noChangeTicks =0
  def test(testList:Seq[(DenseMatrix[Double], DenseMatrix[Double])])(nn:NeuralNetwork):Boolean = {
    var cost=0.0
    testList.foreach(tuple => cost += nn.costFunction(tuple._1,tuple._2))
    if(smallestError > cost) {
      smallestError = cost
      noChangeTicks=0
    }else{
      noChangeTicks+=1
    }
    println(cost)
    noChangeTicks > 10
  }
}