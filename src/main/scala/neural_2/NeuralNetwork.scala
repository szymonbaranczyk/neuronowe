package neural_2
import breeze.linalg._

class NeuralLayer(var weights:DenseMatrix[Double]) {
  var z = DenseMatrix(0.0)
  var a = DenseMatrix(0.0)
  def runOnce(input: DenseMatrix[Double]) = {
    z = input.t * weights
    a = z.map(d => sigmoid(d)).t
    a
  }
  def sigmoid(x: Double): Double = {
    1.0 / (1.0 + math.pow(math.E, -x))
  }

}

//input is vertical vector (one column DenseMatrix)
class NeuralNetwork(inputLength:Int, hiddenLength:Int, outputLength:Int, range:Double){
  var hidden = new NeuralLayer(randomMatrix(range,inputLength,hiddenLength))
  var output = new NeuralLayer(randomMatrix(range,hiddenLength,outputLength))
  def runOnce(input:DenseMatrix[Double]): DenseMatrix[Double] = {
    output.runOnce( hidden.runOnce(input) )
  }
  def costFunction(input:DenseMatrix[Double],expectedOutput:DenseMatrix[Double]) = {
    val out = runOnce(input)
    val deltaOut = (-(expectedOutput - out)) :* (output.z.map(d => sigmoidPrime(d))).t
    val costW2 = hidden.a * deltaOut.t
    val deltaHidden = (deltaOut.t * output.weights.t) :* hidden.z.map(d => sigmoidPrime(d))
    val costW1 = input * deltaHidden
    (costW2,costW1)
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
}
object Test extends App{
  val nn = new NeuralNetwork(2,3,4,0.5)
  nn.costFunction(DenseMatrix((0.5),(0.2)),DenseMatrix((0.5),(0.2),(0.5),(0.2)))
}
