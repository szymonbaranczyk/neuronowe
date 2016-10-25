package neural_2
import breeze.linalg._

class NeuralNetwork(val weights:DenseMatrix[Double]) {
  def runOnce(input: DenseVector[Double]) = {
    input.t * weights
  }
}
