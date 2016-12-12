package neural_4

import scala.math._

/**
  * Created by SBARANCZ on 2016-12-12.
  */
case class Neuron(x: Int, y: Int, weights: Vector[Double])

case class SOM(neurons: Vector[Neuron],
               iteration: Int,
               maxIterations: Int,
               maxR: Double,
               minR: Double) {

  def learn(input: Vector[Double]) = {
    val winner = neurons.foldLeft(neurons.head)((n1, n2) => closer(n1, n2, input))
    newSOM(
      neurons.map(n => updateNeuron(n, winner, iteration))
    )
  }

  private def closer(n1: Neuron, n2: Neuron, input: Vector[Double]) = if (manhattanDistance(n1.weights, input) > manhattanDistance(n2.weights, input))
    n2
  else n1

  private def newSOM(newNeurons: Vector[Neuron]) = SOM(newNeurons,
    iteration + 1,
    maxIterations,
    maxR,
    minR)

  private def manhattanDistance(v1: Vector[Double], v2: Vector[Double]) = {
    v1.zip(v2).foldLeft(0.0)((acc, p) => acc + abs(p._1 - p._2))
  }

  private def updateNeuron(n: Neuron, winner: Neuron, iteration: Int) = {
    val nRange = maxR * pow(minR / maxR, iteration / maxIterations)
    if (sqrt(pow(n.x - winner.x, 2) + pow(n.y - winner.y, 2)) <= nRange)
      1
    else 0
    n //TODO
  }
}

object RunSOM {


}
