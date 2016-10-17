package neural_1

import java.io.FileReader
import java.util.{Locale, Scanner}

import scala.io.StdIn
import scala.util.Random
import breeze.plot._
/**
  * Created by SBARANCZ on 2016-10-05.
  */
class Perceptron(var weights: (Double, Double), α: Double, activationFunction: (Double) => Int, expectedResult: (Int, Int) => Int, toTest: Seq[(Int, Int)], adalineMode: Boolean, adalineErrorLimit: Double) {
  def excitation(v1: Double, v2: Double) = v1 * weights._1 + v2 * weights._2

  def result(tested: (Int, Int)) = activationFunction(excitation(tested._1, tested._2))

  def recalculateWeights(tested: (Int, Int)): Unit = {
    val δ = error(tested)
    weights = (weights._1 + α * δ * tested._1, weights._2 + α * δ * tested._2)
  }

  def error(tested: (Int, Int)): Double = {
    if (adalineMode) expectedResult(tested._1, tested._2) - excitation(tested._1, tested._2)
    else expectedResult(tested._1, tested._2) - result(tested)
  }

  def iteration() = {
    val shuffledIds = util.Random.shuffle[Int, IndexedSeq](toTest.indices)
    shuffledIds.foreach(id => recalculateWeights(toTest(id)))
    println(weights)
    var returned = true
    if (!adalineMode) {
      shuffledIds.foreach(id => {

        if (result(toTest(id)) != expectedResult(toTest(id)._1, toTest(id)._2)) {
          returned = false
        }
        println(toTest(id) + " " + result(toTest(id)))
      })
    } else {
      var errorSum = 0.0d
      shuffledIds.foreach(id => {
        errorSum += Math.pow(error(toTest(id)),2)
      })
      errorSum = errorSum / toTest.length
      if (errorSum > adalineErrorLimit) {
        returned = false
      }
      println(errorSum)
    }
    returned
  }

  def learn() = {
    var stop = false
    var i = 0
    while (!stop) {
      stop = iteration()
      i = i + 1
    }
    println(s"$i iterations were required")
  }

  def answer(x1: Double, x2: Double) = {
    activationFunction(excitation(x1, x2))
  }
}

object Runner extends App {
  val options = readOptions()
  val range = options.getOrElse("weights_range", 0.5)
  val step = -options.getOrElse("bias", -0.5)
  val actFun = options.getOrElse("activation_function", 0)
  val learnedFun = options.getOrElse("learned_function", 0)
  val ifAdaline = options.getOrElse("if_adaline", 0)
  val adalineErrorLimit = options.getOrElse("adaline_error_limit", 0.1)
  val neg = if (actFun == 0) 0 else -1
  val p =
    new Perceptron(
      weights = (Random.nextDouble() * range * 2 - range, Random.nextDouble() * range * 2 - range),
      α = options.getOrElse("learning_factor", 0.01),
      activationFunction = v => if (v < step) {
        neg
      } else 1,
      expectedResult =
        if (learnedFun == 0)
          (v1, v2) => if (v1 == 1 && v2 == 1) 1
          else {
            neg //unipolar or bipolar
          }
        else
          (v1, v2) => if (v1 == 1 || v2 == 1) 1
          else {
            neg
          },
      toTest = Seq((0, 0), (0, 1), (1, 0), (1, 1)),
      adalineMode = !(ifAdaline == 0),
      adalineErrorLimit = adalineErrorLimit
    )
  p.learn()
  var input = ""
  while (input != "stop") {
    println("enter two values, seperated by whitespace, which neuron should be tested against")
    input = StdIn.readLine()
    val s = input.split(" ")
    println(p.answer(s(0).toDouble, s(1).toDouble))
  }

  def readOptions() = {
    val fileReader = new FileReader("params.txt")
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
}