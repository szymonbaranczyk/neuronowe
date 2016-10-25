package neural_1

import java.io.FileReader
import java.util.{Locale, Scanner}

import scala.io.StdIn
import scala.util.Random
/**
  * Created by SBARANCZ on 2016-10-05.
  */
class Perceptron(var weights: Seq[Double], α: Double, activationFunction: (Double) => Int, expectedResult: Seq[Double] => Double, toTest: Seq[Seq[Double]], adalineMode: Boolean, adalineErrorLimit: Double) {
  def excitation(values: Seq[Double]) = values.zipWithIndex.map{case (v,i) => v*weights(i)}.sum

  def result(tested: Seq[Double]) = activationFunction(excitation(tested))

  def recalculateWeights(tested: Seq[Double]): Unit = {
    val δ = error(tested)
    weights=weights.zipWithIndex.map{ case (w,i) => w + α * δ * tested(i)}
  }

  def error(tested: Seq[Double]): Double = {
    if (adalineMode) expectedResult(tested) - excitation(tested)
    else expectedResult(tested) - result(tested)
  }

  def iteration() = {
    val shuffledIds = util.Random.shuffle[Int, IndexedSeq](toTest.indices)
    shuffledIds.foreach(id => recalculateWeights(toTest(id)))
    println(weights)
    var returned = true
    if (!adalineMode) {
      toTest.foreach(tested => {
        if (result(tested) != expectedResult(tested)) {
          returned = false
        }
        println(tested + " " + result(tested))
      })
    } else {
      var errorSum= 0.0d//toTest.foldLeft(0.0d)((sum,tested) => sum+Math.pow(error(tested),2))
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

  def answer(seq: Seq[Double]) = {
    activationFunction(excitation(seq))
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
      weights = Seq(Random.nextDouble() * range * 2 - range, Random.nextDouble() * range * 2 - range),
      α = options.getOrElse("learning_factor", 0.01),
      activationFunction = v => if (v < step) {
        neg
      } else 1,
      expectedResult =
        if (learnedFun == 0)
          seq => if (seq(0)==1 && seq(1)==1) 1
          else {
            neg //unipolar or bipolar
          }
        else
          seq => if (seq(0)==1 || seq(1)==1) 1
          else {
            neg
          },
      toTest = Seq(Seq(0, 0), Seq(0, 1), Seq(1, 0), Seq(1, 1)),
      adalineMode = !(ifAdaline == 0),
      adalineErrorLimit = adalineErrorLimit
    )
  p.learn()
  Plotting.plotPerceptron((p.weights(0),p.weights(1)),step)
  var input = ""
  while (input != "stop") {
    println("enter two values, seperated by whitespace, which neuron should be tested against")
    input = StdIn.readLine()
    val s = input.split(" ")
    println(p.answer(Seq(s(0).toDouble, s(1).toDouble)))
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
}