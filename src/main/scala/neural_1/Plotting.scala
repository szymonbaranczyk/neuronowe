package neural_1

import breeze.plot.{Figure, _}
/**
  * Created by SBARANCZ on 2016-10-17.
  */
object Plotting{
  def plotXY(seqX : Seq[Double],seqY: Seq[Double]) = {
    val fig = Figure()
    val plt = fig.subplot(0)
    plt.xlabel="wejscie 1"
    plt.ylabel="wejscie 2"
    plt.xlim(-1,2)
    plt.ylim(-1,2)
    plt += plot(seqX,seqY)

  }

  def plotPerceptron(weights:(Double,Double),step:Double) = {
    val inputs1 = (-1.0d).to(2.0,0.01)
    var inputs2=Seq[Double]()
    inputs1.foreach(input1 => {
      inputs2 = inputs2 :+ (step - input1*weights._1)/weights._2
    })
    plotXY(inputs1,inputs2)
  }
}
