package neural_1
import breeze.linalg._
import breeze.plot._
import breeze.plot.Figure
/**
  * Created by SBARANCZ on 2016-10-17.
  */
object Plotting{
  def plotXY(seqX : Seq[Double],seqY: Seq[Double]) = {
    val fig = Figure()
    val plt = fig.subplot(0)

    plt += plot(seqX,seqY)
    fig.refresh()
  }

  def plotPerceptron(weights:(Double,Double),step:Double) = {
    val inputs1 = (-1.0d).to(1.0,0.01)
    var inputs2=Seq[Double]()
    inputs1.foreach(input1 => {
      inputs2 = inputs2 :+ (step - input1*weights._1)/weights._2
    })
    plotXY(inputs1,inputs2)
  }
}
