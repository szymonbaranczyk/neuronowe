package neural_2

import java.io.{BufferedWriter, File, FileReader, FileWriter}
import java.util.Scanner

import scala.collection.mutable.ListBuffer
import scala.util.Random

/**
  * Created by SBARANCZ on 2016-11-15.
  */
object ExampleMultiplier extends App {
  val i = 0
  val file = new File(getClass.getResource("/result.txt").getPath)
  val mutFile = new File("/result2.txt")
  mutFile.createNewFile()
  val bw = new BufferedWriter(new FileWriter(mutFile))
  val imgs: ListBuffer[(Int, List[Int])] = ListBuffer()
  val r = new Random()
  for (i <- 0 to 30 by 1) {
    val sc = new Scanner(new FileReader(file))
    while (sc.hasNext) {
      val line = sc.nextLine()
      val tokens = line.split(" ")
      val out = tokens.head.toInt
      val img = tokens.tail.map(t => t.toInt)
      bw.write(out.toString)
      bw.write(" ")
      img.foreach(i => {
        bw.write(if (r.nextDouble() > 0.1) {
          i.toString
        } else {
          if (i == 1) "0" else "1"
        })
        bw.write(" ")
      })
      bw.write("\n")
    }
    sc.close()
  }
  bw.close()

}
