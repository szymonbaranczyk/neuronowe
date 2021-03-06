package neural_2

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO


object BAWImgReader extends App{
  //  val files = getListOfFiles(getClass.getResource("/img").getPath)
  //  val file = new File("/result.txt")
  //  file.createNewFile()
  //  val bw = new BufferedWriter(new FileWriter(file))
  //  files.foreach(f =>{
  //    bw.write(f.getName.charAt(0))
  //    bw.write(" ")
  //    getImageList(ImageIO.read(f)) foreach {
  //      case true => bw.write("1")
  //        bw.write(" ")
  //      case false => bw.write("0")
  //        bw.write(" ")
  //    }
  //    bw.write("\n")
  //  })
  //  bw.close()
  def getImageList(bi: BufferedImage): List[Double] = {
    var list = List[Color]()
    for(i<- bi.getHeight-1 to 0 by -1){
      for(j<- bi.getWidth-1 to 0 by -1) {
        list = new Color(bi.getRGB(j, i)) +: list
      }
    }
    list.map(c => c.getRed.toDouble / 256.0)
  }
  def getListOfFiles(dir: String):List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  def saveImage(seq: Seq[Double]) = {
    val size = 28
    var k = 0
    val img = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
    for (i <- 0 to size) {
      for (j <- 0 to size) {
        val p = (seq(k) * 256).toInt
        k += 1
        img.setRGB(size, size, (p << 24) | (p << 16) | (p << 8) | p)
      }
    }
    val f = new File("C:\\Output.png")
    ImageIO.write(img, "png", f)
  }
}
