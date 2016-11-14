package neural_2

import java.awt.Color
import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.{BufferedWriter, File, FileWriter}
import javax.imageio.ImageIO


object BAWImgReader extends App{
  val files = getListOfFiles(getClass.getResource("/img").getPath)
  val file = new File("/result.txt")
  file.createNewFile()
  val bw = new BufferedWriter(new FileWriter(file))
  files.foreach(f =>{
    bw.write(f.getName.charAt(0))
    bw.write(" ")
    getImageList(ImageIO.read(f)) foreach {
      case true => bw.write("1")
        bw.write(" ")
      case false => bw.write("0")
        bw.write(" ")
    }
    bw.write("\n")
  })
  bw.close()
  def getImageList(bi:BufferedImage):List[Boolean]={
    var list = List[Color]()
    for(i<- bi.getHeight-1 to 0 by -1){
      for(j<- bi.getWidth-1 to 0 by -1) {
        list = new Color(bi.getRGB(j, i)) +: list
      }
    }
    list.map( c => c.getRed>20)
  }
  def getListOfFiles(dir: String):List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }
}
