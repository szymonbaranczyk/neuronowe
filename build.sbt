
name := "neuronowe"

version := "0.1"

scalaVersion := "2.11.8"

scalacOptions ++= Seq("-feature", "-language:postfixOps")

libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalanlp" %% "breeze-natives" % "0.12",
  "org.scalanlp" %% "breeze-viz" % "0.12"
)


resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"