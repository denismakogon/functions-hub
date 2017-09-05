// works only with OpenCV 2.x
// there's no plan to support OpenCV 3.x for Go binding
package main

import (
	"github.com/go-opencv/go-opencv/opencv"
	"runtime"
	"path"
)

var facePath = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
var smilePath = "/usr/local/share/OpenCV/haarcascades/haarcascade_smile.xml"


func main() {
	_, currentfile, _, _ := runtime.Caller(0)
	srcImage := opencv.LoadImage(path.Join(path.Dir(currentfile), "../images/smiles/4.jpg"))
	faceCascade := opencv.LoadHaarClassifierCascade(facePath)
	smileCascade := opencv.LoadHaarClassifierCascade(smilePath)

	dstGray := new(opencv.IplImage)
	opencv.CvtColor(srcImage, dstGray, opencv.CV_BGR2GRAY)
	faces := faceCascade.DetectObjects(srcImage)
	for _, face := range faces {
		opencv.Rectangle(srcImage,
			opencv.Point{face.X() + face.Width(), face.Y()},
			opencv.Point{face.X(), face.Y() + face.Height()},
			opencv.ScalarAll(255.0), 1, 1, 0)
		smiles := smileCascade.DetectObjects(dstGray)
		for _, smile := range smiles {
			opencv.Rectangle(srcImage,
				opencv.Point{smile.X() + smile.Width(), smile.Y()},
				opencv.Point{smile.X(), smile.Y() + smile.Height()},
				opencv.ScalarAll(0.0), 1, 1, 0)
		}
	}

	win := opencv.NewWindow("Face Detection")
	win.ShowImage(srcImage)
	opencv.WaitKey(0)
}
