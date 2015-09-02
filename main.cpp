#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
//#include <SerialStream.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <time.h>

using namespace cv;
using namespace std;
//using namespace LibSerial;

//#define PORT "/dev/ttyACM0" //This is system-specific
//SerialStream ardu;


//const static int SENSITIVITY_VALUE = 80;//20
//const static int BLUR_SIZE = 10;//10
int theObject[2] = {0,0};
Rect objectBoundingRectangle = Rect(0,0,0,0);

//int to string helper function
string intToString(int number){

	//this function has a number input and string output
	std::stringstream ss;
	ss << number;
	return ss.str();
}

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}





static void dbread(const string &filename, vector<Mat> &images, vector<int> &labels, char separator = ';'){
	std::ifstream file(filename.c_str(), ifstream::in);

	if (!file){
		string error = "no valid input file";
		CV_Error(CV_StsBadArg, error);
	}

	string line, path, label;
	while (getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, label);
		if (!path.empty() && !label.empty()){
			//images.push_back(imread(path, 0));
			Mat m = imread(path, 0);
			if ( m.empty() )
			{
			     cerr << path << " could not be read!" << endl;
			     return;
			}
			images.push_back(m);
			labels.push_back(atoi(label.c_str()));
		}
	}
}



void fisherFaceTrainer(){
		/*in this two vector we put the images and labes for training*/
		vector<Mat> images;
		vector<int> labels;

		try{
			string filename = "/home/pi/projects/at.txt";
			dbread(filename, images, labels);

			cout << "size of the images is " << images.size() << endl;
			cout << "size of the labes is " << labels.size() << endl;
			cout << "Training begins...." << endl;
		}
		catch (cv::Exception& e){
			cerr << " Error opening the file " << e.msg << endl;
			exit(1);
		}


		Ptr<FaceRecognizer> model = createFisherFaceRecognizer();

		model->train(images, labels);

		int height = images[0].rows;

		model->save("/home/pi/projects/LBPHface.yml");

		cout << "Training finished...." << endl;

		Mat eigenvalues = model->getMat("eigenvalues");
		// And we can do the same to display the Eigenvectors (read Eigenfaces):
		Mat W = model->getMat("eigenvectors");
		// Get the sample mean from the training data
		Mat mean = model->getMat("mean");

	waitKey(10000);
}



void timer(double seconds){

clock_t x;
x=clock()+seconds*CLOCKS_PER_SEC;
while(clock()<x){}

}


int main( int argc, char* argv[] )
{
    bool isColorizeDisp, isFixedMaxDisp;
    int imageMode;
    bool retrievedImageFlags[5];
    string filename;
    bool isVideoReading;

	//start training
    fisherFaceTrainer();

    //loading classifiers
       cout << "start recognizing..." << endl;

       	//load pre-trained data sets
   	       	Ptr<FaceRecognizer>  model = createFisherFaceRecognizer();
   	       	model->load("/home/pi/projects/LBPHface.yml");

   	       	Mat testSample = imread("/home/pi/projects/facerec/s4/2.pgm", 0);
   	       	int img_width = testSample.cols;
   	       	int img_height = testSample.rows;


       	//lbpcascades/lbpcascade_frontalface.xml
	       	string classifier = "/home/pi/opencv-2.4.10/data/lbpcascades/lbpcascade_frontalface.xml";

       	CascadeClassifier face_cascade;
       	string window = "Capture - face detection";

       	if (!face_cascade.load(classifier)){
       		cout << " Error loading file" << endl;
       		return -1;
       	}
   cout<<"classifiers loaded"<<endl;

    cout << "Device opening ..." << endl;

    VideoCapture capture(0); // open the video camera no. 0
    cout << "done." << endl;

    if( !capture.isOpened() )
    {
        cout << "Can not open a capture object." << endl;
        return -1;
    }else{
    	cout<<"capture is opened"<<endl;
    }



	namedWindow(window, 1);
	long count = 0;

	double dWidth=160;
	double dHeight=120;


	    cout << "Frame size : " << dWidth << " x " << dHeight << endl;

        	int camwidth=dWidth;
        	int camheight=dHeight;


        	int imcount=0;

    for(;;)
    {
    	Mat frame,bgrImage;
    	Mat grayImage;
        Mat frame1,frame2;
            	//their grayscale images (needed for absdiff() function)
        Mat grayImage1,grayImage2;
            	//resulting difference image
        Mat differenceImage;
            	//thresholded difference image (for use in findContours() function)
        Mat thresholdImage;

        vector<Rect> faces;
        //Mat frame;
        //Mat graySacleFrame;
        Mat original;
        imcount=imcount++;

             double confidence = 0;

               bool bSuccess = capture.read(frame); // read a new frame from video

                if (!bSuccess) //if not success, break loop
               {
                    cout << "Cannot read a frame from video stream" << endl;
                    break;
               }
                else
                {

        	capture.retrieve(bgrImage, CV_CAP_OPENNI_BGR_IMAGE );
            	imshow( "rgb image", frame );

            //	resize(bgrImage,bgrImage,Size(320,220));
            	line(bgrImage,Point(displayX,0),Point(displayX,camheight),Scalar(110,220,0),2,8);  //
            	line(bgrImage,Point(0,displayY),Point(camwidth,displayY),Scalar(110,220,0),2,8);  //
            	count = count + 1;//count frames;

            	if (!bgrImage.empty()){

            				//clone from original frame
            		original = frame.clone();
imshow("First capture",original);
            		          	                   				//convert image to gray scale and equalize
            		          	                   				cvtColor(original, grayImage, CV_BGR2GRAY);
            		          	                   				//equalizeHist(graySacleFrame, graySacleFrame);

            		          	                   				//detect face in gray image
            		          	                   				face_cascade.detectMultiScale(grayImage, faces, 1.1, 3, 0, cv::Size(90, 90));

            		          	                   				//number of faces detected
            		          	                   				cout << faces.size() << " faces detected" << endl;
            		          	                   				string frameset = std::to_string(count);
            		          	                   				string faceset = std::to_string(faces.size());
            		      	                   				//region of interest
            		          	                   				//cv::Rect roi;

            		          	                   				//person name
            		          	                   				string Pname = "";

            		          	                   			if (!faces.size()==0){

            		          	                   				for (size_t i = 0; i < faces.size(); i++)
            		          	                   				{
            		          	                   					//region of interest
            		          	                   					Rect face_i = faces[i];

            		          	                   					//crop the roi from gray image
            		          	                   					Mat face = grayImage(face_i);

            		          	                   					//resizing the cropped image to suit to database image sizes
            		          	                   					Mat face_resized;
            		          	                   					cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

            		          	                   					//recognizing what faces detected
            		          	                   					int label = -1;
            		          	                   				 double confidence = 0;
            		          	                   				 int identity;
            		          	                   			for(int j=0;j<3;j++){
            		          	                   				model->predict(face_resized, label, confidence);

            		          	                   				//cout << " confidencde " << confidence << endl;
            		          	                   				//cout<<"label"<< label<<endl;

            		          	                   				  //  double relativeDif = CORRECTION_CONSTANT*(double)diff/(double)(face_resized.rows * face_resized.cols);
            		          	                   				    //double similarity = 1.0 - min(max(relativeDif, 0.0), 1.0);

            		          	                   						if(confidence > 1460){
            		          	                   				            identity = label;
            		          	                   				            cout << " confidencde " << confidence << endl;
            		          	                   				            cout<<"label"<< label<<endl;
            		          	                   				            break;
            		          	                   				        } else{
            		          	                   				        	identity=100;
            		          	                   				        }
            		          	                   				      }
            		          	                   					//cout << " confidencde " << confidence << endl;

            		          	                   					//if (confidence<150){
            		          	                   					//	label=10;
            		          	                   					//}


            		          	                   					//drawing green rectagle in recognize face
            		          	                   					//rectangle(grayImage, face_i, CV_RGB(0, 255, 0), 1);
            		          	                   					Point center(faces[i].x+faces[i].width*0.5,faces[i].y+faces[i].height*0.5);
            		          	                   					ellipse(original,center,Size(faces[i].width*0.5,faces[i].height*0.5),0,0,360,Scalar(255,0,255),4,8,0);
            		          	                   					ellipse(original,Point(center.x,center.y),Size(10,10),0,0,360,Scalar(255,0,255),4,8,0);



            		          	                   					string text = "Detected";
            		          	                   					cout<<"The label is"<<label<<endl;
            		          	                   					if (identity == 0){
            		          	                   						//string text = format("Person is  = %d", label);
            		          	                   						//Pname = "";
            		         	                   						 cout<<"Hello 0"<<endl;
            		          	                   					}else if (identity== 1){
            		          	                   					   cout<<"Hello 1"<<endl;
            		          	                   					}else if (identity == 2){
            		          	                   						cout<<"Hello 2"<<endl;
            		          	                   					   /// timer(2);
            		          	                   					}else if (identity ==3){
            		          	                   						cout<<"Hello 3"<<endl;
            		          	                   					   // timer(2);
            		          	                   					}else{
            		          	                   						 string Pname = "unknown";
            		          	                   						 cout<<"You must leave!!"<<endl;

            		          	                   						 cout<<"Shoot"<<endl;
          		          	                   					}

            		          	                  				imshow(window, original);

            		          	                   					int pos_x = std::max(face_i.tl().x - 10, 0);
            		          	                   					int pos_y = std::max(face_i.tl().y - 10, 0);

            		          	                   					//cout<<"position x"<<pos_x<<endl;
            		          	                   					//cout<<"porition y"<<pos_y<<endl;

            					//name the person who is in the image
            					putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
            					//cv::imwrite("E:/FDB/"+frameset+".jpg", cropImg);
                 				imwrite(format("/home/pi/projects/data/pic_[%d].jpg",imcount),original);

            				}//for

            		          	                   			}
            				putText(original, "Frames: " + frameset, Point(30, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
            				putText(original, "Person: " + Pname, Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
            				//display to the winodw
            				imshow(window, original);

            				//cout << "model infor " << model->getDouble("threshold") << endl;

            			} //if


    }//else

    }//for
    return 0;
}

