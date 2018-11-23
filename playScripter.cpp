#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
//#include <opencv2/video/tracking.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>

static cv::Mat Image;
static cv::Rect2d BBox;
static std::string WindowName;
static bool Paused;
static bool SelectObject = false;
static bool StartSelection = false;

static const char* Keys =
{ "{@tracker_algorithm | | tracker algorithm }"
"{@video_name        | | video name        }"
"{help h usage| |print this message   }"
};

static void Help(void)
{
	std::cout << "\nThis example shows the functionality of \"Long-term optical tracking API\""
		"-- pause video [p] and draw a bounding box around the target to start the tracker\n"
		"Call:\n"
		"./tracker <tracker_algorithm> <video_name>\n"
		<< std::endl;

	std::cout << "\n\nHot Keys: \n"
		"\tq - quit the program\n"
		"\tp - pause video\n";
}

static void OnMouse(int event, int x, int y, int, void*)
{
	if (!SelectObject)
	{
    switch (event)
		{
		case cv::EVENT_LBUTTONDOWN:
			//set origin of the bounding box
			StartSelection = true;
			BBox.x = x;
			BBox.y = y;
			break;
		case cv::EVENT_LBUTTONUP:
			//sei with and height of the bounding box
			BBox.width = std::abs(x - BBox.x);
			BBox.height = std::abs(y - BBox.y);
			Paused = false;
			SelectObject = true;
			break;
		case cv::EVENT_MOUSEMOVE:
			if (StartSelection)
			{
				//draw the bounding box
				cv::Mat currentFrame;
				Image.copyTo(currentFrame);
				cv::rectangle(currentFrame, cv::Point(BBox.x, BBox.y), cv::Point(x, y), cv::Scalar(255, 0, 0), 2, 1);
				cv::imshow(WindowName, currentFrame);
			}
			break;
		}
	}
}

cv::Ptr<cv::Tracker> createTracker(cv::String tracker_algorithm) {
  cv::Ptr<cv::Tracker> tracker;
  if(tracker_algorithm == "BOOSTING"){
    cv::TrackerBoosting::Params params;
    std::cout << "params.featureSetNumFeatures:\nstd: " << params.featureSetNumFeatures;
    params.featureSetNumFeatures = 1050;//1450;
    std::cout << "\ncur: " << params.featureSetNumFeatures;

    std::cout << "\nparams.samplerSearchFactor:\nstd: " << params.samplerSearchFactor;
    params.samplerSearchFactor = 1.8f;//2.5f;
    std::cout << "\ncur: " << params.samplerSearchFactor << std::endl;
#if (CV_VERSION_MAJOR < 3) || ((CV_VERSION_MAJOR == 3) && CV_VERSION_MINOR <= 2)
    tracker = cv::TrackerBoosting::createTracker(params);
#else
    tracker = cv::TrackerBoosting::create(params);
#endif
  }
  else if(tracker_algorithm == "KCF"){
    cv::TrackerKCF::Params params;
#if (CV_VERSION_MAJOR < 3) || ((CV_VERSION_MAJOR == 3) && CV_VERSION_MINOR <= 2)
    tracker = cv::TrackerKCF::createTracker(params);
#else
    tracker = cv::TrackerKCF::create(params);
#endif
  }
  else if(tracker_algorithm == "MEDIANFLOW"){
    cv::TrackerMedianFlow::Params params;
    params.pointsInGrid = 20;
#if (CV_VERSION_MAJOR < 3) || ((CV_VERSION_MAJOR == 3) && CV_VERSION_MINOR <= 2)
    tracker = cv::TrackerMedianFlow::createTracker(params);
#else
    tracker = cv::TrackerMedianFlow::create(params);
#endif
  }
  else if(tracker_algorithm == "MIL"){
    cv::TrackerMIL::Params params;
    params.samplerSearchWinSize = 90;
    params.samplerInitMaxNegNum = 80;
#if (CV_VERSION_MAJOR < 3) || ((CV_VERSION_MAJOR == 3) && CV_VERSION_MINOR <= 2)
    tracker = cv::TrackerMIL::createTracker(params);
#else
    tracker = cv::TrackerMIL::create(params);
#endif
  }
  else if(tracker_algorithm == "TLD"){
    cv::TrackerTLD::Params params;
#if (CV_VERSION_MAJOR < 3) || ((CV_VERSION_MAJOR == 3) && CV_VERSION_MINOR <= 2)
    tracker = cv::TrackerTLD::createTracker(params);
#else
    tracker = cv::TrackerTLD::create(params);
#endif
  }
  else if(tracker_algorithm == "GOTURN"){
    cv::TrackerGOTURN::Params params;
#if (CV_VERSION_MAJOR < 3) || ((CV_VERSION_MAJOR == 3) && CV_VERSION_MINOR <= 2)
    tracker = cv::TrackerGOTURN::createTracker(params);
#else
    tracker = cv::TrackerGOTURN::create(params);
#endif
  }
  else if(tracker_algorithm == "MOSSE"){
#if (CV_VERSION_MAJOR < 3) || ((CV_VERSION_MAJOR == 3) && CV_VERSION_MINOR <= 2)
    // not implemented
#else
    tracker = cv::TrackerMOSSE::create();
#endif
  }
  return tracker;
}

int mainMultitracker(cv::String tracker_algorithm, cv::String video_name){
  cv::Mat frame;
  Paused = false;
  WindowName = "Tracking API: " + tracker_algorithm;
  cv::namedWindow(WindowName, 0);
  cv::setMouseCallback(WindowName, OnMouse, 0);

  std::cout << "\n\nHot Keys: \n"
          "\tq - quit the program\n"
          "\tp - pause video\n";

  //open the capture
  cv::VideoCapture cap;
  cap.open(video_name);
  cv::Ptr<cv::MultiTracker> multiTracker = cv::MultiTracker::create();
  //cv::imshow(WindowName, Image);
  //cv::imwrite("/home/sascha/OpenCV-Apps/PlayScripter/1.jpg", Image);
  bool initialized = false;
  SelectObject = true;
  cv::Mat rectifiedField(600, 1066, CV_8UC3);
  std::vector<cv::Point2f> pointsPixel = {cv::Point2f(1066, 100), cv::Point2f(1066, 500), cv::Point2f(180, 500), cv::Point2f(400, 300)};
//  cv::VideoWriter outputVideo(//"/home/sascha/OpenCV-Apps/PlayScripter/output.mp4",
//                              "appsrc ! autovideoconvert ! v4l2video1h264enc extra-controls=\"encode,h264_level=10,h264_profile=4,frame_level_rate_control_enable=1,video_bitrate=2000000\" ! h264parse ! rtph264pay config-interval=1 pt=96 ! filesink location=file.mp4",
//                              cv::VideoWriter::fourcc('H','2','6','4'),
//                              //-1,
//                              30,
//                              rectifiedField.size());
//  if (!outputVideo.isOpened())
//  {
//          std::cout << "!!! Output video could not be opened" << std::endl;
//          return 0;
//  }
  int imageCounter = 0;
  while (cap.isOpened())
  {
    if (!Paused)
    {
      cap >> frame;
      frame.copyTo(Image);

      if (SelectObject && !initialized)
      {
          //for(cv::Ptr<cv::Tracker> tracker : trackers) {
//                for(int i = 0; i < points.size(); i++){
//                  std::cout << "Initialize " << BBoxes[i] << std::endl;
//                  //initializes the tracker
//                  if (trackers[i]->init(Image, BBoxes[i]))
//                  {
//                          initialized = true;
//                  }
//                }

          //instantiates the specific Tracker
          //std::vector<cv::Point> points = {cv::Point(300, 215), cv::Point(455, 173), cv::Point(542, 151)};
        std::vector<cv::Point> points = {cv::Point(542, 151),
                                         cv::Point(1029, 162),
                                         cv::Point(604, 477),
                                         cv::Point(372, 313)};
                                        //cv::Point(904, 156)};
          int BBlength = 30;
          for(cv::Point point : points){
            //BBoxes.push_back(cv::Rect2d(point.x - 10, point.y - 10, 20, 20));
            multiTracker->add(createTracker(tracker_algorithm), frame, cv::Rect2d(point.x - BBlength / 2, point.y - BBlength / 2, BBlength, BBlength));
          }
          initialized = true;
      }
      if (initialized)
      {
        //for(cv::Ptr<cv::Tracker> tracker : trackers) {
//              for(int i = 0; i < points.size(); i++){
//                //updates the tracker
//                cv::Rect BB;
//                //if (trackers[i]->update(frame, BBoxes[i]))
//                if (trackers[i]->update(frame, BB))
//                {
//                  //std::cout << "Set " << BBoxes[i] << std::endl;
//                  //cv::rectangle(Image, BBoxes[i], cv::Scalar(0, 0, 255), 2, 1);
//                  std::cout << "Set " << BB << std::endl;
//                  cv::rectangle(Image, BB, cv::Scalar(0, 0, 255), 2, 1);
//                }
//                else{
//                        SelectObject = initialized = false;
//                }
//              }
//              return 0;
        std::vector<cv::Point2f> pointsDetected(multiTracker->getObjects().size());
        multiTracker->update(frame);
        // Draw tracked objects
        for(unsigned i = 0; i < multiTracker->getObjects().size(); i++)
        {
          cv::rectangle(Image, multiTracker->getObjects()[i], cv::Scalar(0, 0, 255), 2, 1);
          pointsDetected[i] = (multiTracker->getObjects()[i].tl() + multiTracker->getObjects()[i].br()) / 2;
          std::cout << i << ") Map point " << pointsDetected[i] << " to point " << pointsPixel[i] << std::endl;
        }
        cv::Mat H = cv::findHomography(pointsDetected, pointsPixel);
        //cv::Mat H = cv::findHomography(pointsPixel, pointsDetected);
        std::cout << "H:\n" << H << std::endl;
        cv::warpPerspective(frame, rectifiedField, H, rectifiedField.size());
      }
      imshow(WindowName, rectifiedField);
      std::string counter_str = std::to_string(imageCounter);
      int targetLength = 4;
      std::string filename = std::string(targetLength - counter_str.length(), '0').append(counter_str) + ".jpg";
      imageCounter++;
      cv::imwrite("/home/sascha/OpenCV-Apps/PlayScripter/output/" + filename, rectifiedField);
      //outputVideo.write(rectifiedField);
    }
    char c = static_cast<char>(cv::waitKey(2));
    if (c == 'q')
            break;
    if (c == 'p')
            Paused = !Paused;
  }
  //outputVideo.release();
}

cv::Mat getLineMask(cv::Mat frame, cv::Scalar minColor, cv::Scalar maxColor){
  cv::Mat hsv, binary;
  cv::cvtColor(frame, hsv, CV_BGR2HSV);
//        cv::Vec3d meanColor;
//        int counter = 0;
//        for(int y = 400; y < 500; y++){
//          for(int x = 800; x < 900; x++){
//            meanColor += hsv.at<cv::Vec3b>(y, x);
//            counter++;
//          }
//        }
//        //std::cout << "HSV: " << hsv.at<cv::Vec3b>(400, 800) << std::endl;
//        std::cout << "HSV: " << meanColor / counter << std::endl;
  //cv::Scalar color(38, 144, 117); // field color
//        cv::Scalar lineColor(38, 144, 117);
//        std::cout << "HSV: " << hsv.at<cv::Vec3b>(524, 1254) << std::endl;
  int range = 20;
  //cv::inRange(hsv, cv::Scalar(30, 0, 140), cv::Scalar(40, 255, 255), binary); // line range
  cv::inRange(hsv, minColor, maxColor, binary); // line range
  cv::imwrite("/home/sascha/OpenCV-Apps/PlayScripter/output/0.png", binary);
  cv::Mat se1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2));
  cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, se1);

  cv::Mat se2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2));
  cv::morphologyEx(binary, binary, cv::MORPH_OPEN, se2);
  return binary;
}

cv::Mat getHoughTransform(cv::Mat binary){
  cv::Mat cdst, cdstP;
  cv::cvtColor(binary, cdst, cv::COLOR_GRAY2BGR);
  cdstP = cdst.clone();
  // Standard Hough Line Transform
  std::vector<cv::Vec2f> lines; // will hold the results of the detection
  cv::HoughLines(binary, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection
  // Draw the lines
  for( size_t i = 0; i < lines.size(); i++ )
  {
      float rho = lines[i][0], theta = lines[i][1];
      cv::Point pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a*rho, y0 = b*rho;
      pt1.x = cvRound(x0 + 2000*(-b));
      pt1.y = cvRound(y0 + 2000*(a));
      pt2.x = cvRound(x0 - 2000*(-b));
      pt2.y = cvRound(y0 - 2000*(a));
      cv::line( cdst, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA);
  }
//  // Probabilistic Line Transform
  std::vector<cv::Vec4i> linesP; // will hold the results of the detection
  cv::HoughLinesP(binary, linesP, 1, CV_PI/180, 50, 50, 10 ); // runs the actual detection
  // Draw the lines
  for( size_t i = 0; i < linesP.size(); i++ )
  {
      cv::Vec4i l = linesP[i];
      cv::line( cdstP, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, cv::LINE_AA);
  }
//  // Show results
//  cv::imshow("Source", binary);
//  cv::imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
//  cv::imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
//  cv::waitKey();
  return cdstP;
}

template <typename T>
inline T mapVal(T x, T a, T b, T c, T d)
{
    x = fmax(fmin(x, b), a);
    return c + (d-c) * (x-a) / (b-a);
}

static void colorizeFlow(const cv::Mat &u, const cv::Mat &v, cv::Mat &dst)
{
    double uMin, uMax;
    cv::minMaxLoc(u, &uMin, &uMax, 0, 0);
    double vMin, vMax;
    cv::minMaxLoc(v, &vMin, &vMax, 0, 0);
    uMin = fabs(uMin);
    uMax = fabs(uMax);
    vMin = fabs(vMin);
    vMax = fabs(vMax);
    float dMax = static_cast<float>(fmax(fmax(uMin, uMax), fmax(vMin, vMax)));
    std::cout << "uMin: " << uMin << ", uMax: " << uMax << ", vMin: " << vMin << ", vMax: " << vMax << std::endl;
    std::cout << "dMax: " << dMax << std::endl;

    dst.create(u.size(), CV_8UC3);
    for (int y = 0; y < u.rows; ++y)
    {
        for (int x = 0; x < u.cols; ++x)
        {
            dst.at<uchar>(y,3*x) = 0;
            dst.at<uchar>(y,3*x+1) = (uchar)mapVal(-v.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
            dst.at<uchar>(y,3*x+2) = (uchar)mapVal(u.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
        }
    }
}

void mainOpticalFlow(cv::String video_name){
  cv::Mat prior_frame, prior_frame_gray, frame, frame_gray, flow, colored_flow;
  cv::Mat flowX, flowY;
  Paused = false;
  std::string FlowWindowName = "Flow Window";
  cv::namedWindow(FlowWindowName, 0);
  WindowName = "Video Image";
  cv::namedWindow(WindowName, 0);
  cv::setMouseCallback(WindowName, OnMouse, 0);

  //open the capture
  cv::VideoCapture cap;
  cap.open(video_name);

  cv::Ptr<cv::cuda::FarnebackOpticalFlow> d_calc = cv::cuda::FarnebackOpticalFlow::create();

  std::vector<cv::Point2f> corners;
  int maxCorners = 1000;
  double qualityLevel = 0.01;
  double minDistance = 10;
  while (cap.isOpened())
  {
    cap >> frame;
    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::goodFeaturesToTrack(frame_gray,
                            corners,
                            maxCorners,
                            qualityLevel,
                            minDistance);

    std::cout << "** Number of corners detected: " << corners.size() << std::endl;
    int r = 4;
    for( int i = 0; i < corners.size(); i++ ){
      cv::circle(frame, corners[i], r, cv::Scalar(255, 0, 0), -1);
    }
//    if(!prior_frame.empty()) {
//      cv::cvtColor(prior_frame, prior_frame_gray, CV_BGR2GRAY);

//      cv::cuda::GpuMat prior_frame_gpu(prior_frame_gray), frame_gpu(frame_gray);
//      cv::cuda::GpuMat flow_gpu;
//      d_calc->calc(prior_frame_gpu, frame_gpu, flow_gpu);

//      cv::cuda::GpuMat planes[2];
//      cv::cuda::split(flow_gpu, planes);

//      planes[0].download(flowX);
//      planes[1].download(flowY);
//      //colorizeFlow(flowX, flowY, colored_flow);
//      //cv::normalize(flowX, flowX, 0, 255, cv::NORM_MINMAX);
//      //cv::normalize(flowY, flowY, 0, 255, cv::NORM_MINMAX);
//      float max = .0001;
//      float min = -.0001;
//      flowX.convertTo(flowX, CV_8UC1, 255.0/(max-min), -255.0*min/(max-min));
//      flowY.convertTo(flowY, CV_8UC1, 255.0/(max-min), -255.0*min/(max-min));
//      //std::cout << "types: " << flowX.type() << ", " << flowY.type() << std::endl;

//      std::vector<cv::Mat> channels = {cv::Mat::zeros(frame.size(), CV_8UC1), flowX, flowY};
//      //std::vector<cv::Mat> channels = {flowX, flowX, flowX};
//      merge(channels, colored_flow);

//      cv::imshow(FlowWindowName, colored_flow);
//      cv::imwrite("flow.png", colored_flow);
//    }

    cv::imshow(WindowName, frame);

    char c = static_cast<char>(cv::waitKey(2));
    if (c == 'q')
            break;
    prior_frame = frame;
  }
}

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, Keys);

	cv::String tracker_algorithm = parser.get<cv::String>(0);
	cv::String video_name = parser.get<cv::String>(1);

	parser.about("OpenCV Tracker API Test");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	if (tracker_algorithm.empty() || video_name.empty() || !parser.check())
	{
		parser.printErrors();
		return -1;
	}

  //mainMultitracker(tracker_algorithm, video_name);
  //return 0;
  mainOpticalFlow(video_name);
  return 0;

	//open the capture
	cv::VideoCapture cap;
	cap.open(video_name);

	if (!cap.isOpened())
	{
		Help();
		std::cout << "***Could not initialize capturing...***\n";
		std::cout << "Current parameter's value: \n";
		parser.printMessage();
		return -1;
	}

	cv::Mat frame;
	Paused = true;
	WindowName = "Tracking API: " + tracker_algorithm;
	cv::namedWindow(WindowName, 0);
	cv::setMouseCallback(WindowName, OnMouse, 0);

	std::cout << "\n\nHot Keys: \n"
		"\tq - quit the program\n"
		"\tp - pause video\n";

        Paused = false;
        cv::Ptr<cv::Tracker> grassLineTracker = createTracker(tracker_algorithm);
        cv::Mat centerColors(200, 100, CV_8UC3), centerColorsRGB(200, 100, CV_8UC3);
        cv::namedWindow("centerColors", 0);
        bool initialized = false;
        while (true)
        {
          if (!Paused)
          {
            cap >> frame;
            frame.copyTo(Image);
            if(!initialized) {
              //if (grassLineTracker->init(Image, cv::Rect(513, 129, 70, 45)))
              //if (grassLineTracker->init(Image, cv::Rect(557, 454, 103, 41)))
              //if (grassLineTracker->init(Image, cv::Rect(418, 159, 71, 25)))
              if (grassLineTracker->init(Image, cv::Rect(276, 429, 82, 59)))
              {
                initialized = true;
              }
            }
            cv::Rect2d BB;

            cv::Scalar maxColor(0, 0, 0), minColor(999, 999, 999);
            if (grassLineTracker->update(frame, BB)) {
              cv::rectangle(frame, BB, cv::Scalar(0, 0, 255), 2, 1);
              cv::Mat hsv;
              cv::cvtColor(Image, hsv, CV_BGR2HSV);
//              std::vector<cv::Vec3f> points;
//              for(int y = BB.y; y < BB.y + BB.height; y++){
//                for(int x = BB.x; x < BB.x + BB.width; x++){
//                  points.push_back(frame.at<cv::Vec3f>(y, x));
//                }
//              }
              cv::Mat points(BB.width * BB.height, 3, CV_32F);
              for(int y = 0; y < BB.height; y++){
                for(int x = 0; x < BB.width; x++){
                  for(int z = 0; z < 3; z++){
                    points.at<float>(y + x * BB.height, z) = hsv.at<cv::Vec3b>(y + BB.y, x + BB.x)[z];
                  }
                }
              }

              int clusterCount = 2;
              cv::Mat labels;
              int attempts = 5;
              cv::Mat centers;
              int compactness = cv::kmeans(points, clusterCount, labels,
                         cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                          10, 0.01),
                         attempts, cv::KMEANS_PP_CENTERS, centers);
              std::vector<cv::Scalar> colors(clusterCount);
              int height = 200 / clusterCount;
              for(int i = 0; i < clusterCount; i++){
                colors[i] = cv::Scalar(centers.at<float>(i, 0), centers.at<float>(i, 1), centers.at<float>(i, 2));
                std::cout << "Center " << i << ": " << colors[i] << std::endl;
              }

              // take the lighter color as line color
              //int lineColorIdx = colors[0][3] > colors[1][3] ? 0 : 1;
              int lineColorIdx = colors[0][2] > colors[1][2] ? 0 : 1;
              cv::Scalar lineColor = colors[lineColorIdx];
              cv::Mat new_image(BB.size(), frame.type());
              for( int y = 0; y < BB.height; y++ ) {
                for( int x = 0; x < BB.width; x++ ) {
                  int cluster_idx = labels.at<int>(y + x*BB.height,0);
                  new_image.at<cv::Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
                  new_image.at<cv::Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
                  new_image.at<cv::Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
                  if(cluster_idx == lineColorIdx){
                    for(int i = 0; i < 3; i++){
                      maxColor[i] = MAX(maxColor[i], hsv.at<cv::Vec3b>(y + BB.y, x + BB.x)[i]);
                      minColor[i] = MIN(minColor[i], hsv.at<cv::Vec3b>(y + BB.y, x + BB.x)[i]);
                    }
                  }
                }
              }
              std::cout << "minColor: " << minColor << std::endl;
              std::cout << "maxColor: " << maxColor << std::endl;
              cv::cvtColor(new_image, centerColorsRGB, CV_HSV2BGR);
              //imshow( "clustered image", new_image );
              cv::imshow("centerColors", centerColorsRGB);
            }

            //cv::GaussianBlur(frame, frame, cv::Size(5,5), 0);

            cv::Mat dst;
            //Canny(frame, dst, 50, 200, 3);
            // Copy edges to the images that will display the results in BGR
            //cvtColor(dst, frame, cv::COLOR_GRAY2BGR);
            //cv::Mat binary;
            //binary = dst;

            //imshow(WindowName, frame);
            //cv::waitKey();

            cv::Mat binary = getLineMask(Image, minColor, maxColor);

            //cv::Mat corners;
            //cv::cornerHarris(corners, corners, )
            cv::Mat lines = getHoughTransform(binary);
            //cv::Mat lines = getHoughTransform(dst);



            imshow(WindowName, lines);
            //cv::waitKey();
            char c = static_cast<char>(cv::waitKey(2));
            if (c == 'q')
                    break;
            if (c == 'p')
                    Paused = !Paused;
          }
        }

//        std::vector<cv::Mat> bgr_planes;
//        cv::Rect roi(962, 179, 24, 12);
//        //cv::Rect roi(600, 400, 100, 100);
//        //cv::Rect roi(950, 350, 100, 100);
//        cv::split( hsv(roi), bgr_planes );
//        std::cout << "Size: " << bgr_planes[0].size() << std::endl;
//        int histSize = 100;
//        float range[] = { 0, 256 }; //the upper boundary is exclusive
//        const float* histRange = { range };
//        bool uniform = true, accumulate = false;
//        cv::Mat b_hist, g_hist, r_hist;
//        cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
//        cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
//        cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
//        int hist_w = 500, hist_h = 400;
//        int bin_w = cvRound( (double) hist_w/histSize );
//        cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
//        cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
//        cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
//        cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
//        for( int i = 1; i < histSize; i++ )
//        {
//            cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
//                  cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
//                  cv::Scalar( 255, 0, 0), 2, 8, 0  );
//            cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
//                  cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
//                  cv::Scalar( 0, 255, 0), 2, 8, 0  );
//            cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
//                  cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
//                  cv::Scalar( 0, 0, 255), 2, 8, 0  );
//        }
//        cv::imwrite("../hist.png", histImage);
//        cv::imshow(WindowName, histImage);
//        cv::waitKey();

        return 0;
}
