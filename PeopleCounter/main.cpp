#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include <opencv2/highgui/highgui.hpp>
#include "CTracker.h"
#include <iostream>
#include <vector>


using namespace cv;
using namespace std;

Point p1 = Point(0,140);
Point p2 = Point(500,140);

Mat fgMaskMOG2;
Ptr<BackgroundSubtractor> pMOG2;
Scalar Colors[]={Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(0,255,255),Scalar(255,0,255),Scalar(255,127,255),Scalar(127,0,255),Scalar(127,0,127)};


int main(int ac, char** av)
{
    
    Mat frame,thresh_frame;
    vector<Mat> channels;
    
    
    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    
    
    vector<Point2f> centers;
    
    
    
    
    cv::Mat fore;
    
    
    
    //  cv::BackgroundSubtractorMOG2 bg;
    // cv::Ptr<BackgroundSubtractorMOG2> bg = createBackgroundSubtractorMOG2();
    // bg.nmixtures = 3;
    //bg.bShadowDetection = false;
    pMOG2 = createBackgroundSubtractorMOG2();
    //  Ptr<BackgroundSubtractorMOG2> bg = createBackgroundSubtractorMOG2();
    
    int in=0;
    int out=0;
    int track=0;
    
    VideoCapture cap;
    
    cap.open("/Users/MCube/Downloads/video-1443481497.mp4");
    
    if(!cap.isOpened())
    {
        cerr << "Problem opening video source" << endl;
    }
    
    
    CTracker tracker(0.2,0.5,60.0,10,20);
    
    
    while((char)waitKey(30) != 'q' && cap.grab())
    {
        
        bool bSuccess =cap.retrieve(frame);
        
        
        //        stringstream ss;
        //        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
        //                  cv::Scalar(255,255,255), -1);
        //        ss << cap.get(CV_CAP_PROP_POS_FRAMES);
        //
        //        string frameNumberString = ss.str();
        
        //        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
        //                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        
        line(frame, p1, p2, Scalar(0,0,255), 1);
        
        centers.clear();
        // read a new frame from video
        
        
        if (!bSuccess) //if not success, break loop
        {
            cout << "ERROR: Cannot read a frame from video file" << endl;
            break;
            
        }
        
        //  bg->apply()(frame,fore);
        
        pMOG2->apply(frame, fore);
        
        
        
        //        threshold(fore,fore,127,255,CV_THRESH_BINARY);
        //        medianBlur(fore,fore,3);
        //
        //        erode(fore,fore,Mat());
        //        erode(fore,fore,Mat());
        //        dilate(fore,fore,Mat());
        //        dilate(fore,fore,Mat());
        //        dilate(fore,fore,Mat());
        //        dilate(fore,fore,Mat());
        //        dilate(fore,fore,Mat());
        //        dilate(fore,fore,Mat());
        //        dilate(fore,fore,Mat());
        //        dilate(fore,fore,Mat());
        //        dilate(fore,fore,Mat());
        //        dilate(fore,fore,Mat());
        //        dilate(fore,fore,Mat());
        
        
        //Canny(fore, edges, 10, 100, 3);
        // imshow("contours",edges);
        
        
        cv::normalize(fore, fore, 0, 1., cv::NORM_MINMAX);
        cv::threshold(fore, fore, .5, 1., CV_THRESH_BINARY);
        
        
        split(frame, channels);
        add(channels[0], channels[1], channels[1]);
        subtract(channels[2], channels[1], channels[2]);
        threshold(channels[2], thresh_frame, 50, 255, CV_THRESH_BINARY);
        medianBlur(thresh_frame, thresh_frame, 5);
        
        
        
        findContours(fore, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        
        Mat drawing = Mat::zeros(thresh_frame.size(), CV_8UC1);
        for(size_t i = 0; i < contours.size(); i++)
        {
            //          cout << contourArea(contours[i]) << endl;
            if(contourArea(contours[i]) > 1500)
                drawContours(drawing, contours, i, Scalar::all(255), CV_FILLED, 8, vector<Vec4i>(), 0, Point());
        }
        
        thresh_frame = drawing;
        
        for( size_t i = 0; i < contours.size(); i++ )
        {
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            
        }
        
        for( size_t i = 0; i < contours.size(); i++ )
        {
            if(contourArea(contours[i]) > 1500){
                rectangle( frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0 );
                // Rect r = boundRect[i];
                //  frame(r).copyTo(images_array);
                // img_array.push_back(images_array);
                Point center = Point(boundRect[i].x + (boundRect[i].width /2), boundRect[i].y + (boundRect[i].height/2));
                cv::circle(frame,center, 8, Scalar(0, 0, 255), -1, 1,0);
                centers.push_back(center);
                
            }
        }
        
        
        
        if(centers.size()>0)
        {
            tracker.Update(centers);
            
            //       cout << tracker.tracks.size()  << endl;
            
            for(int i=0;i<tracker.tracks.size();i++)
            {
                if(tracker.tracks[i]->trace.size()>1)
                {
                    for(int j=0;j<tracker.tracks[i]->trace.size()-1;j++)
                    {
                        line(frame,tracker.tracks[i]->trace[j],tracker.tracks[i]->trace[j+1],Colors[tracker.tracks[i]->track_id%9],2,CV_AA);
                        //drawCross(frame, tracker.tracks[i]->trace[j], Scalar(255, 255, 255), 5);
                        if (tracker.tracks[i]->prediction.y >= 140 && tracker.tracks[i]->trace[j].y < 140 && tracker.tracks[i]->track_id > 0) {
                            in++;
                            cout << "IN : " << in << endl;
                            tracker.tracks[i]->track_id = 0;
                        }
                        if (tracker.tracks[i]->prediction.y <= 140 && tracker.tracks[i]->trace[j].y > 140 && tracker.tracks[i]->track_id > 0) {
                            out++;
                            cout << "OUT : " << out << endl;
                            tracker.tracks[i]->track_id = 0;
                        }
                    }
                }
            }
        }
        
        
        imshow("OUTPUT Fist Camera",frame);
        
        
        waitKey(30);
    }
    //delete detector;
    //destroyAllWindows();
    return 0;
    
}



//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(Point2f pt,float dt,float Accel_noise_mag)
{
    //time increment (lower values makes target more "massive")
    deltatime = dt; //0.2
    
    // We don't know acceleration, so, assume it to process noise.
    // But we can guess, the range of acceleration values thich can be achieved by tracked object.
    // Process noise. (standard deviation of acceleration: Ï/Ò^2)
    // shows, woh much target can accelerate.
    //float Accel_noise_mag = 0.5;
    
    //4 state variables, 2 measurements
    kalman = new KalmanFilter( 4, 2, 0 );
    // Transition matrix
    kalman->transitionMatrix = (Mat_<float>(4, 4) << 1,0,deltatime,0,   0,1,0,deltatime,  0,0,1,0,  0,0,0,1);
    
    // init...
    LastResult = pt;
    kalman->statePre.at<float>(0) = pt.x; // x
    kalman->statePre.at<float>(1) = pt.y; // y
    
    kalman->statePre.at<float>(2) = 0;
    kalman->statePre.at<float>(3) = 0;
    
    kalman->statePost.at<float>(0)=pt.x;
    kalman->statePost.at<float>(1)=pt.y;
    
    setIdentity(kalman->measurementMatrix);
    
    kalman->processNoiseCov=(Mat_<float>(4, 4) <<
                             pow(deltatime,4.0)/4.0	,0						,pow(deltatime,3.0)/2.0		,0,
                             0						,pow(deltatime,4.0)/4.0	,0							,pow(deltatime,3.0)/2.0,
                             pow(deltatime,3.0)/2.0	,0						,pow(deltatime,2.0)			,0,
                             0						,pow(deltatime,3.0)/2.0	,0							,pow(deltatime,2.0));
    
    
    kalman->processNoiseCov*=Accel_noise_mag;
    
    setIdentity(kalman->measurementNoiseCov, Scalar::all(0.1));
    
    setIdentity(kalman->errorCovPost, Scalar::all(.1));
    
}

//---------------------------------------------------------------------------
TKalmanFilter::~TKalmanFilter()
{
    delete kalman;
}

//---------------------------------------------------------------------------
Point2f TKalmanFilter::GetPrediction()
{
    Mat prediction = kalman->predict();
    LastResult=Point2f(prediction.at<float>(0),prediction.at<float>(1));
    return LastResult;
}

//---------------------------------------------------------------------------

Point2f TKalmanFilter::Update(Point2f p, bool DataCorrect)
{
    Mat measurement(2,1,CV_32FC1);
    if(!DataCorrect)
    {
        measurement.at<float>(0) = LastResult.x;  //update using prediction
        measurement.at<float>(1) = LastResult.y;
    }
    else
    {
        measurement.at<float>(0) = p.x;  //update using measurements
        measurement.at<float>(1) = p.y;
    }
    // Correction
    Mat estimated = kalman->correct(measurement);
    LastResult.x=estimated.at<float>(0);   //update using measurements
    LastResult.y=estimated.at<float>(1);
    return LastResult;
}

//---------------------------------------------------------------------------






AssignmentProblemSolver::AssignmentProblemSolver()
{
}

AssignmentProblemSolver::~AssignmentProblemSolver()
{
}

double AssignmentProblemSolver::Solve(vector<vector<double>>& DistMatrix,vector<int>& Assignment,TMethod Method)
{
    int N=DistMatrix.size(); // number of columns (tracks)
    int M=DistMatrix[0].size(); // number of rows (measurements)
    
    int *assignment		=new int[N];
    double *distIn		=new double[N*M];
    
    double  cost;
    // Fill matrix with random numbers
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            distIn[i+N*j] = DistMatrix[i][j];
        }
    }
    switch(Method)
    {
        case optimal: assignmentoptimal(assignment, &cost, distIn, N, M); break;
            
        case many_forbidden_assignments: assignmentoptimal(assignment, &cost, distIn, N, M); break;
            
        case without_forbidden_assignments: assignmentoptimal(assignment, &cost, distIn, N, M); break;
    }
    
    // form result
    Assignment.clear();
    for(int x=0; x<N; x++)
    {
        Assignment.push_back(assignment[x]);
    }
    
    delete[] assignment;
    delete[] distIn;
    return cost;
}
// --------------------------------------------------------------------------
// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentoptimal(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns)
{
    double *distMatrix;
    double *distMatrixTemp;
    double *distMatrixEnd;
    double *columnEnd;
    double  value;
    double  minValue;
    
    bool *coveredColumns;
    bool *coveredRows;
    bool *starMatrix;
    bool *newStarMatrix;
    bool *primeMatrix;
    
    int nOfElements;
    int minDim;
    int row;
    int col;
    
    // Init
    *cost = 0;
    for(row=0; row<nOfRows; row++)
    {
        assignment[row] = -1.0;
    }
    
    // Generate distance matrix
    // and check matrix elements positiveness :)
    
    // Total elements number
    nOfElements   = nOfRows * nOfColumns;
    // Memory allocation
    distMatrix    = (double *)malloc(nOfElements * sizeof(double));
    // Pointer to last element
    distMatrixEnd = distMatrix + nOfElements;
    
    //
    for(row=0; row<nOfElements; row++)
    {
        value = distMatrixIn[row];
        if(value < 0)
        {
            cout << "All matrix elements have to be non-negative." << endl;
        }
        distMatrix[row] = value;
    }
    
    // Memory allocation
    coveredColumns = (bool *)calloc(nOfColumns,  sizeof(bool));
    coveredRows    = (bool *)calloc(nOfRows,     sizeof(bool));
    starMatrix     = (bool *)calloc(nOfElements, sizeof(bool));
    primeMatrix    = (bool *)calloc(nOfElements, sizeof(bool));
    newStarMatrix  = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */
    
    /* preliminary steps */
    if(nOfRows <= nOfColumns)
    {
        minDim = nOfRows;
        for(row=0; row<nOfRows; row++)
        {
            /* find the smallest element in the row */
            distMatrixTemp = distMatrix + row;
            minValue = *distMatrixTemp;
            distMatrixTemp += nOfRows;
            while(distMatrixTemp < distMatrixEnd)
            {
                value = *distMatrixTemp;
                if(value < minValue)
                {
                    minValue = value;
                }
                distMatrixTemp += nOfRows;
            }
            /* subtract the smallest element from each element of the row */
            distMatrixTemp = distMatrix + row;
            while(distMatrixTemp < distMatrixEnd)
            {
                *distMatrixTemp -= minValue;
                distMatrixTemp += nOfRows;
            }
        }
        /* Steps 1 and 2a */
        for(row=0; row<nOfRows; row++)
        {
            for(col=0; col<nOfColumns; col++)
            {
                if(distMatrix[row + nOfRows*col] == 0)
                {
                    if(!coveredColumns[col])
                    {
                        starMatrix[row + nOfRows*col] = true;
                        coveredColumns[col]           = true;
                        break;
                    }
                }
            }
        }
    }
    else /* if(nOfRows > nOfColumns) */
    {
        minDim = nOfColumns;
        for(col=0; col<nOfColumns; col++)
        {
            /* find the smallest element in the column */
            distMatrixTemp = distMatrix     + nOfRows*col;
            columnEnd      = distMatrixTemp + nOfRows;
            minValue = *distMatrixTemp++;
            while(distMatrixTemp < columnEnd)
            {
                value = *distMatrixTemp++;
                if(value < minValue)
                {
                    minValue = value;
                }
            }
            /* subtract the smallest element from each element of the column */
            distMatrixTemp = distMatrix + nOfRows*col;
            while(distMatrixTemp < columnEnd)
            {
                *distMatrixTemp++ -= minValue;
            }
        }
        /* Steps 1 and 2a */
        for(col=0; col<nOfColumns; col++)
        {
            for(row=0; row<nOfRows; row++)
            {
                if(distMatrix[row + nOfRows*col] == 0)
                {
                    if(!coveredRows[row])
                    {
                        starMatrix[row + nOfRows*col] = true;
                        coveredColumns[col]           = true;
                        coveredRows[row]              = true;
                        break;
                    }
                }
            }
        }
        
        for(row=0; row<nOfRows; row++)
        {
            coveredRows[row] = false;
        }
    }
    /* move to step 2b */
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    /* compute cost and remove invalid assignments */
    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
    /* free allocated memory */
    free(distMatrix);
    free(coveredColumns);
    free(coveredRows);
    free(starMatrix);
    free(primeMatrix);
    free(newStarMatrix);
    return;
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns)
{
    int row, col;
    for(row=0; row<nOfRows; row++)
    {
        for(col=0; col<nOfColumns; col++)
        {
            if(starMatrix[row + nOfRows*col])
            {
                assignment[row] = col;
                break;
            }
        }
    }
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows)
{
    int row, col;
    for(row=0; row<nOfRows; row++)
    {
        col = assignment[row];
        if(col >= 0)
        {
            *cost += distMatrix[row + nOfRows*col];
        }
    }
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    bool *starMatrixTemp, *columnEnd;
    int col;
    /* cover every column containing a starred zero */
    for(col=0; col<nOfColumns; col++)
    {
        starMatrixTemp = starMatrix     + nOfRows*col;
        columnEnd      = starMatrixTemp + nOfRows;
        while(starMatrixTemp < columnEnd)
        {
            if(*starMatrixTemp++)
            {
                coveredColumns[col] = true;
                break;
            }
        }
    }
    /* move to step 3 */
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    int col, nOfCoveredColumns;
    /* count covered columns */
    nOfCoveredColumns = 0;
    for(col=0; col<nOfColumns; col++)
    {
        if(coveredColumns[col])
        {
            nOfCoveredColumns++;
        }
    }
    if(nOfCoveredColumns == minDim)
    {
        /* algorithm finished */
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    }
    else
    {
        /* move to step 3 */
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    bool zerosFound;
    int row, col, starCol;
    zerosFound = true;
    while(zerosFound)
    {
        zerosFound = false;
        for(col=0; col<nOfColumns; col++)
        {
            if(!coveredColumns[col])
            {
                for(row=0; row<nOfRows; row++)
                {
                    if((!coveredRows[row]) && (distMatrix[row + nOfRows*col] == 0))
                    {
                        /* prime zero */
                        primeMatrix[row + nOfRows*col] = true;
                        /* find starred zero in current row */
                        for(starCol=0; starCol<nOfColumns; starCol++)
                            if(starMatrix[row + nOfRows*starCol])
                            {
                                break;
                            }
                        if(starCol == nOfColumns) /* no starred zero found */
                        {
                            /* move to step 4 */
                            step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return;
                        }
                        else
                        {
                            coveredRows[row]        = true;
                            coveredColumns[starCol] = false;
                            zerosFound              = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    /* move to step 5 */
    step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
{
    int n, starRow, starCol, primeRow, primeCol;
    int nOfElements = nOfRows*nOfColumns;
    /* generate temporary copy of starMatrix */
    for(n=0; n<nOfElements; n++)
    {
        newStarMatrix[n] = starMatrix[n];
    }
    /* star current zero */
    newStarMatrix[row + nOfRows*col] = true;
    /* find starred zero in current column */
    starCol = col;
    for(starRow=0; starRow<nOfRows; starRow++)
    {
        if(starMatrix[starRow + nOfRows*starCol])
        {
            break;
        }
    }
    while(starRow<nOfRows)
    {
        /* unstar the starred zero */
        newStarMatrix[starRow + nOfRows*starCol] = false;
        /* find primed zero in current row */
        primeRow = starRow;
        for(primeCol=0; primeCol<nOfColumns; primeCol++)
        {
            if(primeMatrix[primeRow + nOfRows*primeCol])
            {
                break;
            }
        }
        /* star the primed zero */
        newStarMatrix[primeRow + nOfRows*primeCol] = true;
        /* find starred zero in current column */
        starCol = primeCol;
        for(starRow=0; starRow<nOfRows; starRow++)
        {
            if(starMatrix[starRow + nOfRows*starCol])
            {
                break;
            }
        }
    }
    /* use temporary copy as new starMatrix */
    /* delete all primes, uncover all rows */
    for(n=0; n<nOfElements; n++)
    {
        primeMatrix[n] = false;
        starMatrix[n]  = newStarMatrix[n];
    }
    for(n=0; n<nOfRows; n++)
    {
        coveredRows[n] = false;
    }
    /* move to step 2a */
    step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    double h, value;
    int row, col;
    /* find smallest uncovered element h */
    h = DBL_MAX;
    for(row=0; row<nOfRows; row++)
    {
        if(!coveredRows[row])
        {
            for(col=0; col<nOfColumns; col++)
            {
                if(!coveredColumns[col])
                {
                    value = distMatrix[row + nOfRows*col];
                    if(value < h)
                    {
                        h = value;
                    }
                }
            }
        }
    }
    /* add h to each covered row */
    for(row=0; row<nOfRows; row++)
    {
        if(coveredRows[row])
        {
            for(col=0; col<nOfColumns; col++)
            {
                distMatrix[row + nOfRows*col] += h;
            }
        }
    }
    /* subtract h from each uncovered column */
    for(col=0; col<nOfColumns; col++)
    {
        if(!coveredColumns[col])
        {
            for(row=0; row<nOfRows; row++)
            {
                distMatrix[row + nOfRows*col] -= h;
            }
        }
    }
    /* move to step 3 */
    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}


// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases without forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal2(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns)
{
    int n, row, col, tmpRow, tmpCol, nOfElements;
    double value, minValue, *distMatrix;
    
    
    /* make working copy of distance Matrix */
    nOfElements   = nOfRows * nOfColumns;
    distMatrix    = (double *)malloc(nOfElements * sizeof(double));
    for(n=0; n<nOfElements; n++)
    {
        distMatrix[n] = distMatrixIn[n];
    }
    
    /* initialization */
    *cost = 0;
    for(row=0; row<nOfRows; row++)
    {
        assignment[row] = -1.0;
    }
    
    /* recursively search for the minimum element and do the assignment */
    while(true)
    {
        /* find minimum distance observation-to-track pair */
        minValue = DBL_MAX;
        for(row=0; row<nOfRows; row++)
            for(col=0; col<nOfColumns; col++)
            {
                value = distMatrix[row + nOfRows*col];
                if(value!=DBL_MAX && (value < minValue))
                {
                    minValue = value;
                    tmpRow   = row;
                    tmpCol   = col;
                }
            }
        
        if(minValue!=DBL_MAX)
        {
            assignment[tmpRow] = tmpCol;
            *cost += minValue;
            for(n=0; n<nOfRows; n++)
            {
                distMatrix[n + nOfRows*tmpCol] = DBL_MAX;
            }
            for(n=0; n<nOfColumns; n++)
            {
                distMatrix[tmpRow + nOfRows*n] = DBL_MAX;
            }
        }
        else
            break;
        
    } /* while(true) */
    
    free(distMatrix);
}
// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases with many forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal1(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns)
{
    bool infiniteValueFound, finiteValueFound, repeatSteps, allSinglyValidated, singleValidationFound;
    int n, row, col, tmpRow, tmpCol, nOfElements;
    int *nOfValidObservations, *nOfValidTracks;
    double value, minValue, *distMatrix;
    
    
    /* make working copy of distance Matrix */
    nOfElements   = nOfRows * nOfColumns;
    distMatrix    = (double *)malloc(nOfElements * sizeof(double));
    for(n=0; n<nOfElements; n++)
    {
        distMatrix[n] = distMatrixIn[n];
    }
    /* initialization */
    *cost = 0;
    
    for(row=0; row<nOfRows; row++)
    {
        assignment[row] = -1.0;
    }
    
    /* allocate memory */
    nOfValidObservations  = (int *)calloc(nOfRows,    sizeof(int));
    nOfValidTracks        = (int *)calloc(nOfColumns, sizeof(int));
    
    /* compute number of validations */
    infiniteValueFound = false;
    finiteValueFound  = false;
    for(row=0; row<nOfRows; row++)
    {
        for(col=0; col<nOfColumns; col++)
        {
            if(distMatrix[row + nOfRows*col]!=DBL_MAX)
            {
                nOfValidTracks[col]       += 1;
                nOfValidObservations[row] += 1;
                finiteValueFound = true;
            }
            else
                infiniteValueFound = true;
        }
    }
    
    if(infiniteValueFound)
    {
        if(!finiteValueFound)
        {
            return;
        }
        repeatSteps = true;
        
        while(repeatSteps)
        {
            repeatSteps = false;
            
            /* step 1: reject assignments of multiply validated tracks to singly validated observations		 */
            for(col=0; col<nOfColumns; col++)
            {
                singleValidationFound = false;
                for(row=0; row<nOfRows; row++)
                    if(distMatrix[row + nOfRows*col]!=DBL_MAX && (nOfValidObservations[row] == 1))
                    {
                        singleValidationFound = true;
                        break;
                    }
                
                if(singleValidationFound)
                {
                    for(row=0; row<nOfRows; row++)
                        if((nOfValidObservations[row] > 1) && distMatrix[row + nOfRows*col]!=DBL_MAX)
                        {
                            distMatrix[row + nOfRows*col] = DBL_MAX;
                            nOfValidObservations[row] -= 1;
                            nOfValidTracks[col]       -= 1;
                            repeatSteps = true;
                        }
                }
            }
            
            /* step 2: reject assignments of multiply validated observations to singly validated tracks */
            if(nOfColumns > 1)
            {
                for(row=0; row<nOfRows; row++)
                {
                    singleValidationFound = false;
                    for(col=0; col<nOfColumns; col++)
                    {
                        if(distMatrix[row + nOfRows*col]!=DBL_MAX && (nOfValidTracks[col] == 1))
                        {
                            singleValidationFound = true;
                            break;
                        }
                    }
                    
                    if(singleValidationFound)
                    {
                        for(col=0; col<nOfColumns; col++)
                        {
                            if((nOfValidTracks[col] > 1) && distMatrix[row + nOfRows*col]!=DBL_MAX)
                            {
                                distMatrix[row + nOfRows*col] = DBL_MAX;
                                nOfValidObservations[row] -= 1;
                                nOfValidTracks[col]       -= 1;
                                repeatSteps = true;
                            }
                        }
                    }
                }
            }
        } /* while(repeatSteps) */
        
        /* for each multiply validated track that validates only with singly validated  */
        /* observations, choose the observation with minimum distance */
        for(row=0; row<nOfRows; row++)
        {
            if(nOfValidObservations[row] > 1)
            {
                allSinglyValidated = true;
                minValue = DBL_MAX;
                for(col=0; col<nOfColumns; col++)
                {
                    value = distMatrix[row + nOfRows*col];
                    if(value!=DBL_MAX)
                    {
                        if(nOfValidTracks[col] > 1)
                        {
                            allSinglyValidated = false;
                            break;
                        }
                        else if((nOfValidTracks[col] == 1) && (value < minValue))
                        {
                            tmpCol   = col;
                            minValue = value;
                        }
                    }
                }
                
                if(allSinglyValidated)
                {
                    assignment[row] = tmpCol;
                    *cost += minValue;
                    for(n=0; n<nOfRows; n++)
                    {
                        distMatrix[n + nOfRows*tmpCol] = DBL_MAX;
                    }
                    for(n=0; n<nOfColumns; n++)
                    {
                        distMatrix[row + nOfRows*n] = DBL_MAX;
                    }
                }
            }
        }
        
        /* for each multiply validated observation that validates only with singly validated  */
        /* track, choose the track with minimum distance */
        for(col=0; col<nOfColumns; col++)
        {
            if(nOfValidTracks[col] > 1)
            {
                allSinglyValidated = true;
                minValue = DBL_MAX;
                for(row=0; row<nOfRows; row++)
                {
                    value = distMatrix[row + nOfRows*col];
                    if(value!=DBL_MAX)
                    {
                        if(nOfValidObservations[row] > 1)
                        {
                            allSinglyValidated = false;
                            break;
                        }
                        else if((nOfValidObservations[row] == 1) && (value < minValue))
                        {
                            tmpRow   = row;
                            minValue = value;
                        }
                    }
                }
                
                if(allSinglyValidated)
                {
                    assignment[tmpRow] = col;
                    *cost += minValue;
                    for(n=0; n<nOfRows; n++)
                        distMatrix[n + nOfRows*col] = DBL_MAX;
                    for(n=0; n<nOfColumns; n++)
                        distMatrix[tmpRow + nOfRows*n] = DBL_MAX;
                }
            }
        }
    } /* if(infiniteValueFound) */
    
    
    /* now, recursively search for the minimum element and do the assignment */
    while(true)
    {
        /* find minimum distance observation-to-track pair */
        minValue = DBL_MAX;
        for(row=0; row<nOfRows; row++)
            for(col=0; col<nOfColumns; col++)
            {
                value = distMatrix[row + nOfRows*col];
                if(value!=DBL_MAX && (value < minValue))
                {
                    minValue = value;
                    tmpRow   = row;
                    tmpCol   = col;
                }
            }
        
        if(minValue!=DBL_MAX)
        {
            assignment[tmpRow] = tmpCol;
            *cost += minValue;
            for(n=0; n<nOfRows; n++)
                distMatrix[n + nOfRows*tmpCol] = DBL_MAX;
            for(n=0; n<nOfColumns; n++)
                distMatrix[tmpRow + nOfRows*n] = DBL_MAX;
        }
        else
            break;
        
    } /* while(true) */
    
    /* free allocated memory */
    free(nOfValidObservations);
    free(nOfValidTracks);
}
/*
 // --------------------------------------------------------------------------
 // Usage example
 // --------------------------------------------------------------------------
 void main(void)
 {
	// Matrix size
	int N=8; // tracks
	int M=9; // detects
	// Random numbers generator initialization
	srand (time(NULL));
	// Distance matrix N-th track to M-th detect.
	vector< vector<double> > Cost(N,vector<double>(M));
	// Fill matrix with random values
	for(int i=0; i<N; i++)
	{
 for(int j=0; j<M; j++)
 {
 Cost[i][j] = (double)(rand()%1000)/1000.0;
 std::cout << Cost[i][j] << "\t";
 }
 std::cout << std::endl;
	}
 
	AssignmentProblemSolver APS;
 
	vector<int> Assignment;
	
	cout << APS.Solve(Cost,Assignment) << endl;
	
	// Output the result
	for(int x=0; x<N; x++)
	{
 std::cout << x << ":" << Assignment[x] << "\t";
	}
 
	getchar();
 }
 */
// --------------------------------------------------------------------------




size_t CTrack::NextTrackID=1;
// ---------------------------------------------------------------------------
// Track constructor.
// The track begins from initial point (pt)
// ---------------------------------------------------------------------------
CTrack::CTrack(Point2f pt, float dt, float Accel_noise_mag)
{
    track_id=NextTrackID;
    NextTrackID++;
    // Every track have its own Kalman filter,
    // it user for next point position prediction.
    KF = new TKalmanFilter(pt,dt,Accel_noise_mag);
    // Here stored points coordinates, used for next position prediction.
    prediction=pt;
    skipped_frames=0;
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CTrack::~CTrack()
{
    // Free resources.
    delete KF;
}

// ---------------------------------------------------------------------------
// Tracker. Manage tracks. Create, remove, update.
// ---------------------------------------------------------------------------
CTracker::CTracker(float _dt, float _Accel_noise_mag, double _dist_thres, int _maximum_allowed_skipped_frames,int _max_trace_length)
{
    dt=_dt;
    Accel_noise_mag=_Accel_noise_mag;
    dist_thres=_dist_thres;
    maximum_allowed_skipped_frames=_maximum_allowed_skipped_frames;
    max_trace_length=_max_trace_length;
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::Update(vector<Point2f>& detections)
{
    // -----------------------------------
    // If there is no tracks yet, then every point begins its own track.
    // -----------------------------------
    if(tracks.size()==0)
    {
        // If no tracks yet
        for(int i=0;i<detections.size();i++)
        {
            CTrack* tr=new CTrack(detections[i],dt,Accel_noise_mag);
            tracks.push_back(tr);
        }
    }
    
    // -----------------------------------
    // «десь треки уже есть в любом случае
    // -----------------------------------
    int N=tracks.size();		// треки
    int M=detections.size();	// детекты
    
    // ћатрица рассто¤ний от N-ного трека до M-ного детекта.
    vector< vector<double> > Cost(N,vector<double>(M));
    vector<int> assignment; // назначени¤
    
    // -----------------------------------
    // “реки уже есть, составим матрицу рассто¤ний
    // -----------------------------------
    double dist;
    for(int i=0;i<tracks.size();i++)
    {
        // Point2d prediction=tracks[i]->prediction;
        // cout << prediction << endl;
        for(int j=0;j<detections.size();j++)
        {
            Point2d diff=(tracks[i]->prediction-detections[j]);
            dist=sqrtf(diff.x*diff.x+diff.y*diff.y);
            Cost[i][j]=dist;
        }
    }
    // -----------------------------------
    // Solving assignment problem (tracks and predictions of Kalman filter)
    // -----------------------------------
    AssignmentProblemSolver APS;
    APS.Solve(Cost,assignment,AssignmentProblemSolver::optimal);
    
    // -----------------------------------
    // clean assignment from pairs with large distance
    // -----------------------------------
    // Not assigned tracks
    vector<int> not_assigned_tracks;
    
    for(int i=0;i<assignment.size();i++)
    {
        if(assignment[i]!=-1)
        {
            if(Cost[i][assignment[i]]>dist_thres)
            {
                assignment[i]=-1;
                // Mark unassigned tracks, and increment skipped frames counter,
                // when skipped frames counter will be larger than threshold, track will be deleted.
                not_assigned_tracks.push_back(i);
            }
        }
        else
        {
            // If track have no assigned detect, then increment skipped frames counter.
            tracks[i]->skipped_frames++;
        }
        
    }
    
    // -----------------------------------
    // If track didn't get detects long time, remove it.
    // -----------------------------------
    for(int i=0;i<tracks.size();i++)
    {
        if(tracks[i]->skipped_frames>maximum_allowed_skipped_frames)
        {
            delete tracks[i];
            tracks.erase(tracks.begin()+i);
            assignment.erase(assignment.begin()+i);
            i--;
        }
    }
    // -----------------------------------
    // Search for unassigned detects
    // -----------------------------------
    vector<int> not_assigned_detections;
    vector<int>::iterator it;
    for(int i=0;i<detections.size();i++)
    {
        it=find(assignment.begin(), assignment.end(), i);
        if(it==assignment.end())
        {
            not_assigned_detections.push_back(i);
        }
    }
    
    // -----------------------------------
    // and start new tracks for them.
    // -----------------------------------
    if(not_assigned_detections.size()!=0)
    {
        for(int i=0;i<not_assigned_detections.size();i++)
        {
            CTrack* tr=new CTrack(detections[not_assigned_detections[i]],dt,Accel_noise_mag);
            tracks.push_back(tr);
        }
    }
    
    // Update Kalman Filters state
    
    for(int i=0;i<assignment.size();i++)
    {
        // If track updated less than one time, than filter state is not correct.
        
        tracks[i]->KF->GetPrediction();
        
        if(assignment[i]!=-1) // If we have assigned detect, then update using its coordinates,
        {
            tracks[i]->skipped_frames=0;
            tracks[i]->prediction=tracks[i]->KF->Update(detections[assignment[i]],1);
        }else				  // if not continue using predictions
        {
            tracks[i]->prediction=tracks[i]->KF->Update(Point2f(0,0),0);
        }
        
        if(tracks[i]->trace.size()>max_trace_length)
        {
            tracks[i]->trace.erase(tracks[i]->trace.begin(),tracks[i]->trace.end()-max_trace_length);
        }
        
        tracks[i]->trace.push_back(tracks[i]->prediction);
        tracks[i]->KF->LastResult=tracks[i]->prediction;
    }
    
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CTracker::~CTracker(void)
{
    for(int i=0;i<tracks.size();i++)
    {
        delete tracks[i];
    }
    tracks.clear();
}
