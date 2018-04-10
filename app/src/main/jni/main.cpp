//
// Created by kimhj on 2018-03-31.
//

#include <jni.h>
#include "com_example_myfiltercamera_MainActivity.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <android/log.h>

using namespace cv;
using namespace std;

extern "C"{

    // MainActivity의 Negative filter에서 사용하고자 했던 함수입니다.
    // 작동하지 않기 때문에 주석처리되어 현재 불리지 않는 함수입니다.
//    JNIEXPORT void JNICALL
//    Java_com_example_myfiltercamera_MainActivity_convertNegative
//        (JNIEnv *env,
//         jobject instance,
//         jlong matAddrInput,
//         jlong matAddrOutput){
//
//        Mat &matInput = *(Mat *)matAddrInput;
//        Mat &matOutput = *(Mat *)matAddrOutput;
//
//        double contrast = 1;
//        int brightness = 10;
//
//        matInput.convertTo(matOutput, -1, contrast, brightness);
//
//
//    }


    /*
    * Class:     com_example_myfiltercamera_MainActivity
    * Method:    loadCascade
    * Signature: (Ljava/lang/String;)J
    */
    JNIEXPORT jlong JNICALL
    Java_com_example_myfiltercamera_MainActivity_loadCascade
        (JNIEnv *env,
         jclass type,
         jstring cascadeFileName_){

        const char *nativeFileNameString = env->GetStringUTFChars(cascadeFileName_, 0);

        string baseDir("/storage/emulated/0/");
        baseDir.append(nativeFileNameString);
        const char *pathDir = baseDir.c_str();

        jlong ret = 0;
        ret = (jlong) new CascadeClassifier(pathDir);
        if (((CascadeClassifier *) ret)->empty()) {
            __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ",
                                "CascadeClassifier로 로딩 실패  %s", nativeFileNameString);
        }
        else
            __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ",
                                "CascadeClassifier로 로딩 성공 %s", nativeFileNameString);


        env->ReleaseStringUTFChars(cascadeFileName_, nativeFileNameString);

        return ret;

    };


    float resize(Mat img_src, Mat &img_resize, int resize_width){
        float scale = resize_width / (float)img_src.cols ;
        if (img_src.cols > resize_width) {
            int new_height = cvRound(img_src.rows * scale);
            resize(img_src, img_resize, Size(resize_width, new_height));
        } else {
            img_resize = img_src;
        }
        return scale;
    }

    Mat putMask(Mat src, Mat mask, Point center, Size face_size){

        // 마스크 크기를 얼굴 크기에 맞게 resize하여 resizeMask에 담음
        Mat resizeMask;
        resize(mask,resizeMask,face_size);
        cvtColor(resizeMask, resizeMask, COLOR_BGR2RGBA);

        // 마스크 이미지의 흰색 배경을 투명하게 처리하기위한 부분
        // 먼저 gray scale로 바꿔 grayMask에 담음
        Mat grayMask;
        cvtColor(resizeMask, grayMask, COLOR_RGBA2GRAY);
        // gray scale의 이미지를 검정색과 흰색으로만 표현되도록 변경하여 binaryMask에 담음
        Mat binaryMask;
        threshold(grayMask,binaryMask,230,255,CV_THRESH_BINARY_INV);
        // resizeMask(색 있는 마스크)로 resultMask(흰색 배경 없앤 완성된 마스크)를 만들 때
        // binaryMask(흰,검으로 이뤄진 마스크)에서 0이 아닌 부분에만 복사하도록 함
        Mat resultMask;
        resizeMask.copyTo(resultMask, binaryMask);

        // 마스크를 적용할 부분을 원래 사진에서 찾음
        Rect roi(center.x - face_size.width/2, center.y - face_size.width/2, face_size.width, face_size.width);
        Mat srcRoi(src, roi);
        // 찾은 부분에 마스크를 적용함
        addWeighted(srcRoi, 1, resultMask, 1, 0, srcRoi);

        return src;
    }

    /*
    * Class:     com_example_myfiltercamera_MainActivity
    * Method:    detect
    * Signature: (JJJJ)V
    */
    JNIEXPORT void JNICALL
    Java_com_example_myfiltercamera_MainActivity_detect
        (JNIEnv *env,
         jclass type,
         jlong cascadeClassifier_face,
         jlong cascadeClassifier_eye,
         jlong matAddrInput,
         jlong matAddrResult,
         jlong matAddrMask){

        // 받아온 Mat
        Mat &img_input = *(Mat *) matAddrInput;
        Mat &img_result = *(Mat *) matAddrResult;
        Mat &img_mask = *(Mat *) matAddrMask;

        // result에 input값 복사
        img_result = img_input.clone();

        // 탐지한 얼굴을 담아둘 변수
        std::vector<Rect> faces;
        // 검출을 위해 gray scale인 Mat 생성
        Mat img_gray;

        cvtColor(img_input, img_gray, COLOR_BGR2GRAY);
        equalizeHist(img_gray, img_gray);

        Mat img_resize;
        float resizeRatio = resize(img_gray, img_resize, 640);

        // 얼굴 탐지
        // InputArray, OutputObjectVector, ScaleFactor, MinNeighbors, Flags, MinSize, MaxSize
        ((CascadeClassifier *) cascadeClassifier_face)->detectMultiScale( img_resize, faces, 1.2, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );


        __android_log_print(ANDROID_LOG_DEBUG, (char *) "native-lib :: ", (char *) "face %d found ", faces.size());

        for (int i = 0; i < faces.size(); i++) {
            // 탐지된 얼굴의 시작 좌표
            double real_facesize_x = faces[i].x / resizeRatio;
            double real_facesize_y = faces[i].y / resizeRatio;
            // 탐지된 얼굴의 길이
            double real_facesize_width = faces[i].width / resizeRatio;
            double real_facesize_height = faces[i].height / resizeRatio;
            // 탐지된 좌표를 기준으로 중심 찾기
            Point center( real_facesize_x + real_facesize_width / 2, real_facesize_y + real_facesize_height/2);
            // 얼굴 사이즈
            Size face_size(real_facesize_width, real_facesize_height);
            // 마스크 씌우는 함수
            img_result = putMask(img_result, img_mask, center, face_size);

//            // 탐지된 얼굴에 타원 그리기
//            ellipse(img_result, center, Size( real_facesize_width / 2, real_facesize_height / 2), 0, 0, 360, Scalar(255, 0, 255), 30, 8, 0);
//
//            // 얼굴 범위 내에서 눈 찾기 위해 얼굴 범위를 포함하는 사각형 만듦
//            Rect face_area(real_facesize_x, real_facesize_y, real_facesize_width,real_facesize_height);
//            // 그 사각형 내부를 gray scale의 mat 생성
//            Mat faceROI = img_gray( face_area );
//            // 탐지한 눈을 담아둘 변수
//            std::vector<Rect> eyes;
//
//            // 눈 탐지
//            ((CascadeClassifier *) cascadeClassifier_eye)->detectMultiScale( faceROI, eyes, 1.2, 5, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
//
//            for ( size_t j = 0; j < eyes.size(); j++ )
//            {
//                // 탐지된 눈의 중심 찾기
//                Point eye_center( real_facesize_x + eyes[j].x + eyes[j].width/2, real_facesize_y + eyes[j].y + eyes[j].height/2 );
//                // 탐지된 눈에 그릴 원의 반지름 구하기
//                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
//                // 탐지된 눈에 원 그리기
//                circle( img_result, eye_center, radius, Scalar( 255, 0, 0 ), 30, 8, 0 );
//            }
        }

    };

}