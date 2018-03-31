//
// Created by kimhj on 2018-03-31.
//

#include <jni.h>
#include "com_example_myfiltercamera_MainActivity.h"

#include <opencv2/core/core.hpp>

#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

extern "C"{

    // MainActivity의 Negative filter에서 사용하고자 했던 함수입니다.
    // 작동하지 않기 때문에 주석처리되어 현재 불리지 않는 함수입니다.
    JNIEXPORT void JNICALL
    Java_com_example_myfiltercamera_MainActivity_convertNegative
        (JNIEnv *env,
         jobject instance,
         jlong matAddrInput,
         jlong matAddrOutput){

        Mat &matInput = *(Mat *)matAddrInput;
        Mat &matOutput = *(Mat *)matAddrOutput;

        double contrast = 1;
        int brightness = 10;

        matInput.convertTo(matOutput, -1, contrast, brightness);


    }

}