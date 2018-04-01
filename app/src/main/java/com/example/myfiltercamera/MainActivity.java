package com.example.myfiltercamera;

import android.os.Environment;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

public class MainActivity extends AppCompatActivity implements JavaCameraView.CvCameraViewListener2 {

    // Native 함수를 사용하기 위해 먼저 선언해두는 부분입니다.
    // 이 함수는 Negative filter에서 사용하고자 했었습니다.
    // 미완성입니다.
    public native void convertNegative(long matAddrInput, long matAddrResult);

    // 이 부분은 Android.mk의 LOCAL_MODULE에 입력한 lib 값을 사용하겠다고 선언해주는 부분입니다.
    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("native-lib");
    }

    // OpenCV의 카메라 뷰입니다.
    CameraBridgeViewBase cbvCamera;
    // 필터 목록을 담고있는 Spinner입니다.
    Spinner sFilter;
    ArrayAdapter spinnerAdapter;
    // 사진 촬영하기 위한 버튼입니다.
    Button btnTakePic;

    // SubBackground 필터에 쓰일 class입니다.
    // 픽셀에 변화가 생기는 부분을 찾아주는 역활을 합니다.
    BackgroundSubtractorMOG2 bsMOG2;

    // 사진 찍기 버튼(btnTakePic)을 클릭한 경우 true로 변경되고 onCameraFrame > saveImg 순으로 촬영이 진행됩니다.
    // 촬영이 완료되면 false로 다시 변경됩니다.
    boolean takingPicture = false;
    // 사진 촬영이 성공했는지 실패했는지에 따라 다른 Toast 메시지를 띄워주기 위해 사용되는 변수입니다.
    boolean takenPicture = false;
    // Toast를 띄울 때 사용하기 위한 Handler입니다.
    Handler pictureHandler;

    // 카메라 프리뷰 모드를 변경하기 위해 사용되는 부분입니다.
    // 각 필터마다 하나의 모드를 갖습니다.
    public static final int VIEW_MODE_RGBA = 0;         // 일반
    public static final int VIEW_MODE_BLUR = 1;         // 흐림
    public static final int VIEW_MODE_SEPIA = 2;        // 세피아
    public static final int VIEW_MODE_POSTERIZE = 3;    // 포스터모드
    public static final int VIEW_MODE_PIXELIZE = 4;     // 모자이크
    public static final int VIEW_MODE_NEGATIVE = 5;
    public static final int VIEW_MODE_CANNY = 6;        // 외곽선 검출 모드
    public static final int VIEW_MODE_SOBEL = 7;        // 외곽선 검출 모드 2
    public static final int VIEWM_MODE_BACKGROUND = 8;  // 배경 검은색으로 처리
    public static int viewMode = VIEW_MODE_RGBA;

    // OpenCV의 CameraBridgeViewBase를 초기화하기 위해 사용되는 부분입니다.
    private Mat                  mIntermediateMat;
    private Mat                  mMat0;
    private MatOfInt             mChannels[];
    private MatOfInt             mHistSize;
    private int                  mHistSizeNum = 25;
    private MatOfFloat           mRanges;
    private Scalar               mColorsRGB[];
    private Scalar               mColorsHue[];
    private Scalar               mWhilte;
    private Point                mP1;
    private Point                mP2;
    private float                mBuff[];
    private Mat                  mSepiaKernel;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    cbvCamera.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 화면이 계속 켜져있도록, 풀 스크린으로 실행되도록 설정해줍니다.
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);

        cbvCamera = (CameraBridgeViewBase)findViewById(R.id.jcvCamera);
        sFilter = (Spinner)findViewById(R.id.sFilter);
        btnTakePic = (Button)findViewById(R.id.btnTakePic);

        btnTakePic.setOnClickListener(btnClickListener);
        pictureHandler = new Handler();

        // CameraBridgeViewBase에 Listener를 등록하여 뷰가 시작될 때, 정지될때, 파괴될 때의 action을 정의해줍니다.
        cbvCamera.setVisibility(CameraBridgeViewBase.VISIBLE);
        cbvCamera.setCvCameraViewListener(this);

        // Spinner에 등록하기 위해 필터의 종류를 ArrayList에 담습니다.
        ArrayList<String> filterList = new ArrayList<>();
        filterList.add("None");
        filterList.add("Blur");
        filterList.add("Sepia");
        filterList.add("Posterize");
        filterList.add("Pixelize");
        filterList.add("Negative");
        filterList.add("Canny");
        filterList.add("Sobel");
        filterList.add("SubBackground");
        // 필터의 종류를 담고있는 ArrayList를 이용하여 Adapter에 등록합니다.
        spinnerAdapter = new ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, filterList);
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        // 필터 종류를 등록한 Adapter를 Spinner에 연결해줍니다.
        sFilter.setAdapter(spinnerAdapter);
        sFilter.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                switch (position){
                    case 0:
                        viewMode = VIEW_MODE_RGBA;
                        break;
                    case 1:
                        viewMode = VIEW_MODE_BLUR;
                        break;
                    case 2:
                        viewMode = VIEW_MODE_SEPIA;
                        break;
                    case 3:
                        viewMode = VIEW_MODE_POSTERIZE;
                        break;
                    case 4:
                        viewMode = VIEW_MODE_PIXELIZE;
                        break;
                    case 5:
                        viewMode = VIEW_MODE_NEGATIVE;
                        break;
                    case 6:
                        viewMode = VIEW_MODE_CANNY;
                        break;
                    case 7:
                        viewMode = VIEW_MODE_SOBEL;
                        break;
                    case 8:
                        viewMode = VIEWM_MODE_BACKGROUND;
                        break;
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });

    }

    Button.OnClickListener btnClickListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            switch (v.getId()){
                case R.id.btnTakePic:
                    takingPicture = true;
                    break;
            }
        }
    };

    @Override
    public void onPause()
    {
        super.onPause();
        if (cbvCamera != null)
            cbvCamera.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (cbvCamera != null)
            cbvCamera.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        // history 값을 높일수록 움직임을 빠르게 받아들이지 않습니다.
        // 즉, 낮을 수록 빠르게 배경으로 인식하고, 높을 수록 느리게 배경으로 인식합니다.
        bsMOG2 = Video.createBackgroundSubtractorMOG2(15, 32, false);

        mIntermediateMat = new Mat();
        mChannels = new MatOfInt[] { new MatOfInt(0), new MatOfInt(1), new MatOfInt(2) };
        mBuff = new float[mHistSizeNum];
        mHistSize = new MatOfInt(mHistSizeNum);
        mRanges = new MatOfFloat(0f, 256f);
        mMat0  = new Mat();
        mColorsRGB = new Scalar[] { new Scalar(200, 0, 0, 255), new Scalar(0, 200, 0, 255), new Scalar(0, 0, 200, 255) };
        mColorsHue = new Scalar[] {
                new Scalar(255, 0, 0, 255),   new Scalar(255, 60, 0, 255),  new Scalar(255, 120, 0, 255), new Scalar(255, 180, 0, 255), new Scalar(255, 240, 0, 255),
                new Scalar(215, 213, 0, 255), new Scalar(150, 255, 0, 255), new Scalar(85, 255, 0, 255),  new Scalar(20, 255, 0, 255),  new Scalar(0, 255, 30, 255),
                new Scalar(0, 255, 85, 255),  new Scalar(0, 255, 150, 255), new Scalar(0, 255, 215, 255), new Scalar(0, 234, 255, 255), new Scalar(0, 170, 255, 255),
                new Scalar(0, 120, 255, 255), new Scalar(0, 60, 255, 255),  new Scalar(0, 0, 255, 255),   new Scalar(64, 0, 255, 255),  new Scalar(120, 0, 255, 255),
                new Scalar(180, 0, 255, 255), new Scalar(255, 0, 255, 255), new Scalar(255, 0, 215, 255), new Scalar(255, 0, 85, 255),  new Scalar(255, 0, 0, 255)
        };
        mWhilte = Scalar.all(255);
        mP1 = new Point();
        mP2 = new Point();


    }

    @Override
    public void onCameraViewStopped() {
        // Explicitly deallocate Mats
        if (mIntermediateMat != null)
            mIntermediateMat.release();

        mIntermediateMat = null;
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();

        switch (MainActivity.viewMode) {
            case MainActivity.VIEW_MODE_RGBA:
                // 일반 모드를 선택했기 때문에 아무 작업도 해주지 않습니다.

                if(takingPicture){
                    saveImg(rgba);
                }

                break;

            case MainActivity.VIEW_MODE_BLUR:
                // input 영상에 blur 처리하여 return 해줍니다.
                // size 안의 숫자를 크게 할 수록 심하게 blur 처리 됩니다.
                Imgproc.blur(rgba, rgba, new Size(45, 45));

                if(takingPicture){
                    saveImg(rgba);
                }

                break;

            case MainActivity.VIEW_MODE_SEPIA:
                // sepia 색 정보를 담아두는 mat입니다.
                mSepiaKernel = new Mat(4, 4, CvType.CV_32F);
                mSepiaKernel.put(0, 0, /* R */0.189f, 0.769f, 0.393f, 0f);
                mSepiaKernel.put(1, 0, /* G */0.168f, 0.686f, 0.349f, 0f);
                mSepiaKernel.put(2, 0, /* B */0.131f, 0.534f, 0.272f, 0f);
                mSepiaKernel.put(3, 0, /* A */0.000f, 0.000f, 0.000f, 1f);
                // inputframe에 sepia 색 정보를 입혀서 return합니다.
                Core.transform(rgba, rgba, mSepiaKernel);

                mSepiaKernel.release();

                if(takingPicture){
                    saveImg(rgba);
                }

                break;

            case MainActivity.VIEW_MODE_POSTERIZE:
                Imgproc.Canny(rgba, mIntermediateMat, 80, 90);
                rgba.setTo(new Scalar(0, 0, 0, 255), mIntermediateMat);
                Core.convertScaleAbs(rgba, mIntermediateMat, 1./16, 0);
                Core.convertScaleAbs(mIntermediateMat, rgba, 16, 0);

                if(takingPicture){
                    saveImg(rgba);
                }

                break;

            case MainActivity.VIEW_MODE_PIXELIZE:
                // Size의 크기로 픽셀들을 뭉뜨려줍니다. Size 안의 숫자가 클수록 더 심하게 모자이크처리됩니다.
                Imgproc.resize(rgba, mIntermediateMat, new Size(30, 30), 0.1, 0.1, Imgproc.INTER_NEAREST);
                Imgproc.resize(mIntermediateMat, rgba, rgba.size(), 0., 0., Imgproc.INTER_NEAREST);

                if(takingPicture){
                    saveImg(rgba);
                }

                break;

            case MainActivity.VIEW_MODE_NEGATIVE:
//                Mat rgbaInnerWindow = inputFrame.rgba();
//                Mat rgbaInnerWindow2 = inputFrame.rgba();
//                convertNegative(rgbaInnerWindow.getNativeObjAddr(), rgbaInnerWindow2.getNativeObjAddr());
//                rgbaInnerWindow2.copyTo(rgba);
//                rgbaInnerWindow.release();
//                rgbaInnerWindow.release();
                break;

            case MainActivity.VIEW_MODE_CANNY:
                Imgproc.Canny(rgba, mIntermediateMat, 80, 90);
                Imgproc.cvtColor(mIntermediateMat, rgba, Imgproc.COLOR_GRAY2BGRA, 4);

                if(takingPicture){
                    saveImg(rgba);
                }

                break;

            case MainActivity.VIEW_MODE_SOBEL:
                // Sobel 필터를 씌우기 위해서는 Gray Frame이 필요합니다.
                // 따라서 inputFrame를 gray로 변환한 데이터를 Mat에 담습니다.
                Mat gray = inputFrame.gray();
                // gray 영상에 sobel 처리를 해줍니다.
                Imgproc.Sobel(gray, mIntermediateMat, CvType.CV_8U, 1, 1);
                // 윤곽선을 얼마나 많이 추출할지 alpha 값을 입력해줍니다.
                // 큰 값을 입력할수록 더 민감하게 추출해냅니다.
                Core.convertScaleAbs(mIntermediateMat, mIntermediateMat, 3, 0);
                // 추출할 값을 적용한 결과를 rgba에 적용하여 return할 수 있도록 합니다.
                Imgproc.cvtColor(mIntermediateMat, rgba, Imgproc.COLOR_GRAY2BGRA, 4);

                gray.release();

                if(takingPicture){
                    saveImg(rgba);
                }

                break;

            case MainActivity.VIEWM_MODE_BACKGROUND:
                // RGBA Frame > RGB Frame > backgroundsubtractor apply 순서로 처리하는 부분입니다.
                Mat brgba = inputFrame.rgba();
                Mat brgb = new Mat();
                Mat bmask = new Mat();
                Imgproc.cvtColor(brgba, brgb, Imgproc.COLOR_RGBA2RGB);
                bsMOG2.setDetectShadows(false);
                bsMOG2.apply(brgb, bmask, 0.01);
                // backgroundsubtractor apply의 결과가 gray 이기 때문에 움직이는 물체에 색상을 입히기 위해서는 rgba frame을 이용해야 합니다.
                // backgroundsubtractor apply의 결과은 bmask를 mask로 활용하여 rgba에 입혀줍니다.
                Mat zeros = Mat.zeros(rgba.rows(), rgba.cols(), rgba.type());
                rgba.copyTo(zeros, bmask);
                rgba = zeros.clone();

                brgba.release();
                brgb.release();
                bmask.release();
                zeros.release();

                if(takingPicture){
                    saveImg(rgba);
                }

                break;
        }

        return rgba;
    }

    public void saveImg(Mat rgba){
        // 사진 촬영 요청을 성공적으로 받았으므로 재요청하지 않도록 takingPicture 값을 false로 바꾸ㅃ니다.
        takingPicture = false;

        // 저장할 사진의 파일 이름을 시간으로 설정합니다.
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
        String currentDateandTime = sdf.format(new Date());

        // 사진을 저장할 Directory 경로를 설정합니다.
        File storageDir = new File(Environment.getExternalStorageDirectory().getAbsolutePath() + "/MyFilterCamera/");
        // 위에서 설정한 폴더가 없을 경우 생성해주는 부분입니다.
        if (!storageDir.exists()) {
            storageDir.mkdirs();
        }

        // 완성된 사진 경로입니다.
        String imgName = Environment.getExternalStorageDirectory().getPath()+"/MyFilterCamera/" + currentDateandTime + ".jpg";

        // 그냥 저장하면 사진이 푸른 색으로 저장됩니다.
        // BGR 색상 타입을 RGBA 타입으로 변경해줍니다.
        Imgproc.cvtColor(rgba, rgba,  Imgproc.COLOR_BGR2RGBA, 4);

        // opencv 함수를 이용하여 저장합니다.
        takenPicture = Imgcodecs.imwrite(imgName, rgba);
        pictureHandler.post(new Runnable() {
            @Override
            public void run() {
                if(takenPicture){
                    Toast.makeText(getApplicationContext(), "Success", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(getApplicationContext(), "Fail", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }
}
