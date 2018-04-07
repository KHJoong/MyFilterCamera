package com.example.myfiltercamera;

import android.content.res.AssetManager;
import android.media.CamcorderProfile;
import android.media.MediaRecorder;
import android.media.MediaScannerConnection;
import android.net.Uri;
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
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

public class MainActivity extends AppCompatActivity implements JavaCameraView.CvCameraViewListener2 {

    // Native 함수를 사용하기 위해 먼저 선언해두는 부분입니다.
    // 이 함수는 Negative filter에서 사용하고자 했었습니다.
    // 미완성입니다.
//    public native void convertNegative(long matAddrInput, long matAddrResult);

    // 아래 native 함수들은 face detection을 위해 추가한 함수입니다.
    public static native long loadCascade(String cascadeFileName );
    public static native void detect(long cascadeClassifier_face, long cascadeClassifier_eye, long matAddrInput, long matAddrResult);
    public long cascadeClassifier_face = 0;
    public long cascadeClassifier_eye = 0;

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
    // 영상 녹화하기 위한 버튼입니다.(녹화중에는 중지, 녹화중이 아닐 때는 녹화로 표시됩니다.)
    Button btnTakeVideo;
    // 전면 카메라, 후면 카메라 전환할 수 있는 스위치 버튼입니다.
    Button btnSwitch;

    // SubBackground 필터에 쓰일 class입니다.
    // 픽셀에 변화가 생기는 부분을 찾아주는 역활을 합니다.
    BackgroundSubtractorMOG2 bsMOG2;

    // 사진 찍기 버튼(btnTakePic)을 클릭한 경우 true로 변경되고 onCameraFrame > saveImg 순으로 촬영이 진행됩니다.
    // 촬영이 완료되면 false로 다시 변경됩니다.
    boolean takingPicture = false;
    // 사진 촬영이 성공했는지 실패했는지에 따라 다른 Toast 메시지를 띄워주기 위해 사용되는 변수입니다.
    boolean takenPicture = false;
    // Toast를 띄울 때 사용하기 위한 Handler입니다.
    Handler handler;

    // 영상 녹화 버튼(btnTakeVideo)을 클릭한 경우 true로 변경되고 mr(MediaRecorder)에 의해서 녹화가 시작됩니다.
    // 완성된 영상은 videoPath에 저장됩니다.
    // 영상 녹화를 위해 CameraBridgeViewBase class가 수정되었습니다.
    // 1) deliverAndDrawFrame에서 canvas를 생성하고 그리는 부분에 영상녹화에서 사용되는 부분을 별도로 추가하였습니다.
    // 2) setRecorder, releaseRecord 함수가 맨 위에 추가되었습니다. 녹화를 시작할 때 setRecorder이 실행되고 끝날 때 releaseRecord가 실행됩니다.
    boolean takingVideo = false;
    String videoPath;
    MediaRecorder mr;

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
        btnTakeVideo = (Button)findViewById(R.id.btnTakeVideo);
        btnTakePic.setOnClickListener(btnClickListener);
        btnTakeVideo.setOnClickListener(btnClickListener);
        handler = new Handler();
        btnSwitch = (Button)findViewById(R.id.btnSwitch);
        btnSwitch.setOnClickListener(btnClickListener);

        // 처음 실행될 때 후면 카메라를 기본 실행하도록 합니다.
        cbvCamera.setCameraIndex(0);

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

        read_cascade_file();
    }

    Button.OnClickListener btnClickListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            switch (v.getId()){
                case R.id.btnTakePic:
                    // 사진 촬영 버튼을 클릭했을 때 실행되는 부분입니다.
                    takingPicture = true;
                    break;
                case R.id.btnTakeVideo:
                    // 영상 녹화 버튼을 클릭했을 때 실행되는 부분입니다.
                    // 버튼의 text를 바꾸기 위해서 핸들러를 사용합니다.
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            if(!takingVideo){
                                // 한 버튼으로 녹화 시작과 끝을 구분합니다.
                                // takingVideo 변수가 true면 녹화중, false면 대기모드로 구분합니다.
                                takingVideo = true;

                                Toast.makeText(getApplicationContext(), "녹화를 시작합니다.", Toast.LENGTH_SHORT).show();

                                // 저장할 영상 파일의 경로를 받아옵니다.
                                // type에 video를 넘겨줄 경우 mp4 확장자로 된 경로를 받아옵니다.
                                videoPath = savefilepath("video");

                                try {
                                    // MediaRecorder 초기화하는 부분입니다.
                                    mr = new MediaRecorder();
                                    mr.setAudioSource(MediaRecorder.AudioSource.MIC);
                                    mr.setVideoSource(MediaRecorder.VideoSource.SURFACE);
                                    // 이 부분의 주석을 해제하면 바로 아래 setProfile을 주석처리해야합니다.
                                    // 자세한 설정이 가능하지만 화질이 떨어집니다.
//                                    mr.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
//                                    mr.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
//                                    mr.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
//                                    mr.setVideoFrameRate(120);
//                                    mr.setAudioSamplingRate(16000);
//                                    mr.setVideoSize(cbvCamera.getWidth(), cbvCamera.getHeight());
                                    // 화질을 위해 위 부분을 주석처리하고 setProfile을 추가합니다.
                                    // 위의 설정을 주석처리하지 않는 경우 에러가 발생합니다.
                                    mr.setProfile(CamcorderProfile.get(CamcorderProfile.QUALITY_HIGH));
                                    mr.setOutputFile(videoPath);
                                    mr.prepare();

                                    // MediaRecorder 준비가 완료될 경우 CameraBridgeViewBase에
                                    // MediaRecorder를 넘겨줘서 녹화할 수 있도록 합니다.
                                    // OpenCV 기본 함수가 아닌 사용자 추가 함수입니다.
                                    cbvCamera.setRecorder(mr);

                                    // 설정이 완료되면 녹화를 시작합니다.
                                    mr.start();
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                                // 버튼의 text를 중지로 바꿔서 녹화 정지를 할 수 있게 유도합니다.
                                btnTakeVideo.setText("중지");
                            } else {
                                // 한 버튼으로 녹화 시작과 끝을 구분합니다.
                                // takingVideo 변수가 true면 녹화중, false면 대기모드로 구분합니다.
                                takingVideo = false;

                                Toast.makeText(getApplicationContext(), "녹화를 종료합니다.", Toast.LENGTH_SHORT).show();

                                if(mr != null){
                                    // 선언했던 MediaRecorder를 멈추고 해제해줍니다.
                                    mr.stop();
                                    mr.release();
                                    mr = null;

                                    // 해제 완료되면 CameraBridgeViewBase에서도 쓰지 못하게 해제해줍니다.
                                    // OpenCV 기본 함수가 아닌 사용자 추가 함수입니다.
                                    cbvCamera.releaseRecord();
                                }

                                // 갤러리에 나타나게 하는 부분입니다.
                                MediaScannerConnection.scanFile(getApplicationContext(),
                                        new String[]{videoPath}, null,
                                        new MediaScannerConnection.OnScanCompletedListener() {
                                            public void onScanCompleted(String path, Uri uri) {
                                            }
                                        });

                                // 버튼의 text를 녹화로 바꿔서 새로운 녹화를 할 수 있게 유도합니다.
                                btnTakeVideo.setText("녹화");
                            }
                        }
                    });
                    break;
                case R.id.btnSwitch:
                    // 전면 카메라, 후면 카메라를 전환하는 버튼입니다.
                    // 0이 후면 카메라, 1이 전면 카메라입니다.
                    if(cbvCamera.getCameraIndex()==0){
                        cbvCamera.disableView();
                        cbvCamera.setCameraIndex(1);
                        cbvCamera.enableView();
                    } else if(cbvCamera.getCameraIndex()==1){
                        cbvCamera.disableView();
                        cbvCamera.setCameraIndex(0);
                        cbvCamera.enableView();
                    }
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

//        Core.flip(rgba, rgba, 1);
        // 이 부분은 얼굴 탐지를 위해 사용되는 부분입니다.
        detect(cascadeClassifier_face, cascadeClassifier_eye, rgba.getNativeObjAddr(), rgba.getNativeObjAddr());

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

    // 저장할 사진 or 영상의 실제 경로를 반환해주는 함수입니다.
    // type이 img면 사진(.jpg), video면 영상(.mp4)의 경로를 반환합니다.
    public String savefilepath(String type){
        String filepath = "";

        // 저장할 사진의 파일 이름을 시간으로 설정합니다.
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
        String currentDateandTime = sdf.format(new Date());

        // 사진을 저장할 Directory 경로를 설정합니다.
        File storageDir = new File(Environment.getExternalStorageDirectory().getAbsolutePath() + "/MyFilterCamera/");
        // 위에서 설정한 폴더가 없을 경우 생성해주는 부분입니다.
        if (!storageDir.exists()) {
            storageDir.mkdirs();
        }

        if(type.equals("img")) {
            // 완성된 사진 경로입니다.
            filepath = Environment.getExternalStorageDirectory().getPath() + "/MyFilterCamera/" + currentDateandTime + ".jpg";
        } else if(type.equals("video")){
            // 완성된 비디오 경로입니다.
            filepath = Environment.getExternalStorageDirectory().getPath() + "/MyFilterCamera/" + currentDateandTime + ".mp4";
        }

        return filepath;
    }

    // 사진을 저장하는 함수입니다.
    // onCameraFrame에서 takingPicture의 값이 true일 때 실행됩니다.
    public void saveImg(Mat rgba){
        // 사진 촬영 요청을 성공적으로 받았으므로 재요청하지 않도록 takingPicture 값을 false로 바꾸ㅃ니다.
        takingPicture = false;

        // 받아온 사진 경로입니다.
        String imgName = savefilepath("img");

        // 그냥 저장하면 사진이 푸른 색으로 저장됩니다.
        // BGR 색상 타입을 RGBA 타입으로 변경해줍니다.
        Imgproc.cvtColor(rgba, rgba,  Imgproc.COLOR_BGR2RGBA, 4);

        // opencv 함수를 이용하여 저장합니다.
        takenPicture = Imgcodecs.imwrite(imgName, rgba);

        // 갤러리에 나타나게 하는 부분입니다.
        MediaScannerConnection.scanFile(this,
                new String[]{imgName}, null,
                new MediaScannerConnection.OnScanCompletedListener() {
                    public void onScanCompleted(String path, Uri uri) {
                    }
                });

        handler.post(new Runnable() {
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

    // 이 부분은 얼굴 탐지를 위해 사용되는 부분입니다.
    // CascadeClassifier 객체를 로드하는 함수입니다.
    // 작동 과정 : asset > 기본 위치(Environment.getExternalStorageDirectory().getPath()) > CascadeClassifier 객체로 로드
    public void read_cascade_file(){
        // 로드하기 전에 기기의 기본 위치에 파일을 복사합니다.
        copyFile("haarcascade_frontalface_default.xml");
        copyFile("haarcascade_eye.xml");

        // loadCascade 메소드는 외부 저장소의 특정 위치에서 해당 파일을 읽어와서 CascadeClassifier 객체로 로드합니다.
        cascadeClassifier_face = loadCascade( "haarcascade_frontalface_default.xml");
        cascadeClassifier_eye = loadCascade( "haarcascade_eye.xml");
    }

    // 이 부분은 얼굴 탐지를 위해 사용되는 부분입니다.
    // Assets에서 filename에 해당하는 파일을 가져와 기기의 기본 위치에 저장하는 함수입니다.
    // 기본 위치 : Environment.getExternalStorageDirectory().getPath()
    public void copyFile(String filename){
        String baseDir = Environment.getExternalStorageDirectory().getPath();
        String pathDir = baseDir + File.separator + filename;

        AssetManager assetManager = this.getAssets();

        InputStream inputStream = null;
        OutputStream outputStream = null;

        try {
            inputStream = assetManager.open(filename);
            outputStream = new FileOutputStream(pathDir);

            byte[] buffer = new byte[1024];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            inputStream.close();
            inputStream = null;
            outputStream.flush();
            outputStream.close();
            outputStream = null;
        } catch (Exception e) {
            e.getStackTrace();
        }
    }
}
