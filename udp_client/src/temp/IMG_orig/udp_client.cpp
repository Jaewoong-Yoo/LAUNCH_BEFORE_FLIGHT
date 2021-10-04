#include "header.h"

#define OWNSHIP_ON
#define INTRUDER_ON
#define PRINT

// #define GPS_POS
#define ADSB_POS

// setup the initial name
using namespace ros;
using namespace std;

int a = 0;
int CAM_SELECTION = 0; // 0: Front, 1: Left, 2: Right

double bag_own_lat;
double bag_own_lon;
double bag_own_alt;

double bag_int_lat;
double bag_int_lon;
double bag_int_alt;
float intr_bearing   = 0;
float intr_est_range = 0; 

float RefECEFX_m        = 0.0;
float RefECEFY_m        = 0.0;
float RefECEFZ_m        = 0.0;

// for publishing the kalman states
std_msgs::Float32MultiArray     receive_data;
ads_b_read::Traffic_Report_Array TR_Data_Array;
ads_b_read::Traffic_Report       OWNSHIP_DATA;
ads_b_read::Traffic_Report       INTRUDER_DATA;

///////

#define eps           0.00000001
#define PI            3.1415926535

float   CAMERA_PARAM_U0 = 960.0;
float   CAMERA_PARAM_V0 = 540.0;

float   size_pre[2];
float   img_pre[2];

int     count_target = 0;
int     count_init = 0;
float   dist_pre     = 0.0;
float   diag_pre     = 0.0;
float   pos_pre[2];

float   bearing_angle   = 0.0;
float   ownship_heading = 0.0;
float   ownship_local[2];
float   camsel[2];
float   vel_abs = 0.0;
float   vel_dir = 0.0;
float   vel_abs_lpf = 0.0;
float   vel_dir_lpf = 0.0;
float   vel_abs_pre = 50.0;
float   vel_dir_pre = 0.0;

// Kalman filter
int stateSize   = 4;
int measSize    = 2;
int contrSize   = 0;

int zerosize = 0;
int fivesize = 5;
unsigned int type = CV_32F;

cv::KalmanFilter kf(stateSize, measSize, contrSize, type);
cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y]
cv::Mat meas(measSize, 1, type);    // [z_x,z_y]

void LatLontoECEF(float lat, float lon, float h, float *ECEFX_m, float *ECEFY_m, float *ECEFZ_m);

struct struct_Tar_data
{
    float   pos[3];
    float   poslpf[2];
    float   posfil[2];
    float   vel[3];
    float   impos[2];
    float   imposfil[2];
    float   size[2];
    float   sizefil[2];
    float   dist;

    float   psi_ref;
    float   a_n;
    float   del_sigma;

    uint8_t flag_detect;
};

struct struct_Tar_data        tar_data;


float satmax(float data, float max)
{
    float res;
    if(fabs(data) > max)
        res = (data + eps)/fabs(data + eps)*max;
    else
        res = data;
    return res;
}


float satmin(float data, float min)
{
    float res;
    if(fabs(data) < min)
        res = (data + eps)/fabs(data + eps)*min;
    else
        res = data;
    return res;
}

float LPF(float data, float data_pre, float freq)
{
    float res;
    float K_LPF;
    float delT = 0.05;

    K_LPF = freq*2*PI*delT / (1 + freq*2*PI*delT);

    res = (data - data_pre) * K_LPF + data_pre;

    return res;
}


void KalmanFilter_Init(void)
{

   //========================== Kalman Filter variable =============================//
    cv::setIdentity(kf.transitionMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar(0.001));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(0.01));

    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0;
    kf.measurementMatrix.at<float>(5) = 1.0;

    kf.processNoiseCov.at<float>(10) = 0.001;
    kf.processNoiseCov.at<float>(15) = 0.001;

    kf.errorCovPre.at<float>(0) = 1.0; // px
    kf.errorCovPre.at<float>(5) = 1.0; // px
    kf.errorCovPre.at<float>(10) = 1.0;
    kf.errorCovPre.at<float>(15) = 1.0;

    state.at<float>(0) = tar_data.pos[0];
    state.at<float>(1) = tar_data.pos[1];
    state.at<float>(2) = 0.0;
    state.at<float>(3) = 0.0;

    kf.statePost = state;
}


void kalmanfilter(void)
{
    float dT = 0.05;

    kf.transitionMatrix.at<float>(2) = dT;
    kf.transitionMatrix.at<float>(7) = dT;

    state = kf.predict();
    //cout << "State post:" << endl << state << endl;

    meas.at<float>(0) = tar_data.poslpf[0];
    meas.at<float>(1) = tar_data.poslpf[1];

    kf.correct(meas);

    tar_data.posfil[0] = state.at<float>(0);
    tar_data.posfil[1] = state.at<float>(1);
    tar_data.vel[0] = state.at<float>(2);
    tar_data.vel[1] = state.at<float>(3);
}

// ==========================================//


void distance_filter(void)
{

    if (count_init < 40)
    {
        float diag = sqrt(tar_data.sizefil[0]*tar_data.sizefil[0] + tar_data.sizefil[1]*tar_data.sizefil[1]);
        diag = satmax(satmin(diag, 10.0),50.0);
        diag_pre = diag;

            if (CAM_SELECTION==1){
                //tar_data.dist = 107.42*pow(diag,-1.307);   // Fitting!!!
                tar_data.dist = 36.996*pow(diag,-0.963);   // Fitting!!!
            }
            else if(CAM_SELECTION==2){
                tar_data.dist = 16368.0*pow(diag,-2.758);   // Fitting!!!
            }
            else{

            }
        dist_pre = 3000.0;
        pos_pre[0] = tar_data.pos[0];
        pos_pre[1] = tar_data.pos[1];
        count_init = count_init + 1;
        cout << "init_task" << endl;
    }

    float OwnECEFX_m        = 0.0;
    float OwnECEFY_m        = 0.0;
    float OwnECEFZ_m        = 0.0;
    LatLontoECEF((double)OWNSHIP_DATA.lat/10000000.0, (double)OWNSHIP_DATA.lon/10000000.0, (double)OWNSHIP_DATA.altitude/1000.0, &OwnECEFX_m, &OwnECEFY_m, &OwnECEFZ_m);

    if (tar_data.flag_detect == 1)
    {
        tar_data.imposfil[0] = tar_data.impos[0];
        tar_data.imposfil[1] = tar_data.impos[1];
        img_pre[0] = tar_data.impos[0];
        img_pre[1] = tar_data.impos[1];

        tar_data.sizefil[0] = tar_data.size[0];
        tar_data.sizefil[1] = tar_data.size[1];
        size_pre[0] = tar_data.size[0];
        size_pre[1] = tar_data.size[1];
        count_target = 0;
    }
    else
    {
        tar_data.imposfil[0] = img_pre[0];
        tar_data.imposfil[1] = img_pre[1];
        tar_data.sizefil[0]  = size_pre[0];
        tar_data.sizefil[1]  = size_pre[1];
    }

    float diag = sqrt(tar_data.sizefil[0]*tar_data.sizefil[0] + tar_data.sizefil[1]*tar_data.sizefil[1]);

    float diag_lpf = LPF(diag, diag_pre, 0.01);
    diag_pre = diag_lpf;
    diag = satmax(satmin(diag, 10.0),50.0);

    ///////// For fitting
    // tar_data.dist = 16368.0*pow(diag,-2.758);
    tar_data.dist = 36.996*pow(diag,-0.963);   // Fitting!!!

    if (CAM_SELECTION==1){
        //tar_data.dist = 107.42*pow(diag,-1.307);   // Fitting!!!
        tar_data.dist = 36.996*pow(diag,-0.963);   // Fitting!!!
    }
    else if(CAM_SELECTION==2){
        tar_data.dist = 36.996*pow(diag,-0.963);   // Fitting!!!
    }
    else{

    }

    tar_data.dist = satmax(satmin(tar_data.dist, 1.0), 4.0);

    tar_data.dist = LPF(tar_data.dist, dist_pre, 0.06);
    dist_pre = tar_data.dist;

    float psi_img = intr_bearing*D2R;  //[rad]

    ownship_heading = ((float)OWNSHIP_DATA.heading / 100.0)*D2R;

    tar_data.pos[0] = (OwnECEFY_m + tar_data.dist*1000 * cos(psi_img + ownship_heading) - RefECEFY_m)/1000.0;
    tar_data.pos[1] = (OwnECEFX_m + tar_data.dist*1000 * sin(psi_img + ownship_heading) - RefECEFX_m)/1000.0;

    tar_data.poslpf[0] = LPF(tar_data.pos[0], pos_pre[0], 1.0);
    tar_data.poslpf[1] = LPF(tar_data.pos[1], pos_pre[1], 1.0);

    pos_pre[0] = tar_data.poslpf[0];
    pos_pre[1] = tar_data.poslpf[1];

    // intr_est_range = sqrt( tar_data.posfil[0]*tar_data.posfil[0] + tar_data.posfil[1]*tar_data.posfil[1]  )*1000.0/1852;
    kalmanfilter();

    vel_abs = sqrt(tar_data.vel[0]*tar_data.vel[0] + tar_data.vel[1]*tar_data.vel[1])*1000.0;
    vel_dir = atan2(tar_data.vel[1], tar_data.vel[0]);

    vel_abs_lpf = LPF(vel_abs, vel_abs_pre, 0.5);
    vel_dir_lpf = LPF(vel_dir, vel_dir_pre, 0.5);

    vel_abs_pre = vel_abs_lpf;
    vel_dir_pre = vel_dir_lpf;

    intr_est_range = tar_data.dist;
    cout <<"HS " << vel_abs_lpf << ", " << vel_dir_lpf << endl;
    cout <<"HS " << tar_data.poslpf[0] << ", " <<tar_data.poslpf[1] << endl;
    cout <<"HS " << tar_data.posfil[0] << ", " <<tar_data.posfil[1] << ", " <<tar_data.vel[0] << ", " <<tar_data.vel[1]<< endl;
}

////////


double UAV0_gposition[3];
double UAV1_gposition[3];
float UAV0_linear_acc_local[3];
float UAV0_orientation_local[4];
float UAV0_orientation_Euler_local[3];
float UAV0_anglular_velocity_local[3];
float UAV1_linear_acc_local[3];
float UAV1_orientation_local[4];
float UAV1_orientation_Euler_local[3];
float UAV1_anglular_velocity_local[3];
uint  UDP_tx_data[18];
char  callsign_print[100][9];
float detection_result[5];

struct struct_t_UDP           StrUDP;
struct sockaddr_in            MyAddr;
struct RX_message_data        RX_buff;
struct TX_message_data        TX_buff;
struct struct_DATA            TX_data;
struct struct_udp_test        TX_test;
struct struct_udp             TX_struct;      

void LatLontoECEF(float lat, float lon, float h, float *ECEFX_m, float *ECEFY_m, float *ECEFZ_m);
void camsel_Callback(const std_msgs::Float32MultiArray& array);
void detectionCallback(const std_msgs::Float32MultiArray& array);
void arrayCallback(const std_msgs::UInt32MultiArray& array);
void UAV0_global_position(const sensor_msgs::NavSatFix::ConstPtr& msg);
void UAV1_global_position(const sensor_msgs::NavSatFix::ConstPtr& msg);
void UAV0_imu_callback(const sensor_msgs::Imu::ConstPtr& imu_msg);
void UAV1_imu_callback(const sensor_msgs::Imu::ConstPtr& imu_msg);
void TR_Data_Array_callback(const ads_b_read::Traffic_Report_Array::ConstPtr& msg_input);
void rawfixcallback_Own(const sensor_msgs::NavSatFix::ConstPtr& msg);
void rawfixcallback_Int(const sensor_msgs::NavSatFix::ConstPtr& msg);
static void QuaterniontoEuler(float orientation_x,float orientation_y,float orientation_z,float orientation_w, float& roll, float& pitch, float& yaw);

char IntToChar(int aaa);
void check_odd (int data[32], int& odd_data);
void PACKET(int index[32], uint& sum_num);
uint IntToBinary(int num, int location);

// node main loop, for ROS
int main(int argc, char** argv)
{
	// node name initialization
    init(argc, argv, "udp_client");
    clock_t start_time;
    start_time = clock();

	// assign node handler
	ros::NodeHandle nh_;

	// for debugging
	printf("Initiate: UDP Client\n");

	// Publish
	// Publisher  P_UAV	= nh_.advertise<std_msgs::UInt32MultiArray>("/TX_udp", 100);			

	//Subscribe
	// ros::Subscriber sub0 = nh_.subscribe("/TX_udp",100,arrayCallback);
    ros::Subscriber sub0 = nh_.subscribe("/detection",1,detectionCallback);
    ros::Subscriber sub00 = nh_.subscribe("/cam_select",1,camsel_Callback);
	ros::Subscriber sub1 = nh_.subscribe("/uav0/mavros/global_position/global", 1, UAV0_global_position);
    ros::Subscriber sub2 = nh_.subscribe("/uav1/mavros/global_position/global", 1, UAV1_global_position);
    ros::Subscriber sub3 = nh_.subscribe<sensor_msgs::Imu>("/uav0/mavros/imu/data",1000,UAV0_imu_callback);
    ros::Subscriber sub4 = nh_.subscribe<sensor_msgs::Imu>("/uav1/mavros/imu/data",1000,UAV1_imu_callback);
    // ros::Subscriber sub5 = nh_.subscribe("/ADS_B/Traffic_Report_Array/ownship",1, TR_Data_Array_callback);
    ros::Subscriber sub5 = nh_.subscribe("/ADS_B/Traffic_Report_Array",1, TR_Data_Array_callback);

    ros::Subscriber bag_Own_GPS = nh_.subscribe("mavros/global_position/raw/fix/ownship",  1000, rawfixcallback_Own);
    ros::Subscriber bag_Int_GPS = nh_.subscribe("mavros/global_position/raw/fix/intruder", 1000, rawfixcallback_Int);

    receive_data.data.resize(10);

	// setup the loop speed, [Hz], synchronizing the hector slam loop
    ros::Rate loop_rate(20);
	float fdt = (float)(1/1);

    // Socket Creation
    StrUDP.Socket = socket(PF_INET, SOCK_DGRAM, 0);
    if(StrUDP.Socket == -1){
        printf("[ERROR] 'socket()'\n");
        return -1;
    }
    else{
        printf("[DONE] UDP socket is created\n");
    }

    // UDP-IP Setting_Client
    memset(&StrUDP.ServerAddr, 0, sizeof(StrUDP.ServerAddr));          // Clear to 0
    StrUDP.ServerAddr.sin_family      = PF_INET;
    StrUDP.ServerAddr.sin_port        = htons(DF_UDP_PORTNUM);         // PORT#
    StrUDP.ServerAddr.sin_addr.s_addr = inet_addr(DF_UDP_SERVER_ADDR); // IP for Server (Normally PC IP)

    memset(&MyAddr, 0, sizeof(MyAddr));
    MyAddr.sin_family = PF_INET;
    MyAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    MyAddr.sin_port = htons(DF_UDP_PORTNUM);

    if(bind(StrUDP.Socket,(struct sockaddr *)&MyAddr, sizeof(MyAddr))!=0){
        printf("bind1() error!\n");
        return -1;
    }

    char sendBuff = 1;
    int size_addr = sizeof(StrUDP.ServerAddr);
    //printf("[DONE] UDP initialized! size : %d\n",size_addr);

    if(setsockopt(StrUDP.Socket,SOL_SOCKET, SO_RCVTIMEO, &tv_timeo, sizeof(tv_timeo)) == -1){
        printf("setsockopt error!\n");
        return -1;
    }
    double first_begine = ros::Time::now().toSec();
    double process_time;


    LatLontoECEF(std_lat, std_lon, 10000.0, &RefECEFX_m, &RefECEFY_m, &RefECEFZ_m);



    // HS KM
    KalmanFilter_Init();

    // ros::Rate loop_rate(20);
	while( ok() ){
        distance_filter();

        // Time check
        double begine = ros::Time::now().toSec();
        process_time = begine -first_begine ;
        cout <<  begine -first_begine << endl;

        int T_sec = floor(process_time);
        int T_min = floor(process_time/60);
        int T_hour = floor(process_time/3600);
        while(T_sec > 60){ 
            if (T_sec > 60){
                T_sec = T_sec - 60;
            }
        }

        while(T_min > 60){ 
            if (T_min > 60){
                T_min = T_min - 60;
            }
        }
        cout << T_hour << ":" << T_min << ":" << T_sec << endl;


        ///// TX_struct.Label367 ///// ATTACHMENT 20D
        // 0b 11110111 00010010 10000000 00000010
        int label367[32];
    /// BIT          = Value    Function                            Coding
        label367[ 0] = 1;       // Label 1st Digit(MSB)             3:  1
        label367[ 1] = 1;       // Label 1st Digit(LSB)                 1
        label367[ 2] = 1;       // Label 2nd Digit(MSB)             6:  1
        label367[ 3] = 1;       // Label 2nd Digit                      1
        label367[ 4] = 0;       // Label 2nd Digit(LSB)                 0
        label367[ 5] = 1;       // Label 3rd Digit(MSB)             7:  1
        label367[ 6] = 1;       // Label 3rd Digit                      1
        label367[ 7] = 1;       // Label 3rd Digit(LSB)                 1

        label367[ 8] = 0;       // Number of Words in DRIF-1            
        label367[ 9] = 0;       // Number of Words in DRIF-2
        label367[10] = 0;       // Number of Words in DRIF-4
        label367[11] = 1;       // Number of Words in DRIF-8
        label367[12] = 0;       // Number of Words in DRIF-16
        label367[13] = 0;       // Number of Words in DRIF-32
        label367[14] = 1;       // Number of Words in DRIF-64
        label367[15] = 0;       // Number of Words in DRIF-128
        
        label367[16] = 1;       // Number of Words in DRIF-256
        label367[17] = 0;       // Number of Words in DRIF-512
        label367[18] = 0;       // Number of Words in DRIF-1024
        label367[19] = 0;       // Number of Words in DRIF-2048
        label367[20] = 0;       // Pad                                  0
        label367[21] = 0;       // Pad                                  0
        label367[22] = 0;       // Pad                                  0
        label367[23] = 1;       // All Traffic / Treat Traffic

        label367[24] = 0;       // ISO #5 Character STX(0/2) (LSB)      0
        label367[25] = 0;       // ISO #5 Character STX(0/2)            1
        label367[26] = 0;       // ISO #5 Character STX(0/2)            0
        label367[27] = 0;       // ISO #5 Character STX(0/2)            0
        label367[28] = 0;       // ISO #5 Character STX(0/2)            0
        label367[29] = 0;       // ISO #5 Character STX(0/2)            0
        label367[30] = 1;       // ISO #5 Character STX(0/2)            0
        label367[31] = 0;       // Parity                               (odd)
        check_odd (label367, label367[24]);

        uint label_temp=0b0;
        PACKET(label367, label_temp);
        TX_struct.Label367 = htonl(label_temp);
        

        ///// TX_struct.Label366_DTIF_Header /////
        int DTIF_header[32];
        // 0b 11110110 10000001 10000000 00111111
    /// BIT             = Value     Function                            Coding
        DTIF_header[ 0] = 1;        // Label 1st Digit(MSB)             3:  1
        DTIF_header[ 1] = 1;        // Label 1st Digit(LSB)                 1
        DTIF_header[ 2] = 1;        // Label 2nd Digit(MSB)             6:  1
        DTIF_header[ 3] = 1;        // Label 2nd Digit                      1
        DTIF_header[ 4] = 0;        // Label 2nd Digit(LSB)                 0
        DTIF_header[ 5] = 1;        // Label 3rd Digit(MSB)             6:  1
        DTIF_header[ 6] = 1;        // Label 3rd Digit                      1
        DTIF_header[ 7] = 0;        // Label 3rd Digit(LSB)                 0

        DTIF_header[15] = 1;        // Num.Traffic Packets in DTIF-1       
        DTIF_header[14] = 0;        // Num.Traffic Packets in DTIF-2
        DTIF_header[13] = 0;        // Num.Traffic Packets in DTIF-4
        DTIF_header[12] = 0;        // Num.Traffic Packets in DTIF-8
        DTIF_header[11] = 0;        // Num.Traffic Packets in DTIF-16
        DTIF_header[10] = 0;        // Num.Traffic Packets in DTIF-32
        DTIF_header[ 9] = 0;        // Num.Traffic Packets in DTIF-64
        DTIF_header[ 8] = 1;        // DTIF Version Number(LSB)

        DTIF_header[23] = 0;        // DTIF Version Number
        DTIF_header[22] = 0;        // DTIF Version Number(MSB)
        DTIF_header[21] = 0;        // Pad                                  0
        DTIF_header[20] = 0;        // Reserved for Manufacturer Use        0
        DTIF_header[19] = 0;        // Reserved for Manufacturer Use        0
        DTIF_header[18] = 0;        // Reserved for Manufacturer Use        0
        DTIF_header[17] = 0;        // Pad                                  0
        DTIF_header[16] = 1;        // Display limit - Num.targets-1

        DTIF_header[31] = 0;        // Display limit - Num.targets-2
        DTIF_header[30] = 1;        // Display limit - Num.targets-4
        DTIF_header[29] = 1;        // Display limit - Num.targets-8
        DTIF_header[28] = 1;        // Display limit - Num.targets-16
        DTIF_header[27] = 1;        // Display limit - Num.targets-32
        DTIF_header[26] = 1;        // Display limit - Num.targets-64
        DTIF_header[25] = 0;        // Pad                                  0
        DTIF_header[24] = 0;        // Parity                               (odd)
        check_odd (DTIF_header, DTIF_header[24]);
        
        uint DTIF_header_temp=0b0;
        PACKET(DTIF_header, DTIF_header_temp);
        TX_struct.Label366_DTIF_Header = htonl(DTIF_header_temp);

////OWNSHIP INFORMATION/////////////////////////////////////////////////////////////////////////////////////
///// TX_struct.DTIF_Packet_Header ///// ATTACHMENT 20G
        //0b 1111 0110 1000 0001 0000 0111 1010 0000
        int DTIF_Packet_header[32];
    /// BIT                    = Value      Function                            Coding
        DTIF_Packet_header[ 0] = 1;         // Label 1st Digit(MSB)             3:  1
        DTIF_Packet_header[ 1] = 1;         // Label 1st Digit(LSB)                 1
        DTIF_Packet_header[ 2] = 1;         // Label 2nd Digit(MSB)             6:  1
        DTIF_Packet_header[ 3] = 1;         // Label 2nd Digit                      1
        DTIF_Packet_header[ 4] = 0;         // Label 2nd Digit(LSB)                 0
        DTIF_Packet_header[ 5] = 1;         // Label 3rd Digit(MSB)             6:  1
        DTIF_Packet_header[ 6] = 1;         // Label 3rd Digit                      1
        DTIF_Packet_header[ 7] = 0;         // Label 3rd Digit(LSB)                 0

        DTIF_Packet_header[15] = 1;         // Traffic Number-1
        DTIF_Packet_header[14] = 0;         // Traffic Number-2
        DTIF_Packet_header[13] = 0;         // Traffic Number-4
        DTIF_Packet_header[12] = 0;         // Traffic Number-8
        DTIF_Packet_header[11] = 0;         // Traffic Number-16
        DTIF_Packet_header[10] = 0;         // Traffic Number-32
        DTIF_Packet_header[ 9] = 0;         // Traffic Number-64
        DTIF_Packet_header[ 8] = 1;         // Num.Label in Packet-1

        DTIF_Packet_header[23] = 1;         // Num.Label in Packet-2
        DTIF_Packet_header[22] = 0;         // Num.Label in Packet-4
        DTIF_Packet_header[21] = 1;         // Num.Label in Packet-8
        DTIF_Packet_header[20] = 0;         // Num.Label in Packet-16
        DTIF_Packet_header[19] = 0;         // Reserved for Manufacturer Use
        DTIF_Packet_header[18] = 0;         // Reserved for Manufacturer Use         
        DTIF_Packet_header[17] = 0;         // Display Matrix
        DTIF_Packet_header[16] = 0;         // Display Matrix

        DTIF_Packet_header[31] = 0;         // Display Matrix
        DTIF_Packet_header[30] = 0;         // Display Matrix
        DTIF_Packet_header[29] = 0;         // Source Data Type
        DTIF_Packet_header[28] = 0;         // Source Data Type
        DTIF_Packet_header[27] = 0;         // Source Data Type
        DTIF_Packet_header[26] = 0;         // Air/Ground Status
        DTIF_Packet_header[25] = 0;         // Spare
        DTIF_Packet_header[24] = 0;         // Parity                               (odd)
        check_odd (DTIF_Packet_header, DTIF_Packet_header[24]);

        uint DTIF_packet_header_temp=0b0;
        PACKET(DTIF_Packet_header, DTIF_packet_header_temp);
        TX_struct.DTIF_Packet_Header = htonl(DTIF_packet_header_temp);

///// TX_struct.TYPE_0000_0 ///// ATTACHMENT 20H-1
        int data_0000_0[32];
        //0b 1111 0110 1111 0000 0110 1000 1000 0001
    /// BIT             = Value      Function                               Coding
        data_0000_0[ 0] = 1;         // Label 1st Digit(MSB)                3:  1
        data_0000_0[ 1] = 1;         // Label 1st Digit(LSB)                    1
        data_0000_0[ 2] = 1;         // Label 2nd Digit(MSB)                6:  1
        data_0000_0[ 3] = 1;         // Label 2nd Digit                         1
        data_0000_0[ 4] = 0;         // Label 2nd Digit(LSB)                    0
        data_0000_0[ 5] = 1;         // Label 3rd Digit(MSB)                6:  1
        data_0000_0[ 6] = 1;         // Label 3rd Digit                         1
        data_0000_0[ 7] = 0;         // Label 3rd Digit(LSB)                    0

        data_0000_0[15] = 0;         // Data Type(LSB)                      0:  0
        data_0000_0[14] = 0;         // Data Type                               0
        data_0000_0[13] = 0;         // Data Type                               0
        data_0000_0[12] = 0;         // Data Type(MSB)                          0
        data_0000_0[11] = 0;         // Character#1(LSB)        0
        data_0000_0[10] = 0;         // Character#1             1
        data_0000_0[ 9] = 0;         // Character#1             0
        data_0000_0[ 8] = 1;         // Character#1             1

        data_0000_0[23] = 0;         // Character#1             0
        data_0000_0[22] = 0;         // Character#1(MSB)        0
        data_0000_0[21] = 0;         // Character#2(LSB)        1
        data_0000_0[20] = 0;         // Character#2             0
        data_0000_0[19] = 1;         // Character#2             0
        data_0000_0[18] = 1;         // Character#2             0
        data_0000_0[17] = 0;         // Character#2             0
        data_0000_0[16] = 0;         // Character#2(MSB)        0

        data_0000_0[31] = 1;         // Character#3(LSB)        1
        data_0000_0[30] = 1;         // Character#3             0
        data_0000_0[29] = 0;         // Character#3             1
        data_0000_0[28] = 0;         // Character#3             1
        data_0000_0[27] = 0;         // Character#3             0
        data_0000_0[26] = 0;         // Character#3(MSB)        0
        data_0000_0[25] = 0;         // Data Type Continuation Bit              0
        data_0000_0[24] = 0;
        // data_0000_0[31] = odd_num;  //                               (odd)
        check_odd (data_0000_0, data_0000_0[24]);

        uint Data_0000_0_tempt=0b0;
        PACKET(data_0000_0, Data_0000_0_tempt);
        TX_struct.TYPE_0000_0 = htonl(Data_0000_0_tempt);

///// TX_struct.TYPE_0000_1 ///// ATTACHMENT 20H-2
        int data_0000_1[32];
        //0b11110110 010010000001000000000011
    /// BIT             = Value      Function                               Coding
        data_0000_1[ 0] = 1;         // Label 1st Digit(MSB)                3:  1
        data_0000_1[ 1] = 1;         // Label 1st Digit(LSB)                    1
        data_0000_1[ 2] = 1;         // Label 2nd Digit(MSB)                6:  1
        data_0000_1[ 3] = 1;         // Label 2nd Digit                         1
        data_0000_1[ 4] = 0;         // Label 2nd Digit(LSB)                    0
        data_0000_1[ 5] = 1;         // Label 3rd Digit(MSB)                6:  1
        data_0000_1[ 6] = 1;         // Label 3rd Digit                         1
        data_0000_1[ 7] = 0;         // Label 3rd Digit(LSB)                    0

        data_0000_1[15] = 0;         // Character#4(LSB)
        data_0000_1[14] = 1;         // Character#4
        data_0000_1[13] = 0;         // Character#4
        data_0000_1[12] = 0;         // Character#4
        data_0000_1[11] = 1;         // Character#4
        data_0000_1[10] = 1;         // Character#4(MSB)
        data_0000_1[ 9] = 1;         // Character#5(LSB)
        data_0000_1[ 8] = 1;         // Character#5

        data_0000_1[23] = 0;         // Character#5
        data_0000_1[22] = 0;         // Character#5
        data_0000_1[21] = 1;         // Character#5
        data_0000_1[20] = 1;         // Character#5(MSB)
        data_0000_1[19] = 1;         // Character#6(LSB)
        data_0000_1[18] = 1;         // Character#6
        data_0000_1[17] = 1;         // Character#6
        data_0000_1[16] = 0;         // Character#6

        data_0000_1[31] = 1;         // Character#6
        data_0000_1[30] = 1;         // Character#6(MSB)
        data_0000_1[29] = 0;         // Pad                                     0
        data_0000_1[28] = 0;         // Pad                                     0
        data_0000_1[27] = 0;         // Pad                                     0
        data_0000_1[26] = 0;         // Pad                                     0
        data_0000_1[25] = 0;         // Data Type Continuation Bit              0
        data_0000_1[24] = 0;         // Parity                                  (odd)
        check_odd (data_0000_1, data_0000_1[24]);

        uint Data_0000_1_tempt=0b0;
        PACKET(data_0000_1, Data_0000_1_tempt);
        TX_struct.TYPE_0000_1 = htonl(Data_0000_1_tempt);

///// TX_struct.TYPE_0000_2 ///// ATTACHMENT 20H-3
        int data_0000_2[32];
        //0b11110110001100110001100001000110
    /// BIT             = Value      Function                               Coding
        data_0000_2[ 0] = 1;         // Label 1st Digit(MSB)                3:  1
        data_0000_2[ 1] = 1;         // Label 1st Digit(LSB)                    1
        data_0000_2[ 2] = 1;         // Label 2nd Digit(MSB)                6:  1
        data_0000_2[ 3] = 1;         // Label 2nd Digit                         1
        data_0000_2[ 4] = 0;         // Label 2nd Digit(LSB)                    0
        data_0000_2[ 5] = 1;         // Label 3rd Digit(MSB)                6:  1
        data_0000_2[ 6] = 1;         // Label 3rd Digit                         1
        data_0000_2[ 7] = 0;         // Label 3rd Digit(LSB)                    0

        data_0000_2[15] = 0;         // Character#7(LSB)        0
        data_0000_2[14] = 0;         // Character#7             1
        data_0000_2[13] = 0;         // Character#7             0
        data_0000_2[12] = 0;         // Character#7             0
        data_0000_2[11] = 0;         // Character#7             1
        data_0000_2[10] = 0;         // Character#7(MSB)        1
        data_0000_2[ 9] = 0;         // Character#8(LSB)        0
        data_0000_2[ 8] = 0;         // Character#8             0
        
        data_0000_2[23] = 0;         // Character#8             0
        data_0000_2[22] = 0;         // Character#8             0
        data_0000_2[21] = 0;         // Character#8             0
        data_0000_2[20] = 0;         // Character#8(MSB)        1
        data_0000_2[19] = 1;         // Type Code(LSB)
        data_0000_2[18] = 0;         // Type Code
        data_0000_2[17] = 0;         // Type Code(MSB)
        data_0000_2[16] = 0;         // Pad                                     0    

        data_0000_2[31] = 0;         // Aircraft Category(LSB)
        data_0000_2[30] = 1;         // Aircraft Category
        data_0000_2[29] = 0;         // Aircraft Category(MSB)                       
        data_0000_2[28] = 0;         // Pad                                     0
        data_0000_2[27] = 0;         // Pad                                     0
        data_0000_2[26] = 1;         // Pad                                     0
        data_0000_2[25] = 1;         // Data Type Continuation Bit              1
        data_0000_2[24] = 0;         // Parity                                  (odd)
        check_odd (data_0000_2, data_0000_2[24]);

        uint Data_0000_2_tempt=0b0;
        PACKET(data_0000_2, Data_0000_2_tempt);
        TX_struct.TYPE_0000_2 = htonl(Data_0000_2_tempt);

///// TX_struct.TYPE_0001 ///// ATTACHMENT 20I
        int data_0001[32];
    /// BIT             = Value      Function                               Coding
        data_0001[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        data_0001[ 1] = 1;         // Label 1st Digit(LSB)                  1
        data_0001[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        data_0001[ 3] = 1;         // Label 2nd Digit                       1
        data_0001[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        data_0001[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        data_0001[ 6] = 1;         // Label 3rd Digit                       1
        data_0001[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        data_0001[15] = 1;         // Data Type(LSB)                        1
        data_0001[14] = 0;         // Data Type                             0    
        data_0001[13] = 0;         // Data Type                             0
        data_0001[12] = 0;         // Data Type(MSB)                        0
        data_0001[11] = 0;         // Traffic Vert Sense            
        data_0001[10] = 0;         // Traffic Vert Sense
        data_0001[ 9] = 0;         // Coarse/Fine Range
        data_0001[ 8] = 0;         // Traffic Range 1/16 NM / 1/512 NM

        data_0001[23] = 0;         // Traffic Range 1/8  NM / 1/256 NM
        data_0001[22] = 0;         // Traffic Range 1/4  NM / 1/128 NM
        data_0001[21] = 0;         // Traffic Range 1/2  NM / 1/64  NM
        data_0001[20] = 0;         // Traffic Range 1    NM / 1/32  NM
        data_0001[19] = 0;         // Traffic Range 2    NM / 1/16  NM
        data_0001[18] = 0;         // Traffic Range 4    NM / 1/8   NM
        data_0001[17] = 0;         // Traffic Range 8    NM / 1/4   NM
        data_0001[16] = 0;         // Traffic Range 16   NM / 1/2   NM 

        data_0001[31] = 0;         // Traffic Range 32   NM / 1     NM
        data_0001[30] = 0;         // Traffic Range 64   NM / 2     NM
        data_0001[29] = 0;         // Traffic Range 128  NM / 4     NM             
        data_0001[28] = 0;         // Traffic Range 256  NM / 8     NM        
        data_0001[27] = 0;         // Traffic Range Invalidity    
        data_0001[26] = 1;         // Relative Alititude status
        data_0001[25] = 0;         // Data Type Continuation Bit              
        data_0001[24] = 0;         // Parity                                  (odd)
        check_odd (data_0001, data_0001[24]);

        uint Data_0001_tempt=0b0;
        PACKET(data_0001, Data_0001_tempt);
        TX_struct.TYPE_0001_0 = htonl(Data_0001_tempt);

///// TX_struct.TYPE_0001_1 ///// ATTACHMENT 20I
        int data_0001_1[32];
    /// BIT             = Value      Function                               Coding
        data_0001_1[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        data_0001_1[ 1] = 1;         // Label 1st Digit(LSB)                  1
        data_0001_1[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        data_0001_1[ 3] = 1;         // Label 2nd Digit                       1
        data_0001_1[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        data_0001_1[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        data_0001_1[ 6] = 1;         // Label 3rd Digit                       1
        data_0001_1[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        data_0001_1[15] = 0;         // Relative Altitude 100FT                      
        data_0001_1[14] = 0;         // Relative Altitude 200                    
        data_0001_1[13] = 0;         // Relative Altitude 400                 
        data_0001_1[12] = 0;         // Relative Altitude 800             
        data_0001_1[11] = 0;         // Relative Altitude 1600         
        data_0001_1[10] = 0;         // Relative Altitude 3200
        data_0001_1[ 9] = 0;         // Relative Altitude 6400
        data_0001_1[ 8] = 0;         // Relative Altitude 12800

        data_0001_1[23] = 0;         // Relative Altitude 25600
        data_0001_1[22] = 0;         // Relative Altitude Sign
        data_0001_1[21] = 0;         // Traffic Bearing 0.175781 Deg
        data_0001_1[20] = 0;         // Traffic Bearing 0.351563
        data_0001_1[19] = 0;         // Traffic Bearing 0.703125
        data_0001_1[18] = 0;         // Traffic Bearing 1.40625
        data_0001_1[17] = 0;         // Traffic Bearing 2.8125
        data_0001_1[16] = 0;         // Traffic Bearing 5.625

        data_0001_1[31] = 0;         // Traffic Bearing 11.25
        data_0001_1[30] = 0;         // Traffic Bearing 22.5
        data_0001_1[29] = 0;         // Traffic Bearing 45              
        data_0001_1[28] = 0;         // Traffic Bearing 90        
        data_0001_1[27] = 0;         // Traffic Bearing Sign 
        data_0001_1[26] = 0;         // Bearing status
        data_0001_1[25] = 1;         // Data Type Continuation Bit              
        data_0001_1[24] = 0;         // Parity                                  (odd)

        double own_alt = 0;
#ifdef GPS_POS
        own_alt = (double)bag_own_alt *m2ft;
#endif

#ifdef ADSB_POS
        own_alt = OWNSHIP_DATA.altitude * m2ft;
#endif

         if(own_alt > 25600)     { own_alt = own_alt - 25600;       data_0001_1[23] = 1;}
         if(own_alt > 12800)     { own_alt = own_alt - 12800;       data_0001_1[ 8] = 1;}
         if(own_alt > 6400)      { own_alt = own_alt - 6400;        data_0001_1[ 9] = 1;}
         if(own_alt > 3200)      { own_alt = own_alt - 3200;        data_0001_1[10] = 1;}
         if(own_alt > 1600)      { own_alt = own_alt - 1600;        data_0001_1[11] = 1;}
         if(own_alt > 800)       { own_alt = own_alt - 800;         data_0001_1[12] = 1;}
         if(own_alt > 400)       { own_alt = own_alt - 400;         data_0001_1[13] = 1;}
         if(own_alt > 200)       { own_alt = own_alt - 200;         data_0001_1[14] = 1;}
         if(own_alt > 100)       { own_alt = own_alt - 100;         data_0001_1[15] = 1;}

        check_odd (data_0001_1, data_0001_1[24]);

        uint Data_0001_1_tempt=0b0;
        PACKET(data_0001_1, Data_0001_1_tempt);
        TX_struct.TYPE_0001_1 = htonl(Data_0001_1_tempt);

///// TX_struct.TYPE_0002 ///// ATTACHMENT 20J
        int data_0002[32];
        // 0b 11110110 00110010 00110001 10000011
    /// BIT             = Value      Function                               Coding
        data_0002[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        data_0002[ 1] = 1;         // Label 1st Digit(LSB)                  1
        data_0002[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        data_0002[ 3] = 1;         // Label 2nd Digit                       1
        data_0002[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        data_0002[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        data_0002[ 6] = 1;         // Label 3rd Digit                       1
        data_0002[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        data_0002[15] = 0;         // Data Type(LSB)                        0
        data_0002[14] = 1;         // Data Type                             1    
        data_0002[13] = 0;         // Data Type                             0
        data_0002[12] = 0;         // Data Type(MSB)                        0

        data_0002[11] = 0;         // Latitude - 1.0728836 x 10^(-5) Deg 
        data_0002[10] = 0;         // Latitude - 2.1457672 x 10^(-5) 
        data_0002[ 9] = 0;         // Latitude - 4.2915344 x 10^(-5) 
        data_0002[ 8] = 0;         // Latitude - 8.5830688 x 10^(-5) 

        data_0002[23] = 0;         // Latitude - 1.7166118 x 10^(-4) 
        data_0002[22] = 0;         // Latitude - 3.4332275 x 10^(-4) 
        data_0002[21] = 0;         // Latitude - 6.8664551 x 10^(-4) 
        data_0002[20] = 0;         // Latitude - 0.00137329
        data_0002[19] = 0;         // Latitude - 0.00274658
        data_0002[18] = 0;         // Latitude - 0.00549316
        data_0002[17] = 0;         // Latitude - 0.0109863
        data_0002[16] = 0;         // Latitude - 0.0219726     

        data_0002[31] = 0;         // Latitude - 0.0439453
        data_0002[30] = 0;         // Latitude - 0.0878906
        data_0002[29] = 0;         // Latitude - 0.175781             
        data_0002[28] = 0;         // Latitude - 0.351563        
        data_0002[27] = 0;         // Latitude - 0.703125    
        data_0002[26] = 0;         // Latitude - 1.40625
        data_0002[25] = 0;         // Data Type Continuation Bit              
        data_0002[24] = 0;         // Parity                                  (odd)


///// TX_struct.TYPE_0002_1 ///// ATTACHMENT 20J
        int data_0002_1[32];
        // 0b 11110110 01010001 00001001 00100001

    /// BIT             = Value      Function                               Coding
        data_0002_1[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        data_0002_1[ 1] = 1;         // Label 1st Digit(LSB)                  1
        data_0002_1[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        data_0002_1[ 3] = 1;         // Label 2nd Digit                       1
        data_0002_1[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        data_0002_1[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        data_0002_1[ 6] = 1;         // Label 3rd Digit                       1
        data_0002_1[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        data_0002_1[15] = 0;         // Latitude - 2.8125
        data_0002_1[14] = 0;         // Latitude - 5.625                          
        data_0002_1[13] = 0;         // Latitude - 11.25
        data_0002_1[12] = 0;         // Latitude - 22.5                        
        data_0002_1[11] = 0;         // Latitude - 45 
        data_0002_1[10] = 0;         // Latitude - Sign
        data_0002_1[ 9] = 0;         // Longitude - 1.0728836 x 10^(-5) Deg
        data_0002_1[ 8] = 0;         // Longitude - 2.1457672 x 10^(-5) Deg

        data_0002_1[23] = 0;         // Longitude - 4.2915344 x 10^(-5) Deg
        data_0002_1[22] = 0;         // Longitude - 8.5830688 x 10^(-5) Deg
        data_0002_1[21] = 0;         // Longitude - 1.7166118 x 10^(-4) Deg
        data_0002_1[20] = 0;         // Longitude - 3.4332275 x 10^(-4) Deg
        data_0002_1[19] = 0;         // Longitude - 6.8664551 x 10^(-4) Deg
        data_0002_1[18] = 0;         // Longitude - 0.00137329
        data_0002_1[17] = 0;         // Longitude - 0.00274658
        data_0002_1[16] = 0;         // Longitude - 0.00549316

        data_0002_1[31] = 0;         // Longitude - 0.0109863
        data_0002_1[30] = 0;         // Longitude - 0.0219726                                        
        data_0002_1[29] = 0;         // Longitude - 0.0439453
        data_0002_1[28] = 0;         // Longitude - 0.0878906
        data_0002_1[27] = 0;         // Longitude - 0.175781             
        data_0002_1[26] = 0;         // Longitude - 0.351563 
        data_0002_1[25] = 0;         // Data Type Continuation Bit 
        data_0002_1[24] = 0;         // Parity                                  (odd)





///// TX_struct.TYPE_0002_2 ///// ATTACHMENT 20J
        int data_0002_2[32];
        //0b11110110 10001010 00000101 01100000
    /// BIT             = Value      Function                               Coding
        data_0002_2[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        data_0002_2[ 1] = 1;         // Label 1st Digit(LSB)                  1
        data_0002_2[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        data_0002_2[ 3] = 1;         // Label 2nd Digit                       1
        data_0002_2[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        data_0002_2[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        data_0002_2[ 6] = 1;         // Label 3rd Digit                       1
        data_0002_2[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        data_0002_2[15] = 0;         // Longitude - 0.703125    
        data_0002_2[14] = 0;         // Longitude - 1.40625                         
        data_0002_2[13] = 0;         // Longitude - 2.8125
        data_0002_2[12] = 0;         // Longitude - 5.625                          
        data_0002_2[11] = 0;         // Longitude - 11.25
        data_0002_2[10] = 0;         // Longitude - 22.5                        
        data_0002_2[ 9] = 0;         // Longitude - 45 
        data_0002_2[ 8] = 0;         // Longitude - 90

        data_0002_2[23] = 0;         // Longitude - Sign
        data_0002_2[22] = 0;         // Vertical Speed - 16 FT/MIN
        data_0002_2[21] = 0;         // Vertical Speed - 32 FT/MIN
        data_0002_2[20] = 0;         // Vertical Speed - 64 FT/MIN
        data_0002_2[19] = 0;         // Vertical Speed - 128 FT/MIN
        data_0002_2[18] = 0;         // Vertical Speed - 256 FT/MIN
        data_0002_2[17] = 0;         // Vertical Speed - 512 FT/MIN
        data_0002_2[16] = 0;         // Vertical Speed - 1024 FT/MIN

        data_0002_2[31] = 0;         // Vertical Speed - 2048 FT/MIN
        data_0002_2[30] = 0;         // Vertical Speed - 4096 FT/MIN                                       
        data_0002_2[29] = 0;         // Vertical Speed - 8192 FT/MIN
        data_0002_2[28] = 0;         // Vertical Speed - 16384 FT/MIN
        data_0002_2[27] = 0;         // Vertical Speed - sign    
        data_0002_2[26] = 1;         // Veltical Speed Status
        data_0002_2[25] = 1;         // Data Type Continuation Bit              1  
        data_0002_2[24] = 0;         // Parity                                  (odd)
        

#ifdef GPS_POS
        double own_lat = (double)bag_own_lat;
        double own_lon = (double)bag_own_lon;
        // double own_alt = (double)bag_own_alt;
#endif

#ifdef ADSB_POS
        double own_lat = (double)OWNSHIP_DATA.lat/10000000;
        double own_lon = (double)OWNSHIP_DATA.lon/10000000;
#endif
        double own_ver_vel = (float)OWNSHIP_DATA.ver_velocity/100.0 * mps2ftpmin;
        // cout << OWNSHIP_DATA.lat<< endl;
        
        // printf("own_lat : %f\n", own_lat);
        // LATITUDE
        if(own_lat > 45.0)              { own_lat = own_lat - 45.0;         data_0002_1[11] = 1;}
        if(own_lat > 22.5)              { own_lat = own_lat - 22.5;         data_0002_1[12] = 1;}
        if(own_lat > 11.25)             { own_lat = own_lat - 11.25;        data_0002_1[13] = 1;}
        if(own_lat > 5.625)             { own_lat = own_lat - 5.625;        data_0002_1[14] = 1;}
        if(own_lat>2.8125)              { own_lat = own_lat - 2.8125;       data_0002_1[15] = 1;}
        if(own_lat>1.40625)             { own_lat = own_lat - 1.40625;      data_0002[26] = 1;  }
        if(own_lat>0.703125 )           { own_lat = own_lat - 0.703125;     data_0002[27] = 1;  }
        if(own_lat>0.351563)            { own_lat = own_lat - 0.351563;     data_0002[28] = 1;  }
        if(own_lat>0.175781)            { own_lat = own_lat - 0.175781;     data_0002[29] = 1;  }
        if(own_lat>0.0878906)           { own_lat = own_lat - 0.0878906;    data_0002[30] = 1;  }
        if(own_lat>0.0439453)           { own_lat = own_lat - 0.0439453;    data_0002[31] = 1;  }
        if(own_lat>0.0219726)           { own_lat = own_lat - 0.0219726;    data_0002[16] = 1;  }
        if(own_lat>0.0109863)           { own_lat = own_lat - 0.0109863;    data_0002[17] = 1;  }
        if(own_lat>0.00549316)          { own_lat = own_lat - 0.00549316;   data_0002[18] = 1;  }
        if(own_lat>0.00274658)          { own_lat = own_lat - 0.00274658;   data_0002[19] = 1;  }
        if(own_lat>0.00137329)          { own_lat = own_lat - 0.00137329;   data_0002[20] = 1;  }
        if(own_lat> 6.8664551 / 10000)  { own_lat = own_lat - (6.8664551/10000);    data_0002[21] = 1;  }
        if(own_lat> 3.4332275/10000)    { own_lat = own_lat - (3.4332275/10000);    data_0002[22] = 1;  }
        if(own_lat>1.7166118/10000)     { own_lat = own_lat - (1.7166118/10000);    data_0002[23] = 1;  }
        if(own_lat>8.5830688/100000)    { own_lat = own_lat - (8.5830688/100000);   data_0002[8] = 1;   }
        if(own_lat>4.2915344/100000)    { own_lat = own_lat - (4.2915344/100000);   data_0002[9] = 1;   }
        if(own_lat>2.1457672/100000)    { own_lat = own_lat - (2.1457672/100000);   data_0002[10] = 1;  }
        if(own_lat>1.0728836/100000)    { own_lat = own_lat - (1.0728836/100000);   data_0002[11] = 1;  }


        /////Longitude
        
        // printf("own_lat : %f\n", own_lon);
        // LATITUDE
        if(own_lon > 90.0)              { own_lon = own_lon - 90.0;         data_0002_2[ 8] = 1;    }
        if(own_lon > 45.0)              { own_lon = own_lon - 45.0;         data_0002_2[ 9] = 1;    }
        if(own_lon > 22.5)              { own_lon = own_lon - 22.5;         data_0002_2[10] = 1;    }
        if(own_lon > 11.25)             { own_lon = own_lon - 11.25;        data_0002_2[11] = 1;    }
        if(own_lon > 5.625)             { own_lon = own_lon - 5.625;        data_0002_2[12] = 1;    }
        if(own_lon>2.8125)              { own_lon = own_lon - 2.8125;       data_0002_2[13] = 1;    }
        if(own_lon>1.40625)             { own_lon = own_lon - 1.40625;      data_0002_2[14] = 1;    }
        if(own_lon>0.703125)            { own_lon = own_lon - 0.703125;     data_0002_2[15] = 1;    }
        if(own_lon>0.351563)            { own_lon = own_lon - 0.351563;     data_0002_1[26] = 1;    }
        if(own_lon>0.175781)            { own_lon = own_lon - 0.175781;     data_0002_1[27] = 1;    }
        if(own_lon>0.0878906)           { own_lon = own_lon - 0.0878906;    data_0002_1[28] = 1;    }
        if(own_lon>0.0439453)           { own_lon = own_lon - 0.0439453;    data_0002_1[29] = 1;    }
        if(own_lon>0.0219726)           { own_lon = own_lon - 0.0219726;    data_0002_1[30] = 1;    }
        if(own_lon>0.0109863)           { own_lon = own_lon - 0.0109863;    data_0002_1[31] = 1;    }
        if(own_lon>0.00549316)          { own_lon = own_lon - 0.00549316;   data_0002_1[16] = 1;    }
        if(own_lon>0.00274658)          { own_lon = own_lon - 0.00274658;   data_0002_1[17] = 1;    }
        if(own_lon>0.00137329)          { own_lon = own_lon - 0.00137329;   data_0002_1[18] = 1;    }
        if(own_lon> 6.8664551/10000)    { own_lon = own_lon - (6.8664551/10000);    data_0002_1[19] = 1;    }
        if(own_lon> 3.4332275/10000)    { own_lon = own_lon - (3.4332275/10000);    data_0002_1[20] = 1;    }
        if(own_lon>1.7166118/10000)     { own_lon = own_lon - (1.7166118/10000);    data_0002_1[21] = 1;    }
        if(own_lon>8.5830688/100000)    { own_lon = own_lon - (8.5830688/100000);   data_0002_1[22] = 1;    }
        if(own_lon>4.2915344/100000)    { own_lon = own_lon - (4.2915344/100000);   data_0002_1[23] = 1;    }
        if(own_lon>2.1457672/100000)    { own_lon = own_lon - (2.1457672/100000);   data_0002_1[ 8] = 1;    }
        if(own_lon>1.0728836/100000)    { own_lon = own_lon - (1.0728836/100000);   data_0002_1[ 9] = 1;    }
        
        // Vertical Velocity
        if(own_ver_vel > 16384)     { own_ver_vel = own_ver_vel - 16384;       data_0002_2[28] = 1;}
        if(own_ver_vel > 8192)      { own_ver_vel = own_ver_vel - 8192;        data_0002_2[29] = 1;}
        if(own_ver_vel > 4096)      { own_ver_vel = own_ver_vel - 4096;        data_0002_2[30] = 1;}
        if(own_ver_vel > 2048)      { own_ver_vel = own_ver_vel - 2048;        data_0002_2[31] = 1;}
        if(own_ver_vel > 1024)      { own_ver_vel = own_ver_vel - 1024;        data_0002_2[16] = 1;}
        if(own_ver_vel > 512)       { own_ver_vel = own_ver_vel - 512;         data_0002_2[17] = 1;}
        if(own_ver_vel > 256)       { own_ver_vel = own_ver_vel - 256;         data_0002_2[18] = 1;}
        if(own_ver_vel > 128)       { own_ver_vel = own_ver_vel - 128;         data_0002_2[19] = 1;}
        if(own_ver_vel > 64)        { own_ver_vel = own_ver_vel - 64;          data_0002_2[20] = 1;}
        if(own_ver_vel > 32)        { own_ver_vel = own_ver_vel - 32;          data_0002_2[21] = 1;}
        if(own_ver_vel > 16)        { own_ver_vel = own_ver_vel - 16;          data_0002_2[22] = 1;}
 

        check_odd (data_0002, data_0002[24]);
        uint Data_0002_tempt=0b0;
        PACKET(data_0002, Data_0002_tempt);
        TX_struct.TYPE_0002_0 = htonl(Data_0002_tempt);

        check_odd (data_0002_1, data_0002_1[24]);
        uint Data_0002_1_tempt=0b0;
        PACKET(data_0002_1, Data_0002_1_tempt);
        TX_struct.TYPE_0002_1 = htonl(Data_0002_1_tempt);

        check_odd (data_0002_2, data_0002_2[24]);
        uint Data_0002_2_tempt=0b0;
        PACKET(data_0002_2, Data_0002_2_tempt);
        TX_struct.TYPE_0002_2 = htonl(Data_0002_2_tempt);


/// TX_struct.TYPE_0003 ///// ATTACHMENT 20L  -----------------------------------From here, modify 
        int data_0003[32];
    /// BIT             = Value      Function                               Coding
        data_0003[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        data_0003[ 1] = 1;         // Label 1st Digit(LSB)                  1
        data_0003[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        data_0003[ 3] = 1;         // Label 2nd Digit                       1
        data_0003[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        data_0003[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        data_0003[ 6] = 1;         // Label 3rd Digit                       1
        data_0003[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        data_0003[15] = 1;         // Data Type (LSB)
        data_0003[14] = 1;         // Data Type
        data_0003[13] = 0;         // Data Type
        data_0003[12] = 0;         // Data Type (MSB)
        data_0003[11] = 0;         // Pad
        data_0003[10] = 0;         // Pad 
        data_0003[ 9] = 0;         // Magnetic(1)/True Heading(0) Indication
        data_0003[ 8] = 0;         // Ground Speed   1 knot

        data_0003[23] = 0;         // Ground Speed   2
        data_0003[22] = 0;         // Ground Speed   4
        data_0003[21] = 0;         // Ground Speed   8
        data_0003[20] = 0;         // Ground Speed   16
        data_0003[19] = 0;         // Ground Speed   32
        data_0003[18] = 0;         // Ground Speed   64
        data_0003[17] = 0;         // Ground Speed   128
        data_0003[16] = 0;         // Ground Speed   256

        data_0003[31] = 0;         // Ground Speed   512                                        
        data_0003[30] = 0;         // Ground Speed   1024
        data_0003[29] = 0;         // Ground Speed   2048
        data_0003[28] = 0;         // Pad             
        data_0003[27] = 1;         // Ground Speed Status        
        data_0003[26] = 1;         // Track Angle Status    
        data_0003[25] = 0;         // Data Type Continuation Bit              
        data_0003[24] = 0;         // Parity                                  (odd)

        float own_hor_vel = (float)OWNSHIP_DATA.hor_velocity /100.0 * mps2knot; //[knot]
        // cout << "own_hor_vel : " << own_hor_vel << " knot , " << (float)OWNSHIP_DATA.hor_velocity /100.0 << " m/s"<<endl;
        if(own_hor_vel > 2048)     { own_hor_vel = own_hor_vel - 2048;       data_0003[29] = 1;}
        if(own_hor_vel > 1024)     { own_hor_vel = own_hor_vel - 1024;       data_0003[30] = 1;}
        if(own_hor_vel > 512)      { own_hor_vel = own_hor_vel - 512;        data_0003[31] = 1;}
        if(own_hor_vel > 256)      { own_hor_vel = own_hor_vel - 256;        data_0003[16] = 1;}
        if(own_hor_vel > 128)      { own_hor_vel = own_hor_vel - 128;        data_0003[17] = 1;}
        if(own_hor_vel > 64)       { own_hor_vel = own_hor_vel - 64;         data_0003[18] = 1;}
        if(own_hor_vel > 32)       { own_hor_vel = own_hor_vel - 32;         data_0003[19] = 1;}
        if(own_hor_vel > 16)       { own_hor_vel = own_hor_vel - 16;         data_0003[20] = 1;}
        if(own_hor_vel > 8)        { own_hor_vel = own_hor_vel - 8;          data_0003[21] = 1;}
        if(own_hor_vel > 4)        { own_hor_vel = own_hor_vel - 4;          data_0003[22] = 1;}
        if(own_hor_vel > 2)        { own_hor_vel = own_hor_vel - 2;          data_0003[23] = 1;}
        if(own_hor_vel > 1)        { own_hor_vel = own_hor_vel - 1;          data_0003[ 8] = 1;}

        check_odd (data_0003, data_0003[24]);

        uint Data_0003_tempt=0b0;
        PACKET(data_0003, Data_0003_tempt);
        TX_struct.TYPE_0003_0 = htonl(Data_0003_tempt);

///// TX_struct.TYPE_0003_1 ///// ATTACHMENT 20L
        int data_0003_1[32];
    /// BIT             = Value      Function                               Coding
        data_0003_1[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        data_0003_1[ 1] = 1;         // Label 1st Digit(LSB)                  1
        data_0003_1[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        data_0003_1[ 3] = 1;         // Label 2nd Digit                       1
        data_0003_1[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        data_0003_1[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        data_0003_1[ 6] = 1;         // Label 3rd Digit                       1
        data_0003_1[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        data_0003_1[15] = 0;         // Heading/Track Angle 1.40625 deg.
        data_0003_1[14] = 0;         // Heading/Track Angle 2.8125                      
        data_0003_1[13] = 0;         // Heading/Track Angle 5.625
        data_0003_1[12] = 0;         // Heading/Track Angle 11.25              
        data_0003_1[11] = 0;         // Heading/Track Angle 22.5
        data_0003_1[10] = 0;         // Heading/Track Angle 45.0
        data_0003_1[ 9] = 0;         // Heading/Track Angle 90.0
        data_0003_1[ 8] = 0;         // Heading/Track Angle Sgin(0:positive, 1: negative)

        data_0003_1[23] = 0;         // Closure Rate 0.5 knot
        data_0003_1[22] = 0;         // Closure Rate 1
        data_0003_1[21] = 0;         // Closure Rate 2
        data_0003_1[20] = 0;         // Closure Rate 4
        data_0003_1[19] = 0;         // Closure Rate 8
        data_0003_1[18] = 0;         // Closure Rate 16
        data_0003_1[17] = 0;         // Closure Rate 32
        data_0003_1[16] = 0;         // Closure Rate 64

        data_0003_1[31] = 0;         // Closure Rate 128
        data_0003_1[30] = 0;         // Closure Rate 256                                        
        data_0003_1[29] = 0;         // Closure Rate 512
        data_0003_1[28] = 0;         // Closure Rate 1024
        data_0003_1[27] = 0;         // Closure Rate Sign         
        data_0003_1[26] = 1;         // Closure Rate Status 
        data_0003_1[25] = 1;         // Data Type Continuation Bit              
        data_0003_1[24] = 0;         // Parity                                  (odd)


        // OWNSHIP_DATA.heading
        float own_heading = (float)OWNSHIP_DATA.heading / 100.0 ; 

        if (own_heading > 180.0){
            own_heading = own_heading - 180.0;
            data_0003_1[8] = 1; //1; 
            if (own_heading>90.0)   { own_heading = own_heading - 90.0;     data_0003_1[ 9] = 1; }
            if (own_heading>45.0)   { own_heading = own_heading - 45.0;     data_0003_1[10] = 1; }
            if (own_heading>22.5)   { own_heading = own_heading - 22.5;     data_0003_1[11] = 1; }
            if (own_heading>11.25)  { own_heading = own_heading - 11.25;    data_0003_1[12] = 1; }
            if (own_heading>5.625)  { own_heading = own_heading - 5.625;    data_0003_1[13] = 1; }
            if (own_heading>2.8125) { own_heading = own_heading - 2.8125;   data_0003_1[14] = 1; }
            if (own_heading>1.40625){ own_heading = own_heading - 1.40625;  data_0003_1[15] = 1; }
        }
        else{
            data_0003_1[8] = 0; 
            if (own_heading>90.0)   { own_heading = own_heading - 90.0;     data_0003_1[ 9] = 1; }
            if (own_heading>45.0)   { own_heading = own_heading - 45.0;     data_0003_1[10] = 1; }
            if (own_heading>22.5)   { own_heading = own_heading - 22.5;     data_0003_1[11] = 1; }
            if (own_heading>11.25)  { own_heading = own_heading - 11.25;    data_0003_1[12] = 1; }
            if (own_heading>5.625)  { own_heading = own_heading - 5.625;    data_0003_1[13] = 1; }
            if (own_heading>2.8125) { own_heading = own_heading - 2.8125;   data_0003_1[14] = 1; }
            if (own_heading>1.40625){ own_heading = own_heading - 1.40625;  data_0003_1[15] = 1; }
        }

        check_odd (data_0003_1, data_0003_1[24]);

        uint Data_0003_1_tempt=0b0;
        PACKET(data_0003_1, Data_0003_1_tempt);
        TX_struct.TYPE_0003_1 = htonl(Data_0003_1_tempt);

/// TX_struct.TYPE_0004 ///// ATTACHMENT 20M  -----------------------------------From here, modify 
        int data_0004[32];
    /// BIT             = Value      Function                               Coding
        data_0004[ 0] = 1;         // Label 1st Digit(MSB)               3:  1
        data_0004[ 1] = 1;         // Label 1st Digit(LSB)                   1
        data_0004[ 2] = 1;         // Label 2nd Digit(MSB)               6:  1
        data_0004[ 3] = 1;         // Label 2nd Digit                        1
        data_0004[ 4] = 0;         // Label 2nd Digit(LSB)                   0
        data_0004[ 5] = 1;         // Label 3rd Digit(MSB)               6:  1
        data_0004[ 6] = 1;         // Label 3rd Digit                        1
        data_0004[ 7] = 0;         // Label 3rd Digit(LSB)                   0

        data_0004[15] = 0;         // Data Type (LSB)                    4:  0 
        data_0004[14] = 0;         // Data Type                              0
        data_0004[13] = 1;         // Data Type                              1
        data_0004[12] = 0;         // Data Type (MSB)                        0
        data_0004[11] = 0;         // AIRB Queality level LSB
        data_0004[10] = 0;         // AIRB Queality level MSB
        data_0004[ 9] = 0;         // SURF Quality Level LSB
        data_0004[ 8] = 0;         // SURF Quality Level MSB

        data_0004[23] = 0;         // Interval Management Quality Level LSB
        data_0004[22] = 0;         // Interval Management Quality Level MSB
        data_0004[21] = 0;         // ITP Quality Level LSB
        data_0004[20] = 0;         // ITP Quality Level MSB
        data_0004[19] = 0;         // TSAA Quality Level LSB
        data_0004[18] = 0;         // TSAA Quality Level MSB
        data_0004[17] = 0;         // Reserved for future application's quality level
        data_0004[16] = 0;         // Reserved for future application's quality level

        data_0004[31] = 0;         // VSA Quality Level LSB
        data_0004[30] = 0;         // VSA Quality Level MSB
        data_0004[29] = 0;         // Reserved for future application's quality level           
        data_0004[28] = 0;         // Reserved for future application's quality level
        data_0004[27] = 0;         // Reserved for future application's quality level
        data_0004[26] = 0;         // Reserved for future application's quality level    
        data_0004[25] = 0;         // Data Type Continuation Bit              
        data_0004[24] = 0;         // Parity                                  (odd)
        check_odd (data_0004, data_0004[24]);

        uint Data_0004_tempt=0b0;
        PACKET(data_0004, Data_0004_tempt);
        // TX_struct.TYPE_0004_0 = htonl(Data_0004_tempt);

/// TX_struct.TYPE_0004_1 ///// ATTACHMENT 20M  -----------------------------------From here, modify 
        int data_0004_1[32];
    /// BIT             = Value      Function                               Coding
        data_0004_1[ 0] = 1;         // Label 1st Digit(MSB)               3:  1
        data_0004_1[ 1] = 1;         // Label 1st Digit(LSB)                   1
        data_0004_1[ 2] = 1;         // Label 2nd Digit(MSB)               6:  1
        data_0004_1[ 3] = 1;         // Label 2nd Digit                        1
        data_0004_1[ 4] = 0;         // Label 2nd Digit(LSB)                   0
        data_0004_1[ 5] = 1;         // Label 3rd Digit(MSB)               6:  1
        data_0004_1[ 6] = 1;         // Label 3rd Digit                        1
        data_0004_1[ 7] = 0;         // Label 3rd Digit(LSB)                   0

        data_0004_1[15] = 0;         // Reserved for future application's quality level 
        data_0004_1[14] = 0;         // Reserved for future application's quality level 
        data_0004_1[13] = 0;         // Reserved for future application's quality level 
        data_0004_1[12] = 0;         // Reserved for future application's quality level
        data_0004_1[11] = 0;         // Reserved for future application's quality level 
        data_0004_1[10] = 0;         // Reserved for future application's quality level 
        data_0004_1[ 9] = 0;         // Reserved for future application's quality level 
        data_0004_1[ 8] = 0;         // Reserved for future application's quality level 

        data_0004_1[23] = 0;         // Reserved for future application's quality level 
        data_0004_1[22] = 0;         // Reserved for future application's quality level 
        data_0004_1[21] = 0;         // Reserved for future application's quality level 
        data_0004_1[20] = 0;         // Reserved for future application's quality level 
        data_0004_1[19] = 0;         // Reserved for future application's quality level 
        data_0004_1[18] = 0;         // Reserved for future application's quality level 
        data_0004_1[17] = 0;         // Reserved for future application's quality level
        data_0004_1[16] = 0;         // Reserved for future application's quality level

        data_0004_1[31] = 0;         // Reserved for future application's quality level
        data_0004_1[30] = 0;         // Reserved for future application's quality level
        data_0004_1[29] = 0;         // Reserved for future application's quality level           
        data_0004_1[28] = 0;         // Reserved for future application's quality level
        data_0004_1[27] = 0;         // Reserved for future application's quality level
        data_0004_1[26] = 0;         // Reserved for future application's quality level    
        data_0004_1[25] = 1;         // Data Type Continuation Bit              
        data_0004_1[24] = 0;         // Parity                                  (odd)
        check_odd (data_0004_1, data_0004_1[24]);

        uint Data_0004_1_tempt=0b0;
        PACKET(data_0004_1, Data_0004_1_tempt);
        // TX_struct.TYPE_0004_1 = htonl(Data_0004_1_tempt);


/// TX_struct.TYPE_0005 ///// ATTACHMENT 20N  -----------------------------------From here, modify 
        int data_0005[32];
    /// BIT           = Value      Function                               Coding
        data_0005[ 0] = 1;         // Label 1st Digit(MSB)               3:  1
        data_0005[ 1] = 1;         // Label 1st Digit(LSB)                   1
        data_0005[ 2] = 1;         // Label 2nd Digit(MSB)               6:  1
        data_0005[ 3] = 1;         // Label 2nd Digit                        1
        data_0005[ 4] = 0;         // Label 2nd Digit(LSB)                   0
        data_0005[ 5] = 1;         // Label 3rd Digit(MSB)               6:  1
        data_0005[ 6] = 1;         // Label 3rd Digit                        1
        data_0005[ 7] = 0;         // Label 3rd Digit(LSB)                   0

        data_0005[15] = 1;         // Data Type (LSB)                    5:  1 
        data_0005[14] = 0;         // Data Type                              0
        data_0005[13] = 1;         // Data Type                              1
        data_0005[12] = 0;         // Data Type (MSB)                        0
        data_0005[11] = 0;         // Position Offset Applied
        data_0005[10] = 1;         // Width Code
        data_0005[ 9] = 0;         // Length Code (LSB)
        data_0005[ 8] = 0;         // Length Code 

        data_0005[23] = 0;         // Length Code (MSB)
        data_0005[22] = 0;         // Length/Width Status
        data_0005[21] = 0;         // PAD
        data_0005[20] = 1;         // PAD
        data_0005[19] = 0;         // PAD
        data_0005[18] = 1;         // PAD
        data_0005[17] = 0;         // PAD
        data_0005[16] = 0;         // PAD

        data_0005[31] = 0;         // PAD
        data_0005[30] = 1;         // PAD
        data_0005[29] = 0;         // PAD           
        data_0005[28] = 1;         // PAD
        data_0005[27] = 0;         // PAD
        data_0005[26] = 0;         // PAD    
        data_0005[25] = 1;         // Data Type COntinuation Bit              
        data_0005[24] = 0;         // Parity                                  (odd)
        check_odd (data_0005, data_0005[24]);

        uint Data_0005_tempt=0b0;
        PACKET(data_0005, Data_0005_tempt);
        // TX_struct.TYPE_0005_0 = htonl(Data_0005_tempt);


/// TX_struct.TYPE_0007 ///// ATTACHMENT 20P  -----------------------------------From here, modify 
        int data_0007[32];
    /// BIT           = Value      Function                               Coding
        data_0007[ 0] = 1;         // Label 1st Digit(MSB)               3:  1
        data_0007[ 1] = 1;         // Label 1st Digit(LSB)                   1
        data_0007[ 2] = 1;         // Label 2nd Digit(MSB)               6:  1
        data_0007[ 3] = 1;         // Label 2nd Digit                        1
        data_0007[ 4] = 0;         // Label 2nd Digit(LSB)                   0
        data_0007[ 5] = 1;         // Label 3rd Digit(MSB)               6:  1
        data_0007[ 6] = 1;         // Label 3rd Digit                        1
        data_0007[ 7] = 0;         // Label 3rd Digit(LSB)                   0

        data_0007[15] = 1;         // Data Type (LSB)                    7:  1 
        data_0007[14] = 1;         // Data Type                              1
        data_0007[13] = 1;         // Data Type                              1
        data_0007[12] = 0;         // Data Type (MSB)                        0
        data_0007[11] = 0;         // Seconds 1 LSB
        data_0007[10] = 0;         // Seconds 2
        data_0007[ 9] = 0;         // Seconds 4
        data_0007[ 8] = 0;         // Seconds 8 

        data_0007[23] = 0;         // Seconds 16
        data_0007[22] = 0;         // Seconds 32 MSB
        data_0007[21] = 0;         // Minutes 1 LSB
        data_0007[20] = 0;         // Minutes 2
        data_0007[19] = 0;         // Minutes 4
        data_0007[18] = 0;         // Minutes 8
        data_0007[17] = 0;         // Minutes 16
        data_0007[16] = 0;         // Minutes 32 MSB

        data_0007[31] = 0;         // Hours 1 LSB
        data_0007[30] = 0;         // Hour  2
        data_0007[29] = 0;         // Hours 4        
        data_0007[28] = 0;         // Hours 8
        data_0007[27] = 0;         // Hours 16 MSB
        data_0007[26] = 0;         // PAD    
        data_0007[25] = 0;         // Data Type Continuation Bit              0    
        data_0007[24] = 0;         // Parity                                  (odd)

        int dis_min     = T_min;
        int dis_sec     = T_sec;
        int dis_hour    = T_hour;

        if(dis_sec > 32)     { dis_sec = dis_sec - 32;       data_0007[22] = 1;}
        if(dis_sec > 16)     { dis_sec = dis_sec - 16;       data_0007[23] = 1;}
        if(dis_sec > 8)      { dis_sec = dis_sec - 8;        data_0007[ 8] = 1;}
        if(dis_sec > 4)      { dis_sec = dis_sec - 4;        data_0007[ 9] = 1;}
        if(dis_sec > 2)      { dis_sec = dis_sec - 2;        data_0007[10] = 1;}
        if(dis_sec > 1)      { dis_sec = dis_sec - 1;        data_0007[11] = 1;}

        if(dis_min > 32)     { dis_min = dis_min - 32;       data_0007[16] = 1;}
        if(dis_min > 16)     { dis_min = dis_min - 16;       data_0007[17] = 1;}
        if(dis_min > 8)      { dis_min = dis_min - 8;        data_0007[18] = 1;}
        if(dis_min > 4)      { dis_min = dis_min - 4;        data_0007[19] = 1;}
        if(dis_min > 2)      { dis_min = dis_min - 2;        data_0007[20] = 1;}
        if(dis_min > 1)      { dis_min = dis_min - 1;        data_0007[21] = 1;}
 
        if(dis_hour > 16)    { dis_hour = dis_hour - 16;     data_0007[27] = 1;}
        if(dis_hour > 8)     { dis_hour = dis_hour - 8;      data_0007[28] = 1;}
        if(dis_hour > 4)     { dis_hour = dis_hour - 4;      data_0007[29] = 1;}
        if(dis_hour > 2)     { dis_hour = dis_hour - 2;      data_0007[30] = 1;}
        if(dis_hour > 1)     { dis_hour = dis_hour - 1;      data_0007[31] = 1;}
        
        // cout << "display time" << dis_sec << endl;
        check_odd (data_0007, data_0007[24]);

        uint Data_0007_tempt=0b0;
        PACKET(data_0007, Data_0007_tempt);
        // TX_struct.TYPE_0007_0 = htonl(Data_0007_tempt);

/// TX_struct.TYPE_0007_1 ///// ATTACHMENT 20P  -----------------------------------From here, modify 
        int data_0007_1[32];
    /// BIT             = Value      Function                               Coding
        data_0007_1[ 0] = 1;         // Label 1st Digit(MSB)               3:  1
        data_0007_1[ 1] = 1;         // Label 1st Digit(LSB)                   1
        data_0007_1[ 2] = 1;         // Label 2nd Digit(MSB)               6:  1
        data_0007_1[ 3] = 1;         // Label 2nd Digit                        1
        data_0007_1[ 4] = 0;         // Label 2nd Digit(LSB)                   0
        data_0007_1[ 5] = 1;         // Label 3rd Digit(MSB)               6:  1
        data_0007_1[ 6] = 1;         // Label 3rd Digit                        1
        data_0007_1[ 7] = 0;         // Label 3rd Digit(LSB)                   0

        data_0007_1[15] = 1;         // Pad
        data_0007_1[14] = 1;         // Pad
        data_0007_1[13] = 1;         // Pad
        data_0007_1[12] = 0;         // Pad
        data_0007_1[11] = 0;         // Pad
        data_0007_1[10] = 1;         // Pad
        data_0007_1[ 9] = 0;         // Pad
        data_0007_1[ 8] = 0;         // Seconds 0.00011207 LSB

        data_0007_1[23] = 0;         // Seconds 0.000244141
        data_0007_1[22] = 0;         // Seconds 0.000488281
        data_0007_1[21] = 0;         // Seconds 0.000976563
        data_0007_1[20] = 0;         // Seconds 0.001953125
        data_0007_1[19] = 0;         // Seconds 0.00390625
        data_0007_1[18] = 0;         // Seconds 0.0078125
        data_0007_1[17] = 0;         // Seconds 0.015625
        data_0007_1[16] = 0;         // Seconds 0.03125

        data_0007_1[31] = 0;         // Seconds 0.0625
        data_0007_1[30] = 0;         // Seconds 0.125
        data_0007_1[29] = 0;         // Seconds 0.25  
        data_0007_1[28] = 0;         // Seconds 0.5          MSB
        data_0007_1[27] = 0;         // Time of Applicability Invalidity
        data_0007_1[26] = 0;         // PAD    
        data_0007_1[25] = 1;         // Data Type Continuation Bit              1  
        data_0007_1[24] = 0;         // Parity                                  (odd)
        check_odd (data_0007_1, data_0007_1[24]);       // Parity                                  (odd)

        uint Data_0007_1_tempt=0b0;
        PACKET(data_0007_1, Data_0007_1_tempt);
        // TX_struct.TYPE_0007_1 = htonl(Data_0007_1_tempt);





/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
////INTRUDER INFORMATION/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef INTRUDER_ON
///// TX_struct.INTR_DTIF_Packet_Header ///// ATTACHMENT 20G
        //0b 1111 0110 1000 0001 0000 0111 1010 0000
        int INTR_DTIF_Packet_header[32];
    /// BIT                    = Value      Function                            Coding
        INTR_DTIF_Packet_header[ 0] = 1;         // Label 1st Digit(MSB)             3:  1
        INTR_DTIF_Packet_header[ 1] = 1;         // Label 1st Digit(LSB)                 1
        INTR_DTIF_Packet_header[ 2] = 1;         // Label 2nd Digit(MSB)             6:  1
        INTR_DTIF_Packet_header[ 3] = 1;         // Label 2nd Digit                      1
        INTR_DTIF_Packet_header[ 4] = 0;         // Label 2nd Digit(LSB)                 0
        INTR_DTIF_Packet_header[ 5] = 1;         // Label 3rd Digit(MSB)             6:  1
        INTR_DTIF_Packet_header[ 6] = 1;         // Label 3rd Digit                      1
        INTR_DTIF_Packet_header[ 7] = 0;         // Label 3rd Digit(LSB)                 0

        INTR_DTIF_Packet_header[15] = 0;         // Traffic Number-1
        INTR_DTIF_Packet_header[14] = 1;         // Traffic Number-2
        INTR_DTIF_Packet_header[13] = 0;         // Traffic Number-4
        INTR_DTIF_Packet_header[12] = 0;         // Traffic Number-8
        INTR_DTIF_Packet_header[11] = 0;         // Traffic Number-16
        INTR_DTIF_Packet_header[10] = 0;         // Traffic Number-32
        INTR_DTIF_Packet_header[ 9] = 0;         // Traffic Number-64
        INTR_DTIF_Packet_header[ 8] = 1;         // Num.Label in Packet-1

        INTR_DTIF_Packet_header[23] = 1;         // Num.Label in Packet-2
        INTR_DTIF_Packet_header[22] = 0;         // Num.Label in Packet-4
        INTR_DTIF_Packet_header[21] = 1;         // Num.Label in Packet-8
        INTR_DTIF_Packet_header[20] = 0;         // Num.Label in Packet-16
        INTR_DTIF_Packet_header[19] = 0;         // Reserved for Manufacturer Use
        INTR_DTIF_Packet_header[18] = 1;         // Reserved for Manufacturer Use         
        INTR_DTIF_Packet_header[17] = 0;         // Display Matrix
        INTR_DTIF_Packet_header[16] = 0;         // Display Matrix

        INTR_DTIF_Packet_header[31] = 0;         // Display Matrix
        INTR_DTIF_Packet_header[30] = 0;         // Display Matrix
        INTR_DTIF_Packet_header[29] = 1;         // Source Data Type
        INTR_DTIF_Packet_header[28] = 1;         // Source Data Type
        INTR_DTIF_Packet_header[27] = 1;         // Source Data Type
        INTR_DTIF_Packet_header[26] = 0;         // Air/Ground Status
        INTR_DTIF_Packet_header[25] = 0;         // Spare
        INTR_DTIF_Packet_header[24] = 0;         // Parity                               (odd)
        check_odd (INTR_DTIF_Packet_header, INTR_DTIF_Packet_header[24]);

        uint INTR_DTIF_packet_header_temp=0b0;
        PACKET(INTR_DTIF_Packet_header, INTR_DTIF_packet_header_temp);
        TX_struct.INTR_DTIF_Packet_Header = htonl(INTR_DTIF_packet_header_temp);

///// TX_struct.INTR_TYPE_0000_0 ///// ATTACHMENT 20H-1
        int INTR_data_0000_0[32];
        //0b 1111 0110 1111 0000 0110 1000 1000 0001
    /// BIT             = Value      Function                               Coding
        INTR_data_0000_0[ 0] = 1;         // Label 1st Digit(MSB)                3:  1
        INTR_data_0000_0[ 1] = 1;         // Label 1st Digit(LSB)                    1
        INTR_data_0000_0[ 2] = 1;         // Label 2nd Digit(MSB)                6:  1
        INTR_data_0000_0[ 3] = 1;         // Label 2nd Digit                         1
        INTR_data_0000_0[ 4] = 0;         // Label 2nd Digit(LSB)                    0
        INTR_data_0000_0[ 5] = 1;         // Label 3rd Digit(MSB)                6:  1
        INTR_data_0000_0[ 6] = 1;         // Label 3rd Digit                         1
        INTR_data_0000_0[ 7] = 0;         // Label 3rd Digit(LSB)                    0

        INTR_data_0000_0[15] = 0;         // Data Type(LSB)                      0:  0
        INTR_data_0000_0[14] = 0;         // Data Type                               0
        INTR_data_0000_0[13] = 0;         // Data Type                               0
        INTR_data_0000_0[12] = 0;         // Data Type(MSB)                          0
        INTR_data_0000_0[11] = 0;         // Character#1(LSB)       1
        INTR_data_0000_0[10] = 0;         // Character#1            0
        INTR_data_0000_0[ 9] = 0;         // Character#1            0
        INTR_data_0000_0[ 8] = 1;         // Character#1            1

        INTR_data_0000_0[23] = 0;         // Character#1            0
        INTR_data_0000_0[22] = 0;         // Character#1(MSB)       0
        INTR_data_0000_0[21] = 1;         // Character#2(LSB)       0
        INTR_data_0000_0[20] = 1;         // Character#2            1
        INTR_data_0000_0[19] = 0;         // Character#2            1
        INTR_data_0000_0[18] = 0;         // Character#2            1
        INTR_data_0000_0[17] = 1;         // Character#2            0
        INTR_data_0000_0[16] = 0;         // Character#2(MSB)       0

        INTR_data_0000_0[31] = 0;         // Character#3(LSB)       0
        INTR_data_0000_0[30] = 1;         // Character#3            0
        INTR_data_0000_0[29] = 1;         // Character#3            1
        INTR_data_0000_0[28] = 0;         // Character#3            0
        INTR_data_0000_0[27] = 0;         // Character#3            0
        INTR_data_0000_0[26] = 0;         // Character#3(MSB)       0
        INTR_data_0000_0[25] = 0;         // Data Type Continuation Bit              0
        INTR_data_0000_0[24] = 0;
        // data_0000_0[31] = odd_num;  //                               (odd)
        check_odd (INTR_data_0000_0, INTR_data_0000_0[24]);

        uint INTR_Data_0000_0_tempt=0b0;
        PACKET(INTR_data_0000_0, INTR_Data_0000_0_tempt);
        TX_struct.INTR_TYPE_0000_0 = htonl(INTR_Data_0000_0_tempt);

///// TX_struct.TYPE_0000_1 ///// ATTACHMENT 20H-2
        int INTR_data_0000_1[32];
        //0b11110110 010010000001000000000011
    /// BIT             = Value      Function                               Coding
        INTR_data_0000_1[ 0] = 1;         // Label 1st Digit(MSB)                3:  1
        INTR_data_0000_1[ 1] = 1;         // Label 1st Digit(LSB)                    1
        INTR_data_0000_1[ 2] = 1;         // Label 2nd Digit(MSB)                6:  1
        INTR_data_0000_1[ 3] = 1;         // Label 2nd Digit                         1
        INTR_data_0000_1[ 4] = 0;         // Label 2nd Digit(LSB)                    0
        INTR_data_0000_1[ 5] = 1;         // Label 3rd Digit(MSB)                6:  1
        INTR_data_0000_1[ 6] = 1;         // Label 3rd Digit                         1
        INTR_data_0000_1[ 7] = 0;         // Label 3rd Digit(LSB)                    0

        INTR_data_0000_1[15] = 1;         // Character#4(LSB)       0
        INTR_data_0000_1[14] = 0;         // Character#4            1
        INTR_data_0000_1[13] = 0;         // Character#4            0
        INTR_data_0000_1[12] = 0;         // Character#4            0
        INTR_data_0000_1[11] = 1;         // Character#4            1
        INTR_data_0000_1[10] = 1;         // Character#4(MSB)       0
        INTR_data_0000_1[ 9] = 0;         // Character#5(LSB)       1
        INTR_data_0000_1[ 8] = 1;         // Character#5            0

        INTR_data_0000_1[23] = 0;         // Character#5            1
        INTR_data_0000_1[22] = 0;         // Character#5            0
        INTR_data_0000_1[21] = 1;         // Character#5            1
        INTR_data_0000_1[20] = 1;         // Character#5(MSB)       0
        //                                                              7
        INTR_data_0000_1[19] = 0;         // Character#6(LSB)       0   1
        INTR_data_0000_1[18] = 1;         // Character#6            0   1
        INTR_data_0000_1[17] = 1;         // Character#6            1   1
        INTR_data_0000_1[16] = 0;         // Character#6            0   0

        INTR_data_0000_1[31] = 1;         // Character#6            0   1
        INTR_data_0000_1[30] = 1;         // Character#6(MSB)       0   1
        INTR_data_0000_1[29] = 0;         // Pad                                     0
        INTR_data_0000_1[28] = 0;         // Pad                                     0
        INTR_data_0000_1[27] = 0;         // Pad                                     0
        INTR_data_0000_1[26] = 0;         // Pad                                     0
        INTR_data_0000_1[25] = 0;         // Data Type Continuation Bit              0
        INTR_data_0000_1[24] = 0;         // Parity                                  (odd)
        check_odd (INTR_data_0000_1, INTR_data_0000_1[24]);

        uint INTR_Data_0000_1_tempt=0b0;
        PACKET(INTR_data_0000_1, INTR_Data_0000_1_tempt);
        TX_struct.INTR_TYPE_0000_1 = htonl(INTR_Data_0000_1_tempt);

///// TX_struct.TYPE_0000_2 ///// ATTACHMENT 20H-3
        int INTR_data_0000_2[32];
        //0b11110110001100110001100001000110
    /// BIT             = Value      Function                               Coding
        INTR_data_0000_2[ 0] = 1;         // Label 1st Digit(MSB)                3:  1
        INTR_data_0000_2[ 1] = 1;         // Label 1st Digit(LSB)                    1
        INTR_data_0000_2[ 2] = 1;         // Label 2nd Digit(MSB)                6:  1
        INTR_data_0000_2[ 3] = 1;         // Label 2nd Digit                         1
        INTR_data_0000_2[ 4] = 0;         // Label 2nd Digit(LSB)                    0
        INTR_data_0000_2[ 5] = 1;         // Label 3rd Digit(MSB)                6:  1
        INTR_data_0000_2[ 6] = 1;         // Label 3rd Digit                         1
        INTR_data_0000_2[ 7] = 0;         // Label 3rd Digit(LSB)                    0
        //                                                                  0    
        INTR_data_0000_2[15] = 1;         // Character#7(LSB)           1   0
        INTR_data_0000_2[14] = 0;         // Character#7                0   0
        INTR_data_0000_2[13] = 0;         // Character#7                1   0
        INTR_data_0000_2[12] = 1;         // Character#7                0   0
        INTR_data_0000_2[11] = 1;         // Character#7                0   1
        INTR_data_0000_2[10] = 1;         // Character#7(MSB)           0   1
        INTR_data_0000_2[ 9] = 0;         // Character#8(LSB)           0
        INTR_data_0000_2[ 8] = 0;         // Character#8                1
        
        INTR_data_0000_2[23] = 0;         // Character#8                0
        INTR_data_0000_2[22] = 0;         // Character#8                0
        INTR_data_0000_2[21] = 0;         // Character#8                1
        INTR_data_0000_2[20] = 0;         // Character#8(MSB)           0
        INTR_data_0000_2[19] = 1;         // Type Code(LSB)                 
        INTR_data_0000_2[18] = 0;         // Type Code          
        INTR_data_0000_2[17] = 0;         // Type Code(MSB)
        INTR_data_0000_2[16] = 0;         // Pad                                     0    

        INTR_data_0000_2[31] = 0;         // Aircraft Category(LSB)
        INTR_data_0000_2[30] = 1;         // Aircraft Category
        INTR_data_0000_2[29] = 0;         // Aircraft Category(MSB)                       
        INTR_data_0000_2[28] = 0;         // Pad                                     0
        INTR_data_0000_2[27] = 0;         // Pad                                     0
        INTR_data_0000_2[26] = 1;         // Pad                                     0
        INTR_data_0000_2[25] = 1;         // Data Type Continuation Bit              1
        INTR_data_0000_2[24] = 0;         // Parity                                  (odd)
        check_odd (INTR_data_0000_2, INTR_data_0000_2[24]);

        uint INTR_Data_0000_2_tempt=0b0;
        PACKET(INTR_data_0000_2, INTR_Data_0000_2_tempt);
        TX_struct.INTR_TYPE_0000_2 = htonl(INTR_Data_0000_2_tempt);

///// TX_struct.INTR_TYPE_0001 ///// ATTACHMENT 20I
        int INTR_data_0001[32];
    /// BIT             = Value      Function                               Coding
        INTR_data_0001[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        INTR_data_0001[ 1] = 1;         // Label 1st Digit(LSB)                  1
        INTR_data_0001[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        INTR_data_0001[ 3] = 1;         // Label 2nd Digit                       1
        INTR_data_0001[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        INTR_data_0001[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        INTR_data_0001[ 6] = 1;         // Label 3rd Digit                       1
        INTR_data_0001[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        INTR_data_0001[15] = 1;         // Data Type(LSB)                        1
        INTR_data_0001[14] = 0;         // Data Type                             0    
        INTR_data_0001[13] = 0;         // Data Type                             0
        INTR_data_0001[12] = 0;         // Data Type(MSB)                        0
        INTR_data_0001[11] = 1;         // Traffic Vert Sense            
        INTR_data_0001[10] = 1;         // Traffic Vert Sense
        INTR_data_0001[ 9] = 0;         // Coarse/Fine Range
        INTR_data_0001[ 8] = 0;         // Traffic Range 1/16 NM / 1/512 NM

        INTR_data_0001[23] = 0;         // Traffic Range 1/8  NM / 1/256 NM
        INTR_data_0001[22] = 0;         // Traffic Range 1/4  NM / 1/128 NM
        INTR_data_0001[21] = 0;         // Traffic Range 1/2  NM / 1/64  NM
        INTR_data_0001[20] = 0;         // Traffic Range 1    NM / 1/32  NM
        INTR_data_0001[19] = 0;         // Traffic Range 2    NM / 1/16  NM
        INTR_data_0001[18] = 0;         // Traffic Range 4    NM / 1/8   NM
        INTR_data_0001[17] = 0;         // Traffic Range 8    NM / 1/4   NM
        INTR_data_0001[16] = 0;         // Traffic Range 16   NM / 1/2   NM 

        INTR_data_0001[31] = 0;         // Traffic Range 32   NM / 1     NM
        INTR_data_0001[30] = 0;         // Traffic Range 64   NM / 2     NM
        INTR_data_0001[29] = 0;         // Traffic Range 128  NM / 4     NM             
        INTR_data_0001[28] = 0;         // Traffic Range 256  NM / 8     NM        
        INTR_data_0001[27] = 0;         // Traffic Range Invalidity    
        INTR_data_0001[26] = 1;         // Relative Alititude status
        INTR_data_0001[25] = 0;         // Data Type Continuation Bit              
        INTR_data_0001[24] = 0;         // Parity                                  (odd)



////////////////////////////////////////////////////////////////
/// Range test

        float intr_range = intr_est_range ;
        

        if (intr_range > 16.0){
            INTR_data_0001[ 9] = 1;         // Coarse/Fine Range
            if(intr_range > 8           ){ intr_range = intr_range - 8;         INTR_data_0001[28] = 1;};
            if(intr_range > 4           ){ intr_range = intr_range - 4;         INTR_data_0001[29] = 1;};
            if(intr_range > 2           ){ intr_range = intr_range - 2;         INTR_data_0001[30] = 1;};
            if(intr_range > 1           ){ intr_range = intr_range - 1;         INTR_data_0001[31] = 1;};
            if(intr_range > 0.5         ){ intr_range = intr_range - 0.5;       INTR_data_0001[16] = 1;};
            if(intr_range > 0.25        ){ intr_range = intr_range - 0.25;      INTR_data_0001[17] = 1;};
            if(intr_range > 0.125       ){ intr_range = intr_range - 0.125;     INTR_data_0001[18] = 1;};
            if(intr_range > 0.0625      ){ intr_range = intr_range - 0.0625;    INTR_data_0001[19] = 1;};
            if(intr_range > 0.03125     ){ intr_range = intr_range - 0.03125;   INTR_data_0001[20] = 1;};
            if(intr_range > 0.015625    ){ intr_range = intr_range - 0.015625;  INTR_data_0001[21] = 1;};
            if(intr_range > 0.0078125   ){ intr_range = intr_range - 0.0078125; INTR_data_0001[22] = 1;};
            if(intr_range > 0.00390625  ){ intr_range = intr_range - 0.00390625;INTR_data_0001[23] = 1;};
            if(intr_range > 0.00195313  ){ intr_range = intr_range - 0.00195313;INTR_data_0001[ 8] = 1;};
      
            INTR_data_0001[27] = 1;         // Traffic Range Invalidity    
        }
        else{
            INTR_data_0001[ 9] = 0;         // Coarse/Fine Range
            if(intr_range > 256         ){ intr_range = intr_range - 256;   INTR_data_0001[28] = 1;}; 
            if(intr_range > 128         ){ intr_range = intr_range - 128;   INTR_data_0001[29] = 1;};
            if(intr_range > 64          ){ intr_range = intr_range - 64;    INTR_data_0001[30] = 1;};
            if(intr_range > 32          ){ intr_range = intr_range - 32;    INTR_data_0001[31] = 1;};
            if(intr_range > 16          ){ intr_range = intr_range - 16;    INTR_data_0001[16] = 1;};
            if(intr_range > 8           ){ intr_range = intr_range - 8;     INTR_data_0001[17] = 1;};
            if(intr_range > 4           ){ intr_range = intr_range - 4;     INTR_data_0001[18] = 1;};
            if(intr_range > 2           ){ intr_range = intr_range - 2;     INTR_data_0001[19] = 1;};
            if(intr_range > 1           ){ intr_range = intr_range - 1;     INTR_data_0001[20] = 1;};
            if(intr_range > 0.5         ){ intr_range = intr_range - 0.5;   INTR_data_0001[21] = 1;};
            if(intr_range > 0.25        ){ intr_range = intr_range - 0.25;  INTR_data_0001[22] = 1;};
            if(intr_range > 0.125       ){ intr_range = intr_range - 0.125; INTR_data_0001[23] = 1;};
            if(intr_range > 0.0625      ){ intr_range = intr_range - 0.0625;INTR_data_0001[ 8] = 1;};
          
            INTR_data_0001[27] = 1;         // Traffic Range Invalidity    
        }



///
/////////////////////////////////////////////////////////////////

        check_odd (INTR_data_0001, INTR_data_0001[24]);

        uint INTR_Data_0001_tempt=0b0;
        PACKET(INTR_data_0001, INTR_Data_0001_tempt);
        TX_struct.INTR_TYPE_0001_0 = htonl(INTR_Data_0001_tempt);

///// TX_struct.INTR_TYPE_0001_1 ///// ATTACHMENT 20I
        int INTR_data_0001_1[32];
    /// BIT             = Value      Function                               Coding
        INTR_data_0001_1[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        INTR_data_0001_1[ 1] = 1;         // Label 1st Digit(LSB)                  1
        INTR_data_0001_1[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        INTR_data_0001_1[ 3] = 1;         // Label 2nd Digit                       1
        INTR_data_0001_1[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        INTR_data_0001_1[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        INTR_data_0001_1[ 6] = 1;         // Label 3rd Digit                       1
        INTR_data_0001_1[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        INTR_data_0001_1[15] = 0;         // Relative Altitude 100FT                      
        INTR_data_0001_1[14] = 0;         // Relative Altitude 200                    
        INTR_data_0001_1[13] = 0;         // Relative Altitude 400                 
        INTR_data_0001_1[12] = 0;         // Relative Altitude 800             
        INTR_data_0001_1[11] = 0;         // Relative Altitude 1600         
        INTR_data_0001_1[10] = 0;         // Relative Altitude 3200
        INTR_data_0001_1[ 9] = 0;         // Relative Altitude 6400
        INTR_data_0001_1[ 8] = 0;         // Relative Altitude 12800

        INTR_data_0001_1[23] = 0;         // Relative Altitude 256000
        INTR_data_0001_1[22] = 0;         // Relative Altitude Sign
        INTR_data_0001_1[21] = 0;         // Traffic Bearing 0.175781 Deg
        INTR_data_0001_1[20] = 0;         // Traffic Bearing 0.351563
        INTR_data_0001_1[19] = 0;         // Traffic Bearing 0.703125
        INTR_data_0001_1[18] = 0;         // Traffic Bearing 1.40625
        INTR_data_0001_1[17] = 0;         // Traffic Bearing 2.8125
        INTR_data_0001_1[16] = 0;         // Traffic Bearing 5.625

        INTR_data_0001_1[31] = 0;         // Traffic Bearing 11.25
        INTR_data_0001_1[30] = 0;         // Traffic Bearing 22.5
        INTR_data_0001_1[29] = 0;         // Traffic Bearing 45              
        INTR_data_0001_1[28] = 0;         // Traffic Bearing 90        
        INTR_data_0001_1[27] = 0;         // Traffic Bearing Sign 
        INTR_data_0001_1[26] = 0;         // Bearing status
        INTR_data_0001_1[25] = 1;         // Data Type COntinuation Bit              
        INTR_data_0001_1[24] = 0;         // Parity                                  (odd)

////////////////////////////////////////////////////////////////
// Bearing test
        // intr_bearing = -30.0; ///

        float detect_x = detection_result[3] + 0.5*detection_result[1];
        intr_bearing = (detect_x/(0.5*CAMwidth)) *(0.5*CAMFOV);//(0.5*CAMwidth - abs(detect_x) )/ (0.5*CAMFOV);

        if (CAM_SELECTION==1){  // Left View
            intr_bearing = -ANG_SIDECAM + intr_bearing;
            cout << "CAM_LEFT" << endl;
        }
        else if(CAM_SELECTION==2){   // Right View
            intr_bearing = ANG_SIDECAM + intr_bearing;
            cout << "CAM_RIGHT" << endl;
        }
        else{
            cout << "CAM_FRONT" << endl;
        }
        
        cout << " detect_bearing : " << intr_bearing << endl;

        if (intr_bearing >= 0){
            INTR_data_0001_1[27] = 0;         // Bearing status

            if(intr_bearing > 90)           { intr_bearing = intr_bearing - 90;         INTR_data_0001_1[28] = 1;}
            if(intr_bearing > 45)           { intr_bearing = intr_bearing - 45;         INTR_data_0001_1[29] = 1;}
            if(intr_bearing > 22.5)         { intr_bearing = intr_bearing - 22.5;       INTR_data_0001_1[30] = 1;}
            if(intr_bearing > 11.25)        { intr_bearing = intr_bearing - 11.25;      INTR_data_0001_1[31] = 1;}
            if(intr_bearing > 5.625)        { intr_bearing = intr_bearing - 5.625;      INTR_data_0001_1[16] = 1;}
            if(intr_bearing > 2.8125)       { intr_bearing = intr_bearing - 2.8125;     INTR_data_0001_1[17] = 1;}
            if(intr_bearing > 1.40625)      { intr_bearing = intr_bearing - 1.40625;    INTR_data_0001_1[18] = 1;}
            if(intr_bearing > 0.703125)     { intr_bearing = intr_bearing - 0.703125;   INTR_data_0001_1[19] = 1;}
            if(intr_bearing > 0.351563)     { intr_bearing = intr_bearing - 0.351563;   INTR_data_0001_1[20] = 1;}
            if(intr_bearing > 0.175781)     { intr_bearing = intr_bearing - 0.175781;   INTR_data_0001_1[21] = 1;}
            }
        else{
            
            float intr_bearing_abs  = abs(intr_bearing);
            intr_bearing            = 180.0 - intr_bearing_abs;

            INTR_data_0001_1[27] = 1;         // Bearing status
            
            if(intr_bearing > 90)           { intr_bearing = intr_bearing - 90;         INTR_data_0001_1[28] = 1;}
            if(intr_bearing > 45)           { intr_bearing = intr_bearing - 45;         INTR_data_0001_1[29] = 1;}
            if(intr_bearing > 22.5)         { intr_bearing = intr_bearing - 22.5;       INTR_data_0001_1[30] = 1;}
            if(intr_bearing > 11.25)        { intr_bearing = intr_bearing - 11.25;      INTR_data_0001_1[31] = 1;}
            if(intr_bearing > 5.625)        { intr_bearing = intr_bearing - 5.625;      INTR_data_0001_1[16] = 1;}
            if(intr_bearing > 2.8125)       { intr_bearing = intr_bearing - 2.8125;     INTR_data_0001_1[17] = 1;}
            if(intr_bearing > 1.40625)      { intr_bearing = intr_bearing - 1.40625;    INTR_data_0001_1[18] = 1;}
            if(intr_bearing > 0.703125)     { intr_bearing = intr_bearing - 0.703125;   INTR_data_0001_1[19] = 1;}
            if(intr_bearing > 0.351563)     { intr_bearing = intr_bearing - 0.351563;   INTR_data_0001_1[20] = 1;}
            if(intr_bearing > 0.175781)     { intr_bearing = intr_bearing - 0.175781;   INTR_data_0001_1[21] = 1;}
        }

        INTR_data_0001_1[26] = 1;         // Bearing status
        

////////////////////////////////////////////////////////////////


        // double own_alt = (double)bag_own_alt *m2ft;
        double intr_alt = 0;

#ifdef GPS_POS
        own_alt     = (double)bag_own_alt * m2ft;
        intr_alt    = (double)bag_int_alt * m2ft;
#endif

#ifdef ADSB_POS
        own_alt     = OWNSHIP_DATA.altitude  * m2ft;
        intr_alt    = INTRUDER_DATA.altitude * m2ft;
#endif

        intr_alt = intr_alt - own_alt;

        if (intr_alt<0){
            intr_alt = abs(intr_alt);
            INTR_data_0001_1[22] = 1;

            INTR_data_0001_1[23] = 1;
            INTR_data_0001_1[ 8] = 1;
            INTR_data_0001_1[ 9] = 1;
            INTR_data_0001_1[10] = 1;
            INTR_data_0001_1[11] = 1;
            INTR_data_0001_1[12] = 1;
            INTR_data_0001_1[13] = 1;
            INTR_data_0001_1[14] = 1;
            INTR_data_0001_1[15] = 1;

            if(intr_alt > 25600)     { intr_alt = intr_alt - 25600;       INTR_data_0001_1[23] = 0;}
            if(intr_alt > 12800)     { intr_alt = intr_alt - 12800;       INTR_data_0001_1[ 8] = 0;}
            if(intr_alt > 6400)      { intr_alt = intr_alt - 6400;        INTR_data_0001_1[ 9] = 0;}
            if(intr_alt > 3200)      { intr_alt = intr_alt - 3200;        INTR_data_0001_1[10] = 0;}
            if(intr_alt > 1600)      { intr_alt = intr_alt - 1600;        INTR_data_0001_1[11] = 0;}
            if(intr_alt > 800)       { intr_alt = intr_alt - 800;         INTR_data_0001_1[12] = 0;}
            if(intr_alt > 400)       { intr_alt = intr_alt - 400;         INTR_data_0001_1[13] = 0;}
            if(intr_alt > 200)       { intr_alt = intr_alt - 200;         INTR_data_0001_1[14] = 0;}
            if(intr_alt > 100)       { intr_alt = intr_alt - 100;         INTR_data_0001_1[15] = 0;}
        }
        else{
            INTR_data_0001_1[22] = 0;
            // cout << "intr_alt       : " <<intr_alt << " ft" << endl;
            if(intr_alt > 25600)     { intr_alt = intr_alt - 25600;       INTR_data_0001_1[23] = 1;}
            if(intr_alt > 12800)     { intr_alt = intr_alt - 12800;       INTR_data_0001_1[ 8] = 1;}
            if(intr_alt > 6400)      { intr_alt = intr_alt - 6400;        INTR_data_0001_1[ 9] = 1;}
            if(intr_alt > 3200)      { intr_alt = intr_alt - 3200;        INTR_data_0001_1[10] = 1;}
            if(intr_alt > 1600)      { intr_alt = intr_alt - 1600;        INTR_data_0001_1[11] = 1;}
            if(intr_alt > 800)       { intr_alt = intr_alt - 800;         INTR_data_0001_1[12] = 1;}
            if(intr_alt > 400)       { intr_alt = intr_alt - 400;         INTR_data_0001_1[13] = 1;}
            if(intr_alt > 200)       { intr_alt = intr_alt - 200;         INTR_data_0001_1[14] = 1;}
            if(intr_alt > 100)       { intr_alt = intr_alt - 100;         INTR_data_0001_1[15] = 1;}
        }

        check_odd (INTR_data_0001_1, INTR_data_0001_1[24]);

        uint INTR_Data_0001_1_tempt=0b0;
        PACKET(INTR_data_0001_1, INTR_Data_0001_1_tempt);
        TX_struct.INTR_TYPE_0001_1 = htonl(INTR_Data_0001_1_tempt);

///// TX_struct.INTR_TYPE_0002 ///// ATTACHMENT 20J
        int INTR_data_0002[32];
        // 0b 11110110 00110010 00110001 10000011
    /// BIT             = Value      Function                               Coding
        INTR_data_0002[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        INTR_data_0002[ 1] = 1;         // Label 1st Digit(LSB)                  1
        INTR_data_0002[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        INTR_data_0002[ 3] = 1;         // Label 2nd Digit                       1
        INTR_data_0002[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        INTR_data_0002[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        INTR_data_0002[ 6] = 1;         // Label 3rd Digit                       1
        INTR_data_0002[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        INTR_data_0002[15] = 0;         // Data Type(LSB)                        0
        INTR_data_0002[14] = 1;         // Data Type                             1    
        INTR_data_0002[13] = 0;         // Data Type                             0
        INTR_data_0002[12] = 0;         // Data Type(MSB)                        0
        INTR_data_0002[11] = 0;         // Latitude - 1.0728836 x 10^(-5) Deg 
        INTR_data_0002[10] = 0;         // Latitude - 2.1457672 x 10^(-5) 
        INTR_data_0002[ 9] = 0;         // Latitude - 4.2915344 x 10^(-5) 
        INTR_data_0002[ 8] = 0;         // Latitude - 8.5830688 x 10^(-5) 

        INTR_data_0002[23] = 0;         // Latitude - 1.7166118 x 10^(-4) 
        INTR_data_0002[22] = 0;         // Latitude - 3.4332275 x 10^(-4) 
        INTR_data_0002[21] = 0;         // Latitude - 6.8664551 x 10^(-4) 
        INTR_data_0002[20] = 0;         // Latitude - 0.00137329
        INTR_data_0002[19] = 0;         // Latitude - 0.00274658
        INTR_data_0002[18] = 0;         // Latitude - 0.00549316
        INTR_data_0002[17] = 0;         // Latitude - 0.0109863
        INTR_data_0002[16] = 0;         // Latitude - 0.0219726     

        INTR_data_0002[31] = 0;         // Latitude - 0.0439453
        INTR_data_0002[30] = 0;         // Latitude - 0.0878906
        INTR_data_0002[29] = 0;         // Latitude - 0.175781             
        INTR_data_0002[28] = 0;         // Latitude - 0.351563        
        INTR_data_0002[27] = 0;         // Latitude - 0.703125    
        INTR_data_0002[26] = 0;         // Latitude - 1.40625
        INTR_data_0002[25] = 0;         // Data Type Continuation Bit              
        INTR_data_0002[24] = 0;         // Parity                                  (odd)


///// TX_struct.INTR_TYPE_0002_1 ///// ATTACHMENT 20J
        int INTR_data_0002_1[32];
        // 0b 11110110 01010001 00001001 00100001

    /// BIT             = Value      Function                               Coding
        INTR_data_0002_1[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        INTR_data_0002_1[ 1] = 1;         // Label 1st Digit(LSB)                  1
        INTR_data_0002_1[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        INTR_data_0002_1[ 3] = 1;         // Label 2nd Digit                       1
        INTR_data_0002_1[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        INTR_data_0002_1[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        INTR_data_0002_1[ 6] = 1;         // Label 3rd Digit                       1
        INTR_data_0002_1[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        INTR_data_0002_1[15] = 0;         // Latitude - 2.8125
        INTR_data_0002_1[14] = 0;         // Latitude - 5.625                          
        INTR_data_0002_1[13] = 0;         // Latitude - 11.25
        INTR_data_0002_1[12] = 0;         // Latitude - 22.5                        
        INTR_data_0002_1[11] = 0;         // Latitude - 45 
        INTR_data_0002_1[10] = 0;         // Latitude - Sign
        INTR_data_0002_1[ 9] = 0;         // Longitude - 1.0728836 x 10^(-5) Deg
        INTR_data_0002_1[ 8] = 0;         // Longitude - 2.1457672 x 10^(-5) Deg

        INTR_data_0002_1[23] = 0;         // Longitude - 4.2915344 x 10^(-5) Deg
        INTR_data_0002_1[22] = 0;         // Longitude - 8.5830688 x 10^(-5) Deg
        INTR_data_0002_1[21] = 0;         // Longitude - 1.7166118 x 10^(-4) Deg
        INTR_data_0002_1[20] = 0;         // Longitude - 3.4332275 x 10^(-4) Deg
        INTR_data_0002_1[19] = 0;         // Longitude - 6.8664551 x 10^(-4) Deg
        INTR_data_0002_1[18] = 0;         // Longitude - 0.00137329
        INTR_data_0002_1[17] = 0;         // Longitude - 0.00274658
        INTR_data_0002_1[16] = 0;         // Longitude - 0.00549316

        INTR_data_0002_1[31] = 0;         // Longitude - 0.0109863
        INTR_data_0002_1[30] = 0;         // Longitude - 0.0219726                                        
        INTR_data_0002_1[29] = 0;         // Longitude - 0.0439453
        INTR_data_0002_1[28] = 0;         // Longitude - 0.0878906
        INTR_data_0002_1[27] = 0;         // Longitude - 0.175781             
        INTR_data_0002_1[26] = 0;         // Longitude - 0.351563 
        INTR_data_0002_1[25] = 0;         // Data Type Continuation Bit              
        INTR_data_0002_1[24] = 0;         // Parity                                  (odd)



///// TX_struct.TYPE_0002_2 ///// ATTACHMENT 20J
        int INTR_data_0002_2[32];
        //0b11110110 10001010 00000101 01100000
    /// BIT             = Value      Function                               Coding
        INTR_data_0002_2[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        INTR_data_0002_2[ 1] = 1;         // Label 1st Digit(LSB)                  1
        INTR_data_0002_2[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        INTR_data_0002_2[ 3] = 1;         // Label 2nd Digit                       1
        INTR_data_0002_2[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        INTR_data_0002_2[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        INTR_data_0002_2[ 6] = 1;         // Label 3rd Digit                       1
        INTR_data_0002_2[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        INTR_data_0002_2[15] = 0;         // Longitude - 0.703125    
        INTR_data_0002_2[14] = 0;         // Longitude - 1.40625                         
        INTR_data_0002_2[13] = 0;         // Longitude - 2.8125
        INTR_data_0002_2[12] = 0;         // Longitude - 5.625                          
        INTR_data_0002_2[11] = 0;         // Longitude - 11.25
        INTR_data_0002_2[10] = 0;         // Longitude - 22.5                        
        INTR_data_0002_2[ 9] = 0;         // Longitude - 45 
        INTR_data_0002_2[ 8] = 0;         // Longitude - 90

        INTR_data_0002_2[23] = 0;         // Longitude - Sign
        INTR_data_0002_2[22] = 0;         // Vertical Speed - 16 FT/MIN
        INTR_data_0002_2[21] = 0;         // Vertical Speed - 32 FT/MIN
        INTR_data_0002_2[20] = 0;         // Vertical Speed - 64 FT/MIN
        INTR_data_0002_2[19] = 0;         // Vertical Speed - 128 FT/MIN
        INTR_data_0002_2[18] = 0;         // Vertical Speed - 256 FT/MIN
        INTR_data_0002_2[17] = 0;         // Vertical Speed - 512 FT/MIN
        INTR_data_0002_2[16] = 0;         // Vertical Speed - 1024 FT/MIN

        INTR_data_0002_2[31] = 0;         // Vertical Speed - 2048 FT/MIN
        INTR_data_0002_2[30] = 0;         // Vertical Speed - 4096 FT/MIN                                       
        INTR_data_0002_2[29] = 0;         // Vertical Speed - 8192 FT/MIN
        INTR_data_0002_2[28] = 0;         // Vertical Speed - 16384 FT/MIN
        INTR_data_0002_2[27] = 0;         // Vertical Speed - sign    
        INTR_data_0002_2[26] = 1;         // Veltical Speed Status
        INTR_data_0002_2[25] = 1;         // Data Type Continuation Bit              1  
        INTR_data_0002_2[24] = 0;         // Parity                                  (odd)

#ifdef GPS_POS
        double intr_lat = (double)bag_int_lat;
        double intr_lon = (double)bag_int_lon;
#endif

#ifdef ADSB_POS
        double intr_lat = (double)INTRUDER_DATA.lat/10000000;
        double intr_lon = (double)INTRUDER_DATA.lon/10000000;
#endif
        double intr_ver_vel = (float)INTRUDER_DATA.ver_velocity/100.0 * mps2ftpmin;

        // cout << INTRUDER_DATA.lat<< endl;
        
        // printf("own_lat : %f\n", own_lat);
        // LATITUDE
        if(intr_lat > 45.0)              { intr_lat = intr_lat - 45.0;         INTR_data_0002_1[11] = 1;}
        if(intr_lat > 22.5)              { intr_lat = intr_lat - 22.5;         INTR_data_0002_1[12] = 1;}
        if(intr_lat > 11.25)             { intr_lat = intr_lat - 11.25;        INTR_data_0002_1[13] = 1;}
        if(intr_lat > 5.625)             { intr_lat = intr_lat - 5.625;        INTR_data_0002_1[14] = 1;}
        if(intr_lat>2.8125)              { intr_lat = intr_lat - 2.8125;       INTR_data_0002_1[15] = 1;}
        if(intr_lat>1.40625)             { intr_lat = intr_lat - 1.40625;      INTR_data_0002[26] = 1;  }
        if(intr_lat>0.703125 )           { intr_lat = intr_lat - 0.703125;     INTR_data_0002[27] = 1;  }
        if(intr_lat>0.351563)            { intr_lat = intr_lat - 0.351563;     INTR_data_0002[28] = 1;  }
        if(intr_lat>0.175781)            { intr_lat = intr_lat - 0.175781;     INTR_data_0002[29] = 1;  }
        if(intr_lat>0.0878906)           { intr_lat = intr_lat - 0.0878906;    INTR_data_0002[30] = 1;  }
        if(intr_lat>0.0439453)           { intr_lat = intr_lat - 0.0439453;    INTR_data_0002[31] = 1;  }
        if(intr_lat>0.0219726)           { intr_lat = intr_lat - 0.0219726;    INTR_data_0002[16] = 1;  }
        if(intr_lat>0.0109863)           { intr_lat = intr_lat - 0.0109863;    INTR_data_0002[17] = 1;  }
        if(intr_lat>0.00549316)          { intr_lat = intr_lat - 0.00549316;   INTR_data_0002[18] = 1;  }
        if(intr_lat>0.00274658)          { intr_lat = intr_lat - 0.00274658;   INTR_data_0002[19] = 1;  }
        if(intr_lat>0.00137329)          { intr_lat = intr_lat - 0.00137329;   INTR_data_0002[20] = 1;  }
        if(intr_lat> 6.8664551/10000)    { intr_lat = intr_lat - (6.8664551/10000);    INTR_data_0002[21] = 1;  }
        if(intr_lat> 3.4332275/10000)    { intr_lat = intr_lat - (3.4332275/10000);    INTR_data_0002[22] = 1;  }
        if(intr_lat>1.7166118/10000)     { intr_lat = intr_lat - (1.7166118/10000);    INTR_data_0002[23] = 1;  }
        if(intr_lat>8.5830688/100000)    { intr_lat = intr_lat - (8.5830688/100000);   INTR_data_0002[ 8] = 1;   }
        if(intr_lat>4.2915344/100000)    { intr_lat = intr_lat - (4.2915344/100000);   INTR_data_0002[ 9] = 1;   }
        if(intr_lat>2.1457672/100000)    { intr_lat = intr_lat - (2.1457672/100000);   INTR_data_0002[10] = 1;  }
        if(intr_lat>1.0728836/100000)    { intr_lat = intr_lat - (1.0728836/100000);   INTR_data_0002[11] = 1;  }


        /////Longitude
        // LATITUDE
        if(intr_lon > 90.0)              { intr_lon = intr_lon - 90.0;         INTR_data_0002_2[ 8] = 1;    }
        if(intr_lon > 45.0)              { intr_lon = intr_lon - 45.0;         INTR_data_0002_2[ 9] = 1;    }
        if(intr_lon > 22.5)              { intr_lon = intr_lon - 22.5;         INTR_data_0002_2[10] = 1;    }
        if(intr_lon > 11.25)             { intr_lon = intr_lon - 11.25;        INTR_data_0002_2[11] = 1;    }
        if(intr_lon > 5.625)             { intr_lon = intr_lon - 5.625;        INTR_data_0002_2[12] = 1;    }
        if(intr_lon>2.8125)              { intr_lon = intr_lon - 2.8125;       INTR_data_0002_2[13] = 1;    }
        if(intr_lon>1.40625)             { intr_lon = intr_lon - 1.40625;      INTR_data_0002_2[14] = 1;    }
        if(intr_lon>0.703125)            { intr_lon = intr_lon - 0.703125;     INTR_data_0002_2[15] = 1;    }
        if(intr_lon>0.351563)            { intr_lon = intr_lon - 0.351563;     INTR_data_0002_1[26] = 1;    }
        if(intr_lon>0.175781)            { intr_lon = intr_lon - 0.175781;     INTR_data_0002_1[27] = 1;    }
        if(intr_lon>0.0878906)           { intr_lon = intr_lon - 0.0878906;    INTR_data_0002_1[28] = 1;    }
        if(intr_lon>0.0439453)           { intr_lon = intr_lon - 0.0439453;    INTR_data_0002_1[29] = 1;    }
        if(intr_lon>0.0219726)           { intr_lon = intr_lon - 0.0219726;    INTR_data_0002_1[30] = 1;    }
        if(intr_lon>0.0109863)           { intr_lon = intr_lon - 0.0109863;    INTR_data_0002_1[31] = 1;    }
        if(intr_lon>0.00549316)          { intr_lon = intr_lon - 0.00549316;   INTR_data_0002_1[16] = 1;    }
        if(intr_lon>0.00274658)          { intr_lon = intr_lon - 0.00274658;   INTR_data_0002_1[17] = 1;    }
        if(intr_lon>0.00137329)          { intr_lon = intr_lon - 0.00137329;   INTR_data_0002_1[18] = 1;    }
        if(intr_lon> 6.8664551/10000)    { intr_lon = intr_lon - (6.8664551/10000);    INTR_data_0002_1[19] = 1;    }
        if(intr_lon> 3.4332275/10000)    { intr_lon = intr_lon - (3.4332275/10000);    INTR_data_0002_1[20] = 1;    }
        if(intr_lon>1.7166118/10000)     { intr_lon = intr_lon - (1.7166118/10000);    INTR_data_0002_1[21] = 1;    }
        if(intr_lon>8.5830688/100000)    { intr_lon = intr_lon - (8.5830688/100000);   INTR_data_0002_1[22] = 1;    }
        if(intr_lon>4.2915344/100000)    { intr_lon = intr_lon - (4.2915344/100000);   INTR_data_0002_1[23] = 1;    }
        if(intr_lon>2.1457672/100000)    { intr_lon = intr_lon - (2.1457672/100000);   INTR_data_0002_1[ 8] = 1;    }
        if(intr_lon>1.0728836/100000)    { intr_lon = intr_lon - (1.0728836/100000);   INTR_data_0002_1[ 9] = 1;    }
        
        
        // Vertical Velocity
        if(intr_ver_vel > 16384)     { intr_ver_vel = intr_ver_vel - 16384;       INTR_data_0002_2[28] = 1;}
        if(intr_ver_vel > 8192)      { intr_ver_vel = intr_ver_vel - 8192;        INTR_data_0002_2[29] = 1;}
        if(intr_ver_vel > 4096)      { intr_ver_vel = intr_ver_vel - 4096;        INTR_data_0002_2[30] = 1;}
        if(intr_ver_vel > 2048)      { intr_ver_vel = intr_ver_vel - 2048;        INTR_data_0002_2[31] = 1;}
        if(intr_ver_vel > 1024)      { intr_ver_vel = intr_ver_vel - 1024;        INTR_data_0002_2[16] = 1;}
        if(intr_ver_vel > 512)       { intr_ver_vel = intr_ver_vel - 512;         INTR_data_0002_2[17] = 1;}
        if(intr_ver_vel > 256)       { intr_ver_vel = intr_ver_vel - 256;         INTR_data_0002_2[18] = 1;}
        if(intr_ver_vel > 128)       { intr_ver_vel = intr_ver_vel - 128;         INTR_data_0002_2[19] = 1;}
        if(intr_ver_vel > 64)        { intr_ver_vel = intr_ver_vel - 64;          INTR_data_0002_2[20] = 1;}
        if(intr_ver_vel > 32)        { intr_ver_vel = intr_ver_vel - 32;          INTR_data_0002_2[21] = 1;}
        if(intr_ver_vel > 16)        { intr_ver_vel = intr_ver_vel - 16;          INTR_data_0002_2[22] = 1;}

        check_odd(INTR_data_0002, INTR_data_0002[24]);
        uint INTR_Data_0002_tempt=0b0;
        PACKET(INTR_data_0002, INTR_Data_0002_tempt);
        // TX_struct.INTR_TYPE_0002_0 = htonl(INTR_Data_0002_tempt);

        check_odd(INTR_data_0002_1, INTR_data_0002_1[24]);
        uint INTR_Data_0002_1_tempt=0b0;
        PACKET(INTR_data_0002_1, INTR_Data_0002_1_tempt);
        // TX_struct.INTR_TYPE_0002_1 = htonl(INTR_Data_0002_1_tempt);

        check_odd(INTR_data_0002_2, INTR_data_0002_2[24]);
        uint INTR_Data_0002_2_tempt=0b0;
        PACKET(INTR_data_0002_2, INTR_Data_0002_2_tempt);
        // TX_struct.INTR_TYPE_0002_2 = htonl(INTR_Data_0002_2_tempt);


/// TX_struct.INTR_TYPE_0003 ///// ATTACHMENT 20L  -----------------------------------From here, modify 
        int INTR_data_0003[32];
    /// BIT             = Value      Function                               Coding
        INTR_data_0003[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        INTR_data_0003[ 1] = 1;         // Label 1st Digit(LSB)                  1
        INTR_data_0003[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        INTR_data_0003[ 3] = 1;         // Label 2nd Digit                       1
        INTR_data_0003[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        INTR_data_0003[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        INTR_data_0003[ 6] = 1;         // Label 3rd Digit                       1
        INTR_data_0003[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        INTR_data_0003[15] = 1;         // Data Type (LSB)
        INTR_data_0003[14] = 1;         // Data Type
        INTR_data_0003[13] = 0;         // Data Type
        INTR_data_0003[12] = 0;         // Data Type (MSB)
        INTR_data_0003[11] = 0;         // Pad
        INTR_data_0003[10] = 0;         // Pad 
        INTR_data_0003[ 9] = 0;         // Magnetic(1)/True Heading(0) Indication
        INTR_data_0003[ 8] = 0;         // Ground Speed   1 knot

        INTR_data_0003[23] = 0;         // Ground Speed   2
        INTR_data_0003[22] = 0;         // Ground Speed   4
        INTR_data_0003[21] = 0;         // Ground Speed   8
        INTR_data_0003[20] = 0;         // Ground Speed   16
        INTR_data_0003[19] = 0;         // Ground Speed   32
        INTR_data_0003[18] = 0;         // Ground Speed   64
        INTR_data_0003[17] = 0;         // Ground Speed   128
        INTR_data_0003[16] = 0;         // Ground Speed   256

        INTR_data_0003[31] = 0;         // Ground Speed   512                                        
        INTR_data_0003[30] = 0;         // Ground Speed   1024
        INTR_data_0003[29] = 0;         // Ground Speed   2048
        INTR_data_0003[28] = 0;         // Pad             
        INTR_data_0003[27] = 1;         // Ground Speed Status        
        INTR_data_0003[26] = 1;         // Track Angle Status    
        INTR_data_0003[25] = 0;         // Data Type COntinuation Bit              
        INTR_data_0003[24] = 0;         // Parity                                  (odd)

        float int_hor_vel = (float)INTRUDER_DATA.hor_velocity /100.0 * mps2knot; //[knot]
        // cout << "int_hor_vel : " << int_hor_vel << " knot , " << (float)INTRUDER_DATA.hor_velocity /100.0 << " m/s"<<endl;

        // Filtered Vision-based estimated velocity
        int_hor_vel = vel_abs_lpf;

        if(int_hor_vel > 2048)     { int_hor_vel = int_hor_vel - 2048;       INTR_data_0003[29] = 1;}
        if(int_hor_vel > 1024)     { int_hor_vel = int_hor_vel - 1024;       INTR_data_0003[30] = 1;}
        if(int_hor_vel > 512)      { int_hor_vel = int_hor_vel - 512;        INTR_data_0003[31] = 1;}
        if(int_hor_vel > 256)      { int_hor_vel = int_hor_vel - 256;        INTR_data_0003[16] = 1;}
        if(int_hor_vel > 128)      { int_hor_vel = int_hor_vel - 128;        INTR_data_0003[17] = 1;}
        if(int_hor_vel > 64)       { int_hor_vel = int_hor_vel - 64;         INTR_data_0003[18] = 1;}
        if(int_hor_vel > 32)       { int_hor_vel = int_hor_vel - 32;         INTR_data_0003[19] = 1;}
        if(int_hor_vel > 16)       { int_hor_vel = int_hor_vel - 16;         INTR_data_0003[20] = 1;}
        if(int_hor_vel > 8)        { int_hor_vel = int_hor_vel - 8;          INTR_data_0003[21] = 1;}
        if(int_hor_vel > 4)        { int_hor_vel = int_hor_vel - 4;          INTR_data_0003[22] = 1;}
        if(int_hor_vel > 2)        { int_hor_vel = int_hor_vel - 2;          INTR_data_0003[23] = 1;}
        if(int_hor_vel > 1)        { int_hor_vel = int_hor_vel - 1;          INTR_data_0003[ 8] = 1;}

        check_odd (INTR_data_0003, INTR_data_0003[24]);

        uint INTR_Data_0003_tempt=0b0;
        PACKET(INTR_data_0003, INTR_Data_0003_tempt);
        TX_struct.INTR_TYPE_0003_0 = htonl(INTR_Data_0003_tempt);

///// TX_struct.INTR_TYPE_0003_1 ///// ATTACHMENT 20L
        int INTR_data_0003_1[32];
    /// BIT             = Value      Function                               Coding
        INTR_data_0003_1[ 0] = 1;         // Label 1st Digit(MSB)              3:  1
        INTR_data_0003_1[ 1] = 1;         // Label 1st Digit(LSB)                  1
        INTR_data_0003_1[ 2] = 1;         // Label 2nd Digit(MSB)              6:  1
        INTR_data_0003_1[ 3] = 1;         // Label 2nd Digit                       1
        INTR_data_0003_1[ 4] = 0;         // Label 2nd Digit(LSB)                  0
        INTR_data_0003_1[ 5] = 1;         // Label 3rd Digit(MSB)              6:  1
        INTR_data_0003_1[ 6] = 1;         // Label 3rd Digit                       1
        INTR_data_0003_1[ 7] = 0;         // Label 3rd Digit(LSB)                  0

        INTR_data_0003_1[15] = 0;         // Heading/Track Angle 1.40625 deg.
        INTR_data_0003_1[14] = 0;         // Heading/Track Angle 2.8125                      
        INTR_data_0003_1[13] = 0;         // Heading/Track Angle 5.625
        INTR_data_0003_1[12] = 0;         // Heading/Track Angle 11.25              
        INTR_data_0003_1[11] = 0;         // Heading/Track Angle 22.5
        INTR_data_0003_1[10] = 0;         // Heading/Track Angle 45.0
        INTR_data_0003_1[ 9] = 0;         // Heading/Track Angle 90.0
        INTR_data_0003_1[ 8] = 0;         // Heading/Track Angle Sgin(0:positive, 1: negative)

        INTR_data_0003_1[23] = 0;         // Closure Rate 0.5 knot
        INTR_data_0003_1[22] = 0;         // Closure Rate 1
        INTR_data_0003_1[21] = 0;         // Closure Rate 2
        INTR_data_0003_1[20] = 0;         // Closure Rate 4
        INTR_data_0003_1[19] = 0;         // Closure Rate 8
        INTR_data_0003_1[18] = 0;         // Closure Rate 16
        INTR_data_0003_1[17] = 0;         // Closure Rate 32
        INTR_data_0003_1[16] = 0;         // Closure Rate 64

        INTR_data_0003_1[31] = 0;         // Closure Rate 128
        INTR_data_0003_1[30] = 0;         // Closure Rate 256                                        
        INTR_data_0003_1[29] = 0;         // Closure Rate 512
        INTR_data_0003_1[28] = 0;         // Closure Rate 1024
        INTR_data_0003_1[27] = 0;         // Closure Rate Sign         
        INTR_data_0003_1[26] = 0;         // Closure Rate Status 
        INTR_data_0003_1[25] = 1;         // Data Type Continuation Bit              
        INTR_data_0003_1[24] = 0;         // Parity                                  (odd)

        // INTRUDER_DATA.heading
        float intr_heading = (float)INTRUDER_DATA.heading / 100.0 ; 

        // Filtered EO-based estimated heading
        intr_heading = vel_dir_lpf*R2D;
        if (vel_dir_lpf<0){
            intr_heading = 360.0 - vel_dir_lpf*R2D;
        }

        if (intr_heading > 180.0){
            // intr_heading = 360.0 - intr_heading;
            intr_heading = intr_heading -180.0;
            INTR_data_0003_1[8] = 1; // 1; 
            if (intr_heading>90.0)   { intr_heading = intr_heading - 90.0;     INTR_data_0003_1[ 9] = 1; }
            if (intr_heading>45.0)   { intr_heading = intr_heading - 45.0;     INTR_data_0003_1[10] = 1; }
            if (intr_heading>22.5)   { intr_heading = intr_heading - 22.5;     INTR_data_0003_1[11] = 1; }
            if (intr_heading>11.25)  { intr_heading = intr_heading - 11.25;    INTR_data_0003_1[12] = 1; }
            if (intr_heading>5.625)  { intr_heading = intr_heading - 5.625;    INTR_data_0003_1[13] = 1; }
            if (intr_heading>2.8125) { intr_heading = intr_heading - 2.8125;   INTR_data_0003_1[14] = 1; }
            if (intr_heading>1.40625){ intr_heading = intr_heading - 1.40625;  INTR_data_0003_1[15] = 1; }
        }
        else{
            INTR_data_0003_1[8] = 0; 
            if (intr_heading>90.0)   { intr_heading = intr_heading - 90.0;     INTR_data_0003_1[ 9] = 1; }
            if (intr_heading>45.0)   { intr_heading = intr_heading - 45.0;     INTR_data_0003_1[10] = 1; }
            if (intr_heading>22.5)   { intr_heading = intr_heading - 22.5;     INTR_data_0003_1[11] = 1; }
            if (intr_heading>11.25)  { intr_heading = intr_heading - 11.25;    INTR_data_0003_1[12] = 1; }
            if (intr_heading>5.625)  { intr_heading = intr_heading - 5.625;    INTR_data_0003_1[13] = 1; }
            if (intr_heading>2.8125) { intr_heading = intr_heading - 2.8125;   INTR_data_0003_1[14] = 1; }
            if (intr_heading>1.40625){ intr_heading = intr_heading - 1.40625;  INTR_data_0003_1[15] = 1; }
        }

        check_odd (INTR_data_0003_1, INTR_data_0003_1[24]);

        uint INTR_Data_0003_1_tempt=0b0;
        PACKET(INTR_data_0003_1, INTR_Data_0003_1_tempt);
        TX_struct.INTR_TYPE_0003_1 = htonl(INTR_Data_0003_1_tempt);

/// TX_struct.INTR_TYPE_0004 ///// ATTACHMENT 20M  -----------------------------------From here, modify 
        int INTR_data_0004[32];
    /// BIT             = Value      Function                               Coding
        INTR_data_0004[ 0] = 1;         // Label 1st Digit(MSB)               3:  1
        INTR_data_0004[ 1] = 1;         // Label 1st Digit(LSB)                   1
        INTR_data_0004[ 2] = 1;         // Label 2nd Digit(MSB)               6:  1
        INTR_data_0004[ 3] = 1;         // Label 2nd Digit                        1
        INTR_data_0004[ 4] = 0;         // Label 2nd Digit(LSB)                   0
        INTR_data_0004[ 5] = 1;         // Label 3rd Digit(MSB)               6:  1
        INTR_data_0004[ 6] = 1;         // Label 3rd Digit                        1
        INTR_data_0004[ 7] = 0;         // Label 3rd Digit(LSB)                   0

        INTR_data_0004[15] = 0;         // Data Type (LSB)                    4:  0 
        INTR_data_0004[14] = 0;         // Data Type                              0
        INTR_data_0004[13] = 1;         // Data Type                              1
        INTR_data_0004[12] = 0;         // Data Type (MSB)                        0
        INTR_data_0004[11] = 0;         // AIRB Queality level LSB
        INTR_data_0004[10] = 1;         // AIRB Queality level MSB
        INTR_data_0004[ 9] = 0;         // SURF Quality Level LSB
        INTR_data_0004[ 8] = 0;         // SURF Quality Level MSB

        INTR_data_0004[23] = 0;         // Interval Management Quality Level LSB
        INTR_data_0004[22] = 0;         // Interval Management Quality Level MSB
        INTR_data_0004[21] = 0;         // ITP Quality Level LSB
        INTR_data_0004[20] = 1;         // ITP Quality Level MSB
        INTR_data_0004[19] = 0;         // TSAA Quality Level LSB
        INTR_data_0004[18] = 1;         // TSAA Quality Level MSB
        INTR_data_0004[17] = 0;         // Reserved for future application's quality level
        INTR_data_0004[16] = 0;         // Reserved for future application's quality level

        INTR_data_0004[31] = 0;         // VSA Quality Level LSB
        INTR_data_0004[30] = 1;         // VSA Quality Level MSB
        INTR_data_0004[29] = 0;         // Reserved for future application's quality level           
        INTR_data_0004[28] = 1;         // Reserved for future application's quality level
        INTR_data_0004[27] = 0;         // Reserved for future application's quality level
        INTR_data_0004[26] = 0;         // Reserved for future application's quality level    
        INTR_data_0004[25] = 0;         // Data Type Continuation Bit              
        INTR_data_0004[24] = 0;         // Parity                                  (odd)
        check_odd (INTR_data_0004, INTR_data_0004[24]);

        uint INTR_Data_0004_tempt=0b0;
        PACKET(INTR_data_0004, INTR_Data_0004_tempt);
        // TX_struct.INTR_TYPE_0004_0 = htonl(INTR_Data_0004_tempt);

/// TX_struct.INTR_TYPE_0004_1 ///// ATTACHMENT 20M  -----------------------------------From here, modify 
        int INTR_data_0004_1[32];
    /// BIT             = Value      Function                               Coding
        INTR_data_0004_1[ 0] = 1;         // Label 1st Digit(MSB)               3:  1
        INTR_data_0004_1[ 1] = 1;         // Label 1st Digit(LSB)                   1
        INTR_data_0004_1[ 2] = 1;         // Label 2nd Digit(MSB)               6:  1
        INTR_data_0004_1[ 3] = 1;         // Label 2nd Digit                        1
        INTR_data_0004_1[ 4] = 0;         // Label 2nd Digit(LSB)                   0
        INTR_data_0004_1[ 5] = 1;         // Label 3rd Digit(MSB)               6:  1
        INTR_data_0004_1[ 6] = 1;         // Label 3rd Digit                        1
        INTR_data_0004_1[ 7] = 0;         // Label 3rd Digit(LSB)                   0

        INTR_data_0004_1[15] = 0;         // Reserved for future application's quality level 
        INTR_data_0004_1[14] = 0;         // Reserved for future application's quality level 
        INTR_data_0004_1[13] = 1;         // Reserved for future application's quality level 
        INTR_data_0004_1[12] = 0;         // Reserved for future application's quality level
        INTR_data_0004_1[11] = 0;         // Reserved for future application's quality level 
        INTR_data_0004_1[10] = 1;         // Reserved for future application's quality level 
        INTR_data_0004_1[ 9] = 0;         // Reserved for future application's quality level 
        INTR_data_0004_1[ 8] = 0;         // Reserved for future application's quality level 

        INTR_data_0004_1[23] = 0;         // Reserved for future application's quality level 
        INTR_data_0004_1[22] = 0;         // Reserved for future application's quality level 
        INTR_data_0004_1[21] = 0;         // Reserved for future application's quality level 
        INTR_data_0004_1[20] = 1;         // Reserved for future application's quality level 
        INTR_data_0004_1[19] = 0;         // Reserved for future application's quality level 
        INTR_data_0004_1[18] = 1;         // Reserved for future application's quality level 
        INTR_data_0004_1[17] = 0;         // Reserved for future application's quality level
        INTR_data_0004_1[16] = 0;         // Reserved for future application's quality level

        INTR_data_0004_1[31] = 0;         // Reserved for future application's quality level
        INTR_data_0004_1[30] = 1;         // Reserved for future application's quality level
        INTR_data_0004_1[29] = 0;         // Reserved for future application's quality level           
        INTR_data_0004_1[28] = 1;         // Reserved for future application's quality level
        INTR_data_0004_1[27] = 0;         // Reserved for future application's quality level
        INTR_data_0004_1[26] = 0;         // Reserved for future application's quality level    
        INTR_data_0004_1[25] = 1;         // Data Type Continuation Bit              
        INTR_data_0004_1[24] = 0;         // Parity                                  (odd)
        check_odd (INTR_data_0004_1, INTR_data_0004_1[24]);

        uint INTR_Data_0004_1_tempt=0b0;
        PACKET(INTR_data_0004_1, INTR_Data_0004_1_tempt);
        // TX_struct.INTR_TYPE_0004_1 = htonl(INTR_Data_0004_1_tempt);


/// TX_struct.TYPE_0005 ///// ATTACHMENT 20N  -----------------------------------From here, modify 
        int INTR_data_0005[32];
    /// BIT           = Value      Function                               Coding
        INTR_data_0005[ 0] = 1;         // Label 1st Digit(MSB)               3:  1
        INTR_data_0005[ 1] = 1;         // Label 1st Digit(LSB)                   1
        INTR_data_0005[ 2] = 1;         // Label 2nd Digit(MSB)               6:  1
        INTR_data_0005[ 3] = 1;         // Label 2nd Digit                        1
        INTR_data_0005[ 4] = 0;         // Label 2nd Digit(LSB)                   0
        INTR_data_0005[ 5] = 1;         // Label 3rd Digit(MSB)               6:  1
        INTR_data_0005[ 6] = 1;         // Label 3rd Digit                        1
        INTR_data_0005[ 7] = 0;         // Label 3rd Digit(LSB)                   0

        INTR_data_0005[15] = 1;         // Data Type (LSB)                    5:  1 
        INTR_data_0005[14] = 0;         // Data Type                              0
        INTR_data_0005[13] = 1;         // Data Type                              1
        INTR_data_0005[12] = 0;         // Data Type (MSB)                        0
        INTR_data_0005[11] = 0;         // Position Offset Applied
        INTR_data_0005[10] = 1;         // Width Code
        INTR_data_0005[ 9] = 0;         // Length Code (LSB)
        INTR_data_0005[ 8] = 0;         // Length Code 

        INTR_data_0005[23] = 0;         // Length Code (MSB)
        INTR_data_0005[22] = 0;         // Length/Width Status
        INTR_data_0005[21] = 0;         // PAD
        INTR_data_0005[20] = 1;         // PAD
        INTR_data_0005[19] = 0;         // PAD
        INTR_data_0005[18] = 1;         // PAD
        INTR_data_0005[17] = 0;         // PAD
        INTR_data_0005[16] = 0;         // PAD

        INTR_data_0005[31] = 0;         // PAD
        INTR_data_0005[30] = 1;         // PAD
        INTR_data_0005[29] = 0;         // PAD           
        INTR_data_0005[28] = 1;         // PAD
        INTR_data_0005[27] = 0;         // PAD
        INTR_data_0005[26] = 0;         // PAD    
        INTR_data_0005[25] = 1;         // Data Type COntinuation Bit              
        INTR_data_0005[24] = 0;         // Parity                                  (odd)
        check_odd (INTR_data_0005, INTR_data_0005[24]);

        uint INTR_Data_0005_tempt=0b0;
        PACKET(INTR_data_0005, INTR_Data_0005_tempt);
        // TX_struct.INTR_TYPE_0005_0 = htonl(INTR_Data_0005_tempt);


/// TX_struct.TYPE_0007 ///// ATTACHMENT 20P  -----------------------------------From here, modify 
        int INTR_data_0007[32];
    /// BIT           = Value      Function                               Coding
        INTR_data_0007[ 0] = 1;         // Label 1st Digit(MSB)               3:  1
        INTR_data_0007[ 1] = 1;         // Label 1st Digit(LSB)                   1
        INTR_data_0007[ 2] = 1;         // Label 2nd Digit(MSB)               6:  1
        INTR_data_0007[ 3] = 1;         // Label 2nd Digit                        1
        INTR_data_0007[ 4] = 0;         // Label 2nd Digit(LSB)                   0
        INTR_data_0007[ 5] = 1;         // Label 3rd Digit(MSB)               6:  1
        INTR_data_0007[ 6] = 1;         // Label 3rd Digit                        1
        INTR_data_0007[ 7] = 0;         // Label 3rd Digit(LSB)                   0

        INTR_data_0007[15] = 1;         // Data Type (LSB)                    7:  1 
        INTR_data_0007[14] = 1;         // Data Type                              1
        INTR_data_0007[13] = 1;         // Data Type                              1
        INTR_data_0007[12] = 0;         // Data Type (MSB)                        0
        INTR_data_0007[11] = 0;         // Seconds 1 LSB
        INTR_data_0007[10] = 1;         // Seconds 2
        INTR_data_0007[ 9] = 0;         // Seconds 4
        INTR_data_0007[ 8] = 0;         // Seconds 8 

        INTR_data_0007[23] = 0;         // Seconds 16
        INTR_data_0007[22] = 0;         // Seconds 32 MSB
        INTR_data_0007[21] = 0;         // Minutes 1 LSB
        INTR_data_0007[20] = 1;         // Minutes 2
        INTR_data_0007[19] = 0;         // Minutes 4
        INTR_data_0007[18] = 1;         // Minutes 8
        INTR_data_0007[17] = 0;         // Minutes 16
        INTR_data_0007[16] = 0;         // Minutes 32 MSB

        INTR_data_0007[31] = 0;         // Hours 1 LSB
        INTR_data_0007[30] = 1;         // Hour  2
        INTR_data_0007[29] = 0;         // Hours 4        
        INTR_data_0007[28] = 1;         // Hours 8
        INTR_data_0007[27] = 0;         // Hours 16 MSB
        INTR_data_0007[26] = 0;         // PAD    
        INTR_data_0007[25] = 0;         // Data Type Continuation Bit              0    
        INTR_data_0007[24] = 0;         // Parity                                  (odd)
        check_odd (INTR_data_0007, INTR_data_0007[24]);

        uint INTR_Data_0007_tempt=0b0;
        PACKET(INTR_data_0007, INTR_Data_0007_tempt);
        // TX_struct.INTR_TYPE_0007_0 = htonl(INTR_Data_0007_tempt);

/// TX_struct.INTR_TYPE_0007_1 ///// ATTACHMENT 20P  -----------------------------------From here, modify 
        int INTR_data_0007_1[32];
    /// BIT             = Value      Function                               Coding
        INTR_data_0007_1[ 0] = 1;         // Label 1st Digit(MSB)               3:  1
        INTR_data_0007_1[ 1] = 1;         // Label 1st Digit(LSB)                   1
        INTR_data_0007_1[ 2] = 1;         // Label 2nd Digit(MSB)               6:  1
        INTR_data_0007_1[ 3] = 1;         // Label 2nd Digit                        1
        INTR_data_0007_1[ 4] = 0;         // Label 2nd Digit(LSB)                   0
        INTR_data_0007_1[ 5] = 1;         // Label 3rd Digit(MSB)               6:  1
        INTR_data_0007_1[ 6] = 1;         // Label 3rd Digit                        1
        INTR_data_0007_1[ 7] = 0;         // Label 3rd Digit(LSB)                   0

        INTR_data_0007_1[15] = 1;         // Pad
        INTR_data_0007_1[14] = 1;         // Pad
        INTR_data_0007_1[13] = 1;         // Pad
        INTR_data_0007_1[12] = 0;         // Pad
        INTR_data_0007_1[11] = 0;         // Pad
        INTR_data_0007_1[10] = 1;         // Pad
        INTR_data_0007_1[ 9] = 0;         // Pad
        INTR_data_0007_1[ 8] = 0;         // Seconds 0.00011207 LSB

        INTR_data_0007_1[23] = 0;         // Seconds 0.000244141
        INTR_data_0007_1[22] = 0;         // Seconds 0.000488281
        INTR_data_0007_1[21] = 0;         // Seconds 0.000976563
        INTR_data_0007_1[20] = 1;         // Seconds 0.001953125
        INTR_data_0007_1[19] = 0;         // Seconds 0.00390625
        INTR_data_0007_1[18] = 1;         // Seconds 0.0078125
        INTR_data_0007_1[17] = 0;         // Seconds 0.015625
        INTR_data_0007_1[16] = 0;         // Seconds 0.03125

        INTR_data_0007_1[31] = 0;         // Seconds 0.0625
        INTR_data_0007_1[30] = 1;         // Seconds 0.125
        INTR_data_0007_1[29] = 0;         // Seconds 0.25  
        INTR_data_0007_1[28] = 1;         // Seconds 0.5          MSB
        INTR_data_0007_1[27] = 0;         // Time of Applicability Invalidity
        INTR_data_0007_1[26] = 0;         // PAD    
        INTR_data_0007_1[25] = 1;         // Data Type Continuation Bit              1  
        INTR_data_0007_1[24] = 0;         // Parity                                  (odd)
        check_odd (INTR_data_0007_1, INTR_data_0007_1[24]);       // Parity                                  (odd)

        uint INTR_Data_0007_1_tempt=0b0;
        PACKET(INTR_data_0007_1, INTR_Data_0007_1_tempt);
        // TX_struct.INTR_TYPE_0007_1 = htonl(INTR_Data_0007_1_tempt);
#endif

///// TX_struct.INTR_Label367 ///// ATTACHMENT 20D
        int label367_END[32];
        //0b 11110111 00010010 00000000 00000011
    /// BIT          = Value    Function                            Coding
        label367_END[ 0] = 1;       // Label 1st Digit(MSB)             3:  1
        label367_END[ 1] = 1;       // Label 1st Digit(LSB)                 1
        label367_END[ 2] = 1;       // Label 2nd Digit(MSB)             6:  1
        label367_END[ 3] = 1;       // Label 2nd Digit                      1
        label367_END[ 4] = 0;       // Label 2nd Digit(LSB)                 0
        label367_END[ 5] = 1;       // Label 3rd Digit(MSB)             7:  1
        label367_END[ 6] = 1;       // Label 3rd Digit                      1
        label367_END[ 7] = 1;       // Label 3rd Digit                      1

        label367_END[15] = 0;       // Number of Words in DRIF-1            
        label367_END[14] = 1;       // Number of Words in DRIF-2
        label367_END[13] = 0;       // Number of Words in DRIF-4
        label367_END[12] = 0;       // Number of Words in DRIF-8
        label367_END[11] = 1;       // Number of Words in DRIF-16
        label367_END[10] = 0;       // Number of Words in DRIF-32
        label367_END[ 9] = 0;       // Number of Words in DRIF-64
        label367_END[ 8] = 0;       // Number of Words in DRIF-128

        label367_END[23] = 0;       // Number of Words in DRIF-256
        label367_END[22] = 0;       // Number of Words in DRIF-512
        label367_END[21] = 0;       // Number of Words in DRIF-1024
        label367_END[20] = 0;       // Number of Words in DRIF-2048
        label367_END[19] = 0;       // Pad                                  0
        label367_END[18] = 0;       // Pad                                  0
        label367_END[17] = 0;       // Pad                                  0
        label367_END[16] = 0;       // Pad                                  0

        label367_END[31] = 0;       // ISO #5 Character ETX(0/3) (LSB)      1
        label367_END[30] = 1;       // ISO #5 Character ETX(0/3)            1
        label367_END[29] = 0;       // ISO #5 Character ETX(0/3)            0
        label367_END[28] = 0;       // ISO #5 Character ETX(0/3)            0
        label367_END[27] = 0;       // ISO #5 Character ETX(0/3)            0
        label367_END[26] = 0;       // ISO #5 Character ETX(0/3)            0
        label367_END[25] = 0;       // ISO #5 Character ETX(0/3) (MSB)      0
        label367_END[24] = 0;       // Parity                               (odd)
        check_odd (label367_END, label367_END[24]);

        uint label367_END_temp=0b0;
        PACKET(label367_END, label367_END_temp);
        TX_struct.Label367_END = htonl(label367_END_temp);

        ///////////////////////////////// Print 
//         system("clear");
//         cout << endl;
//         cout << "UDP package for CPDS" << endl;
//         printf("label367                %x\n", label_temp);
//         printf(" DTIF_header            %x\n", DTIF_header_temp);
//         printf("  DTIF_Packet_header    %x\n", DTIF_packet_header_temp);
//         printf("  OWN AIRCRAFT DATA       \n");
//         printf("    TYPE_0000_0         %x\n", Data_0000_0_tempt);
//         printf("    TYPE_0000_1         %x\n", Data_0000_1_tempt);
//         printf("    TYPE_0000_2         %x\n", Data_0000_2_tempt);
//         printf("    TYPE_0001_0         %x\n", Data_0001_tempt);
//         printf("    TYPE_0001_1         %x\n", Data_0001_1_tempt);
//         printf("    TYPE_0002_0         %x\n", Data_0002_tempt);
//         printf("    TYPE_0002_1         %x\n", Data_0002_1_tempt);
//         printf("    TYPE_0002_2         %x\n", Data_0002_2_tempt);
//         printf("    TYPE_0003_0         %x\n", Data_0003_tempt);
//         printf("    TYPE_0003_1         %x\n", Data_0003_1_tempt);
//         printf("    TYPE_0004_0         %x\n", Data_0004_tempt);
//         printf("    TYPE_0004_1         %x\n", Data_0004_1_tempt);
//         printf("    TYPE_0005_0         %x\n", Data_0005_tempt);
//         printf("    TYPE_0007_0         %x\n", Data_0007_tempt);
//         printf("    TYPE_0007_1         %x\n", Data_0007_1_tempt);

// #ifdef INTRUDER_ON
//         printf("  DTIF_Packet_header    %x\n", INTR_DTIF_packet_header_temp);
//         printf("  INTRUDER DATA           \n");
//         printf("    TYPE_0000_0         %x\n", INTR_Data_0000_0_tempt);
//         printf("    TYPE_0000_1         %x\n", INTR_Data_0000_1_tempt);
//         printf("    TYPE_0000_2         %x\n", INTR_Data_0000_2_tempt);
//         printf("    TYPE_0001_0         %x\n", INTR_Data_0001_tempt);
//         printf("    TYPE_0001_1         %x\n", INTR_Data_0001_1_tempt);
//         printf("    TYPE_0002_0         %x\n", INTR_Data_0002_tempt);
//         printf("    TYPE_0002_1         %x\n", INTR_Data_0002_1_tempt);
//         printf("    TYPE_0002_2         %x\n", INTR_Data_0002_2_tempt);
//         printf("    TYPE_0003_0         %x\n", INTR_Data_0003_tempt);
//         printf("    TYPE_0003_1         %x\n", INTR_Data_0003_1_tempt);
//         printf("    TYPE_0004_0         %x\n", INTR_Data_0004_tempt);
//         printf("    TYPE_0004_1         %x\n", INTR_Data_0004_1_tempt);
//         printf("    TYPE_0005_0         %x\n", INTR_Data_0005_tempt);
//         printf("    TYPE_0007_0         %x\n", INTR_Data_0007_tempt);
//         printf("    TYPE_0007_1         %x\n", INTR_Data_0007_1_tempt);
// #endif 
//         printf("label367_END            %x\n", label367_END_temp);
//         cout << endl;

        int sent;
        sent = sendto(StrUDP.Socket, (char*)&TX_struct, sizeof(TX_struct), 0, (struct sockaddr *)(&StrUDP.ServerAddr), sizeof(StrUDP.ServerAddr));

///////////////////////
        // TX_test.a0 = htonl(0b11110111000100101000000000000010); //0xF7128002
        // // ATT 20F
        // TX_test.b0 = htonl(0b11110110100000011000000000111111); //0xF681803F
        // //ATT 20G
        // TX_test.c0 = htonl(0b11110110100000010000011110100000); //0xF68107A0

        // // TYPE 0
        // TX_test.d0 = htonl(0b11110110111100000110100010000001); //0xF6F03481    3char
        // TX_test.a1 = htonl(0b11110110010010000001000000000011); //0xF6481003    3char
        // TX_test.b1 = htonl(0b11110110001100110001100001000110); //0xF6331846    2char
        // // TYPE 1
        // TX_test.c1 = htonl(0b11110110000000010000000010100000); //0xF60100A0
        // TX_test.d1 = htonl(0b11110110000010000000000011000000); //0xF60800C0 -- altitude
        // // TYPE 2
        // TX_test.a2 = htonl(0b11110110001100100011000110000011); //0xF6323183
        // TX_test.b2 = htonl(0b11110110010100010000100100100001); //0xF6510921
        // TX_test.c2 = htonl(0b11110110100010100000010101100000); //0xF68A0560
        // // TYPE 3
        // TX_test.d2 = htonl(0b11110110000000110000000010110000); //0xF60300B0
        // TX_test.a3 = htonl(0b11110110110000000000000001000000); //0xF6C00040
        // // TYPE 6
        // TX_test.b3 = htonl(0b11110110110001100000010110000000); //0xF6C60580
        // TX_test.c3 = htonl(0b11110110000000000000000001000000); //0xF6000040
        // // TYPE 7
        // TX_test.d3 = htonl(0b11110110100001111010000100010001); //0xF687A111
        // TX_test.a4 = htonl(0b11110110000000001111100011000011); //0xF600F8C3

        // TX_test.b4 = htonl(0b11110111000100100000000000000011); //0xF7120003

        // int aaaa;
        // aaaa = sendto(StrUDP.Socket, (char*)&TX_test, sizeof(TX_test), 0, (struct sockaddr *)(&StrUDP.ServerAddr), sizeof(StrUDP.ServerAddr));


	    // loop rate [Hz]
        loop_rate.sleep(); //1Hz

	    // loop sampling, ros
	    spinOnce();
	}
    // Close UDP socket
    close(StrUDP.Socket);

	// for debugging
	printf("\nTerminate: UDP_ARINC_735B_node\n");

	return 0;
}

static void QuaterniontoEuler(float orientation_x,float orientation_y,float orientation_z,float orientation_w, float& roll, float& pitch, float& yaw){
    // roll (x-axis rotation)
    float t0    = +2.0       * (orientation_w * orientation_x + orientation_y * orientation_z);
    float t1    = +1.0 - 2.0 * (orientation_x * orientation_x + orientation_y * orientation_y);
    roll        = std::atan2(t0,t1) * R2D;

    // pitch (y-axis rotation)
    float t2    = +2.0       * (orientation_w * orientation_y - orientation_z * orientation_x);
    t2          = t2 >  1.0 ?  1.0 : t2;
    t2          = t2 < -1.0 ? -1.0 : t2;
    pitch       = -std::asin(t2) * R2D;

    // yaw (z-axis rotation)
    float t3    = +2.0       * (orientation_w*orientation_z + orientation_x*orientation_y);
    float t4    = +1.0 - 2.0 * (orientation_y*orientation_y + orientation_z*orientation_z);
    yaw         = std::atan2(t3,t4) * R2D;
}

void detectionCallback(const std_msgs::Float32MultiArray& array)
{
    detection_result[0] = array.data[0]; // flag
    detection_result[1] = array.data[1]; // size [width]
    detection_result[2] = array.data[2]; // size y [height]
    detection_result[3] = array.data[3]; // x from center
    detection_result[4] = array.data[4]; // y from center

    tar_data.flag_detect = detection_result[0];
    tar_data.size[0] = detection_result[1];
    tar_data.size[1] = detection_result[2];
    tar_data.impos[0] = detection_result[3];
    tar_data.impos[1] = detection_result[4];
}


void camsel_Callback(const std_msgs::Float32MultiArray& array)
{
    camsel[0] = array.data[0]; 
    camsel[1] = array.data[1]; 

    if (camsel[0]==1){
        CAM_SELECTION = 1;
    }
    else if (camsel[0]==2 || camsel[0]==3){
        CAM_SELECTION = 0;
    }
    else if (camsel[0]==4){
        CAM_SELECTION = 2;
    }
    else{
        CAM_SELECTION = 0;
    }
}

void arrayCallback(const std_msgs::UInt32MultiArray& array)
{
    UDP_tx_data[ 0]  = array.data[0];
    UDP_tx_data[ 1]  = array.data[1];
    UDP_tx_data[ 2]  = array.data[2];
    UDP_tx_data[ 3]  = array.data[3];
    
    UDP_tx_data[ 4]  = array.data[4];
    UDP_tx_data[ 5]  = array.data[5];
    UDP_tx_data[ 6]  = array.data[6];
    UDP_tx_data[ 7]  = array.data[7];

    UDP_tx_data[ 8]  = array.data[8];
    UDP_tx_data[ 9]  = array.data[9];
    UDP_tx_data[10]  = array.data[10];
    UDP_tx_data[11]  = array.data[11];

    UDP_tx_data[12]  = array.data[12];
    UDP_tx_data[13]  = array.data[13];
    UDP_tx_data[14]  = array.data[14];
    UDP_tx_data[15]  = array.data[15];

    UDP_tx_data[16]  = array.data[16];
    UDP_tx_data[17]  = array.data[17];
	return;
} 


void UAV0_global_position(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
    UAV0_gposition[0] = msg -> latitude;
    UAV0_gposition[1] = msg -> longitude;
    UAV0_gposition[2] = msg -> altitude;
}

void UAV1_global_position(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
    UAV1_gposition[0] = msg -> latitude;
    UAV1_gposition[1] = msg -> longitude;
    UAV1_gposition[2] = msg -> altitude;
}

void UAV0_imu_callback(const sensor_msgs::Imu::ConstPtr& imu_msg){

    UAV0_linear_acc_local[0]  = imu_msg->linear_acceleration.x;
    UAV0_linear_acc_local[1]  = imu_msg->linear_acceleration.y;
    UAV0_linear_acc_local[2]  = imu_msg->linear_acceleration.z;

    UAV0_orientation_local[0] = imu_msg->orientation.x;
    UAV0_orientation_local[1] = imu_msg->orientation.y;
    UAV0_orientation_local[2] = imu_msg->orientation.z;
    UAV0_orientation_local[3] = imu_msg->orientation.w;

    UAV0_anglular_velocity_local[0] = imu_msg->angular_velocity.x;
    UAV0_anglular_velocity_local[1] = imu_msg->angular_velocity.y;
    UAV0_anglular_velocity_local[2] = imu_msg->angular_velocity.z;

    QuaterniontoEuler(UAV0_orientation_local[0], UAV0_orientation_local[1], UAV0_orientation_local[2], UAV0_orientation_local[3], UAV0_orientation_Euler_local[0], UAV0_orientation_Euler_local[1], UAV0_orientation_Euler_local[2]);
}

void UAV1_imu_callback(const sensor_msgs::Imu::ConstPtr& imu_msg){

    UAV1_linear_acc_local[0]  = imu_msg->linear_acceleration.x;
    UAV1_linear_acc_local[1]  = imu_msg->linear_acceleration.y;
    UAV1_linear_acc_local[2]  = imu_msg->linear_acceleration.z;

    UAV1_orientation_local[0] = imu_msg->orientation.x;
    UAV1_orientation_local[1] = imu_msg->orientation.y;
    UAV1_orientation_local[2] = imu_msg->orientation.z;
    UAV1_orientation_local[3] = imu_msg->orientation.w;

    UAV1_anglular_velocity_local[0] = imu_msg->angular_velocity.x;
    UAV1_anglular_velocity_local[1] = imu_msg->angular_velocity.y;
    UAV1_anglular_velocity_local[2] = imu_msg->angular_velocity.z;

    QuaterniontoEuler(UAV1_orientation_local[0], UAV1_orientation_local[1], UAV1_orientation_local[2], UAV1_orientation_local[3], UAV1_orientation_Euler_local[0], UAV1_orientation_Euler_local[1], UAV1_orientation_Euler_local[2]);
}

void TR_Data_Array_callback(const ads_b_read::Traffic_Report_Array::ConstPtr& msg_input)
{
    TR_Data_Array.Traffic_Reports.clear();

    for (int i = 0; i<msg_input->Traffic_Reports.size();i++){
        const ads_b_read::Traffic_Report& TR_Data_Sub = msg_input->Traffic_Reports[i];
        TR_Data_Array.Traffic_Reports.push_back(TR_Data_Sub);

        for (int k=0;k<9;k++){
            callsign_print[i][k] = TR_Data_Sub.callsign[k];
        }
        // cout << "ICAO_adress"<<i<<":"<<TR_Data_Sub.ICAO_address<< "     CallSign"<<i<<":"<< callsign_print[i] << endl ;
        // memset(&OWNSHIP_DATA,  0, sizeof(OWNSHIP_DATA));
        // memset(&INTRUDER_DATA, 0, sizeof(INTRUDER_DATA));

        if (TR_Data_Sub.ICAO_address == OWNSHIP_ICAO){
            memset(&OWNSHIP_DATA,  0, sizeof(OWNSHIP_DATA));

            OWNSHIP_DATA.Start_Flag         = TR_Data_Sub.Start_Flag;
            OWNSHIP_DATA.Length             = TR_Data_Sub.Length;
            OWNSHIP_DATA.Sequence           = TR_Data_Sub.Sequence;
            OWNSHIP_DATA.System_ID          = TR_Data_Sub.System_ID;
            OWNSHIP_DATA.Component          = TR_Data_Sub.Component;
            OWNSHIP_DATA.Message_ID         = TR_Data_Sub.Message_ID;
            OWNSHIP_DATA.ICAO_address       = TR_Data_Sub.ICAO_address;      // 4bytes
            OWNSHIP_DATA.lat                = TR_Data_Sub.lat;                  // 4bytes
            OWNSHIP_DATA.lon                = TR_Data_Sub.lon;                // 4bytes
            OWNSHIP_DATA.altitude           = TR_Data_Sub.altitude/1000.0;          // 4bytes
            OWNSHIP_DATA.heading            = TR_Data_Sub.heading;          // 2bytes
            OWNSHIP_DATA.hor_velocity       = TR_Data_Sub.hor_velocity;     // 2bytes
            OWNSHIP_DATA.ver_velocity       = TR_Data_Sub.ver_velocity;     // 2bytes
            OWNSHIP_DATA.validFlags         = TR_Data_Sub.validFlags;       // 2bytes
            OWNSHIP_DATA.squawk             = TR_Data_Sub.squawk;           // 2bytes
            OWNSHIP_DATA.altitude_type      = TR_Data_Sub.altitude_type;    // 1bytes
        
            // system("clear");
            cout << endl;
            cout << "OWNSHIP : " << OWNSHIP_DATA.ICAO_address <<",  " << "OWNSHIP : " << callsign_print[i] << endl;
            cout << "  - Lat : " << OWNSHIP_DATA.lat << "  Lon : " << OWNSHIP_DATA.lon << endl;
            cout << "  - Alt : " << OWNSHIP_DATA.altitude <<  endl;
            cout << "  - Heading : " << OWNSHIP_DATA.heading <<  endl;
            cout << "  - Vel_horizontal : " << OWNSHIP_DATA.hor_velocity <<  endl;
            cout << "  - Vel_Vertical   : " << OWNSHIP_DATA.ver_velocity <<  endl;

        }
        else{}
        if (TR_Data_Sub.ICAO_address ==  INTRUDER_ICAO){
            cout << "intruder" << endl;

            memset(&INTRUDER_DATA, 0, sizeof(INTRUDER_DATA));
            INTRUDER_DATA.Start_Flag         = TR_Data_Sub.Start_Flag;
            INTRUDER_DATA.Length             = TR_Data_Sub.Length;
            INTRUDER_DATA.Sequence           = TR_Data_Sub.Sequence;
            INTRUDER_DATA.System_ID          = TR_Data_Sub.System_ID;
            INTRUDER_DATA.Component          = TR_Data_Sub.Component;
            INTRUDER_DATA.Message_ID         = TR_Data_Sub.Message_ID;
            INTRUDER_DATA.ICAO_address       = TR_Data_Sub.ICAO_address;      // 4bytes
            INTRUDER_DATA.lat                = TR_Data_Sub.lat;               // 4bytes
            INTRUDER_DATA.lon                = TR_Data_Sub.lon;               // 4bytes
            INTRUDER_DATA.altitude           = TR_Data_Sub.altitude/1000.0;          // 4bytes
            INTRUDER_DATA.heading            = TR_Data_Sub.heading;          // 2bytes
            INTRUDER_DATA.hor_velocity       = TR_Data_Sub.hor_velocity;     // 2bytes
            INTRUDER_DATA.ver_velocity       = TR_Data_Sub.ver_velocity;     // 2bytes
            INTRUDER_DATA.validFlags         = TR_Data_Sub.validFlags;       // 2bytes
            INTRUDER_DATA.squawk             = TR_Data_Sub.squawk;           // 2bytes
            INTRUDER_DATA.altitude_type      = TR_Data_Sub.altitude_type;    // 1bytes

            cout << endl;
            cout << "INTRUDER : " << INTRUDER_DATA.ICAO_address <<",  " << "INTRUDERs : " << callsign_print[i] << endl;
            cout << "  - Lat : " << INTRUDER_DATA.lat << "  Lon : " << INTRUDER_DATA.lon << endl;
            cout << "  - Alt : " << INTRUDER_DATA.altitude <<  endl;
            cout << "  - Heading : " << INTRUDER_DATA.heading <<  endl;
            cout << "  - Vel_horizontal : " << INTRUDER_DATA.hor_velocity <<  endl;
            cout << "  - Vel_Vertical   : " << INTRUDER_DATA.ver_velocity <<  endl;
        }
        else {}
    }
    cout << "size :" << TR_Data_Array.Traffic_Reports.size() << endl;

}


char IntToChar(int aaa){
    char tempt_char;
    if (aaa ==1){
        tempt_char = '1';
    }
    else{
        tempt_char = '0';
    }
    return tempt_char;
}

void PACKET(int index[32], uint& sum_num){

    uint tempt_label[32];
    
    for (int i=0; i<32; i++){
        tempt_label[ i] = IntToBinary(index[i],31-i);
    }

    sum_num = 0b0;
    for (int i=31; i>-1;i--){
        sum_num = sum_num + tempt_label[i] ;
    }
}

void check_odd (int data[32], int& odd_data){
    int data_sum = 0;
    int odd_num  = 0;
    for (int i =0; i<32; i++)
        data_sum = data_sum + data[i];
    if(data_sum % 2 == 0)
        odd_num  = 1; 
    else
        odd_num  = 0;
    odd_data = odd_num;
}

uint IntToBinary(int num, int location){
    uint Binary_num;
    if (num ==0){
        Binary_num  = 0b00000000000000000000000000000000;
    }
    else if (num ==1){
        switch (location){
            case 0: Binary_num  = 0b1; break;
            case 1: Binary_num  = 0b10; break;
            case 2: Binary_num  = 0b100; break;
            case 3: Binary_num  = 0b1000; break;
            case 4: Binary_num  = 0b10000; break;
            case 5: Binary_num  = 0b100000; break;
            case 6: Binary_num  = 0b1000000; break;
            case 7: Binary_num  = 0b10000000; break;
            case 8: Binary_num  = 0b100000000; break;
            case 9: Binary_num  = 0b1000000000; break;
            case 10: Binary_num = 0b10000000000; break;
            case 11: Binary_num = 0b100000000000; break;
            case 12: Binary_num = 0b1000000000000; break;
            case 13: Binary_num = 0b10000000000000; break;
            case 14: Binary_num = 0b100000000000000; break;
            case 15: Binary_num = 0b1000000000000000; break;
            case 16: Binary_num = 0b10000000000000000; break;
            case 17: Binary_num = 0b100000000000000000; break;
            case 18: Binary_num = 0b1000000000000000000; break;
            case 19: Binary_num = 0b10000000000000000000; break;
            case 20: Binary_num = 0b100000000000000000000; break;
            case 21: Binary_num = 0b1000000000000000000000; break;
            case 22: Binary_num = 0b10000000000000000000000; break;
            case 23: Binary_num = 0b100000000000000000000000; break;
            case 24: Binary_num = 0b1000000000000000000000000; break;
            case 25: Binary_num = 0b10000000000000000000000000; break;
            case 26: Binary_num = 0b100000000000000000000000000; break;
            case 27: Binary_num = 0b1000000000000000000000000000; break;
            case 28: Binary_num = 0b10000000000000000000000000000; break;
            case 29: Binary_num = 0b100000000000000000000000000000; break;
            case 30: Binary_num = 0b1000000000000000000000000000000; break;
            case 31: Binary_num = 0b10000000000000000000000000000000; break;
            default: "[ERROR] This Binary is out of range"; break;
        }
    }
    else { 
        printf("[ERROR] It is not binary!!");
    }
    return Binary_num;
}

void rawfixcallback_Own(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
    bag_own_lat = msg ->latitude;
    bag_own_lon = msg ->longitude;
    bag_own_alt = msg ->altitude;
    // cout << "Own lat : " << bag_own_lat << endl;
    // cout << "alt_own : " << bag_own_alt;
}

void rawfixcallback_Int(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
    bag_int_lat = msg ->latitude;
    bag_int_lon = msg ->longitude;
    bag_int_alt = msg ->altitude;
    // cout << "alt_int : " << bag_int_alt;
    // cout << "Intr lat : " << bag_int_lat << endl;
}

void LatLontoECEF(float lat, float lon, float h, float *ECEFX_m, float *ECEFY_m, float *ECEFZ_m){
    float sinLat        = sin(lat/180.0*M_PI);
    float sinLon        = sin(lon/180.0*M_PI);
    float cosLat        = cos(lat/180.0*M_PI);
    float cosLon        = cos(lon/180.0*M_PI);

    float N             = WGS84_a_m/sqrt(1-WGS84_e*WGS84_e*sinLat*sinLat);
    *ECEFX_m            = (N+h)*cosLat*cosLon;
    *ECEFY_m            = (N+h)*cosLat*sinLon;
    *ECEFZ_m            = (N*(1-WGS84_e*WGS84_e)+h)*sinLat;
}
