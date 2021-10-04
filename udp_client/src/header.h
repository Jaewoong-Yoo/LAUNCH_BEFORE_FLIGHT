// essential header for ROS-OpenCV operation
#include <ros/ros.h>
#include "std_msgs/String.h"
#include "sensor_msgs/NavSatFix.h"
#include <sensor_msgs/Imu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "opencv2/opencv.hpp"

// for using standard messages, float 32 type
// communicate to image processing algorithm result
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/UInt32MultiArray.h>

// for using serial communication
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <arpa/inet.h>

// msg Structure
#include "struct_msg.h"
#include "Struct_comm.h"
#include "definition.h"

// ADS-B massage sets
#include <ads_b_read/Traffic_Report.h>
#include <ads_b_read/DataStream_Request.h>
#include <ads_b_read/Status.h>
#include <ads_b_read/Static.h>
#include <ads_b_read/Dynamic_Ownship.h> 
#include <ads_b_read/Traffic_Report_Array.h>
#include <sensor_msgs/NavSatFix.h>

#define std_lat             36.593133       // Hanseo Univ.
#define std_lon             126.295440      // Hanseo Univ.

#define OWNSHIP_ICAO  7463479
// #define INTRUDER_ICAO 7441001               // 7441001(1269)
// #define INTRUDER_ICAO 7441008               // 1270
#define INTRUDER_ICAO 7463475               // 1271

#define DF_UDP_BUFFER_SIZE  8192
#define DF_UDP_PORTNUM      31000 //31000						
#define DF_UDP_SERVER_ADDR  "192.168.1.100" //"192.168.0.104"				
struct timeval tv_timeo = { 0, 500000 };   // 0.5 seconds

#define D2R             3.141592/180.0
#define R2D             180.0/3.141592
#define mps2knot        1.943844 
#define mps2ftpmin      196.8504  
#define m2ft            3.28084

#define CAMFOV          60.0
#define ANG_SIDECAM45   45.0
#define ANG_SIDECAM90   90
#define CAMwidth        1920
#define CAMheight       1080

#define M_PI                3.14159265358979323846
#define WGS84_a_m           6378137.0
#define WGS84_e             0.08181919

struct struct_t_UDP
{
    int    Socket;
	struct sockaddr_in ServerAddr;
    char   TXBuffer[DF_UDP_BUFFER_SIZE];
    char   RXBuffer[DF_UDP_BUFFER_SIZE];
};

struct struct_udp_test
{
    // -------------
    // Packet Header
    // -------------
    // uint example = 0xF7128002F681803FF68107A0F6F03481F6481003F6331846F60100A0F60800C0F6323183F6510921F68A0560F60300B0F6C00040F6C60580F68A0560F60300B0F600F8C3F7120003;

    // uint32_t Label367; // 0xFE;
    // uint32_t Label366_Header;
    // uint32_t Label366_Packet_Header;
    // uint32_t System_ID;
    // uint32_t Component;
    // uint32_t Message_ID;

    uint32_t a0; // START
    uint32_t b0;
    uint32_t c0;
    uint32_t d0;

    uint32_t a1;
    uint32_t b1;
    uint32_t c1;
    uint32_t d1;

    uint32_t a2;
    uint32_t b2;
    uint32_t c2;
    uint32_t d2;

    uint32_t a3;
    uint32_t b3;
    uint32_t c3;
    uint32_t d3;

    uint32_t a4;
    uint32_t b4; // END
};
