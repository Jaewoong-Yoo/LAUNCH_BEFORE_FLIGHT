#include <ros/ros.h>
#include <serial/serial.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Empty.h>
#include <sensor_msgs/NavSatFix.h>

#define PORT1 	 	"/dev/ttyUSB0"
#define BAUDRATE 	57600
#define ROS_FREQ 	10
#define ROS_FREQ_HIGH   1000
#define DEBUG_MODE 	1
#define SEND_DYNAMIC 	0
#define SEND_STATIC 	0

#define MAX_PLANES      200



// Header
#pragma pack(1)
struct struct_t_Header
{
    // -------------
    // Packet Header
    // -------------
    uint8_t Start_Flag; // 0xFE;
    uint8_t Length;
    uint8_t Sequence;
    uint8_t System_ID;
    uint8_t Component;
    uint8_t Message_ID;
};
#pragma pack()

//
// Checksum
#pragma pack(1)
struct struct_t_Checksum
{
    // -------------
    // Packet Header
    // -------------
    uint8_t CKA; // low byte
    uint8_t CKB; // high byte
};
#pragma pack()


//
// DataStream Request
#pragma pack(1)
struct struct_t_DataStream_Request
{
    struct_t_Header header;

    // ----
    // Data
    // ----
    uint16_t req_message_rate;
    uint8_t  target_system;
    uint8_t  target_component;
    uint8_t  req_stream_id;
    uint8_t  start_stop;

    struct_t_Checksum checksum;
    uint8_t CRC_PASS;    // Pass the Checksum test
    static const uint8_t CRC_EXTRA = 148;
};
#pragma pack()


//
// Traffic Report
#pragma pack(1)
struct struct_t_Traffic_Report
{
    struct_t_Header header;

    // ----
    // Data
    // ----
    uint32_t ICAO_address;
    int32_t  lat;
    int32_t  lon;
    int32_t  altitude;
    uint16_t heading;
    uint16_t hor_velocity;
    int16_t  ver_velocity;
    uint16_t validFlags;
    uint16_t squawk;
    uint8_t  altitude_type;
    char     callsign[9];
    uint8_t  emitter_type;
    uint8_t  tslc;

    struct_t_Checksum checksum;
    uint8_t CRC_PASS;    // Pass the Checksum test
    static const uint8_t CRC_EXTRA = 184;
};
#pragma pack()


//
// Status
#pragma pack(1)
struct struct_t_Status
{
    struct_t_Header header;

    // ----
    // Data
    // ----
    uint8_t status;

    struct_t_Checksum checksum;
    uint8_t CRC_PASS;    // Pass the Checksum test
    static const uint8_t CRC_EXTRA = 85;
};
#pragma pack()


//
// Dynamic/Ownship
#pragma pack(1)
struct struct_t_Dynamic_Ownship
{
    struct_t_Header header;

    // ----
    // Data
    // ----
    uint32_t utcTime;
    int32_t  latitude;
    int32_t  longitude;
    int32_t  altPres;
    int32_t  altGNSS;
    uint32_t accHoriz;
    uint16_t accVert;
    uint16_t accVel;
    int16_t  velVert;
    int16_t  nsVog;
    int16_t  ewVog;
    uint16_t state;
    uint16_t squawk;
    uint8_t  fixType;
    uint8_t  numSats;
    uint8_t  emStatus;
    uint8_t  control;

    struct_t_Checksum checksum;
    uint8_t CRC_PASS;    // Pass the Checksum test
    static const uint8_t CRC_EXTRA = 7;
};
#pragma pack()


//
// Static
#pragma pack(1)
struct struct_t_Static
{
    struct_t_Header header;

    // ----
    // Data
    // ----
    uint8_t  ICAO[3];
    uint8_t  integrity;
    uint16_t stallSpeed;
    char     Callsign_Flt_Plan[8];
    uint8_t  capability;
    uint8_t  emitter;
    uint8_t  alwEncode;
    uint8_t  gpsLatOffs;
    uint8_t  gpsLonOffs;

    struct_t_Checksum checksum;
    static const uint8_t CRC_EXTRA = 126;
};
#pragma pack()
