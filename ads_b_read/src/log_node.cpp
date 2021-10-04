#include "header.h"

#include <stdio.h>
#include <boost/filesystem.hpp>

#include <ads_b_read/Traffic_Report.h>
#include <ads_b_read/Traffic_Report_Array.h>

using namespace std;

struct struct_t_Traffic_Report     StrTraffic_Report;

ads_b_read::Traffic_Report       TR_Ownship_Data;
ads_b_read::Traffic_Report       TR_Intruder_Data;

ros::Subscriber TR_Ownship_sub;
ros::Subscriber TR_Intruder_sub;

void TR_Ownship_callback(const ads_b_read::Traffic_Report& msg_input);
void TR_Intruder_callback(const ads_b_read::Traffic_Report& msg_input);

uint32_t count_ros = 0;
FILE* TR_Ownship_File;
char TR_Ownship_FolderPath[100];
char TR_Ownship_FilePath[100];
char TR_Ownship_String[1000];

FILE* TR_Intruder_File;
char TR_Intruder_FolderPath[100];
char TR_Intruder_FilePath[100];
char TR_Intruder_String[1000];

int main (int argc, char** argv){
    ros::init(argc, argv, "log_node");
    ros::NodeHandle nh;

    TR_Ownship_sub  = nh.subscribe("/ADS_B/TR_Ownship", 1000, TR_Ownship_callback);
    TR_Intruder_sub  = nh.subscribe("/ADS_B/TR_Intruder", 1000, TR_Intruder_callback);

    ros::Rate loop_rate(ROS_FREQ_HIGH); // 1000

    struct tm *date;
    const time_t t = time(NULL);
    date = localtime(&t);

    sprintf(TR_Ownship_FolderPath, "/home/usrg/Civil/ADS_B_txt_log");
    boost::filesystem::create_directories(TR_Ownship_FolderPath);

    sprintf(TR_Ownship_FilePath, "/home/usrg/Civil/ADS_B_txt_log/%d_%d_%d_%d_%d_%d_Ownship.txt",
            (date->tm_year+1900), (date->tm_mon+1), date->tm_mday, date->tm_hour, date->tm_min, date->tm_sec);
    cout << TR_Ownship_FilePath << endl;

    sprintf(TR_Intruder_FilePath, "/home/usrg/Civil/ADS_B_txt_log/%d_%d_%d_%d_%d_%d_Intruder.txt",
            (date->tm_year+1900), (date->tm_mon+1), date->tm_mday, date->tm_hour, date->tm_min, date->tm_sec);
    cout << TR_Intruder_FilePath << endl;

    TR_Ownship_File = fopen(TR_Ownship_FilePath, "w");
    fputs("[Ownship_Data]\n", TR_Ownship_File);
    fputs("timestamp, Start_Flag, Length, Sequence, System_ID, Component, Message_ID, ICAO_address, "
          "lat, lon, altitude, heading, hor_velocity, ver_velocity, validFlags, squawk, "
          "altitude_type, emitter_type, tslc, CKA, CKB, CRC_PASS, callsign\n", TR_Ownship_File);
    ROS_INFO_STREAM("Start Logging Ownship Data");

    TR_Intruder_File = fopen(TR_Intruder_FilePath, "w");
    fputs("[Intruder_Data]\n", TR_Intruder_File);
    fputs("timestamp, Start_Flag, Length, Sequence, System_ID, Component, Message_ID, ICAO_address, "
          "lat, lon, altitude, heading, hor_velocity, ver_velocity, validFlags, squawk, "
          "altitude_type, emitter_type, tslc, CKA, CKB, CRC_PASS, callsign\n", TR_Intruder_File);
    ROS_INFO_STREAM("Start Logging Intruder Data");



    while(ros::ok())
    {
        count_ros++;
        loop_rate.sleep();
        ros::spinOnce();
    }

    fclose(TR_Ownship_File);
    fclose(TR_Intruder_File);
}

void TR_Ownship_callback(const ads_b_read::Traffic_Report& msg_input)
{
    const ads_b_read::Traffic_Report& TR_Ownship_Data = msg_input;

    ROS_INFO_STREAM("Write [Ownship  Data: " << TR_Ownship_Data.ICAO_address << "] from /ADS_B/TR_Ownship  to /home/usrg-asus/Civil/ADS_B_txt_log/");
    sprintf(TR_Ownship_String, "%f, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, "
                               "%d, %d, %d, %d, %d, %c%c%c%c%c%c%c%c%c",
            TR_Ownship_Data.header.stamp.toSec(),
            TR_Ownship_Data.Start_Flag, TR_Ownship_Data.Length, TR_Ownship_Data.Sequence, TR_Ownship_Data.System_ID,
            TR_Ownship_Data.Component, TR_Ownship_Data.Message_ID, TR_Ownship_Data.ICAO_address, TR_Ownship_Data.lat,
            TR_Ownship_Data.lon, TR_Ownship_Data.altitude, TR_Ownship_Data.heading, TR_Ownship_Data.hor_velocity,
            TR_Ownship_Data.ver_velocity, TR_Ownship_Data.validFlags, TR_Ownship_Data.squawk, TR_Ownship_Data.altitude_type,
            TR_Ownship_Data.emitter_type, TR_Ownship_Data.tslc, TR_Ownship_Data.CKA, TR_Ownship_Data.CKB,
            TR_Ownship_Data.CRC_PASS,
            TR_Ownship_Data.callsign[0], TR_Ownship_Data.callsign[1], TR_Ownship_Data.callsign[2], TR_Ownship_Data.callsign[3],
            TR_Ownship_Data.callsign[4], TR_Ownship_Data.callsign[5], TR_Ownship_Data.callsign[6], TR_Ownship_Data.callsign[7],
            TR_Ownship_Data.callsign[8]);
    fputs(TR_Ownship_String, TR_Ownship_File);
    fputs("\n", TR_Ownship_File);
}

void TR_Intruder_callback(const ads_b_read::Traffic_Report& msg_input)
{
    const ads_b_read::Traffic_Report& TR_Intruder_Data = msg_input;

    ROS_INFO_STREAM("Write [Intruder Data: " << TR_Intruder_Data.ICAO_address << "] from /ADS_B/TR_Intruder to /home/usrg/ADS_B_txt_log/");
    sprintf(TR_Intruder_String, "%f, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, "
                               "%d, %d, %d, %d, %d, %c%c%c%c%c%c%c%c%c",
            TR_Intruder_Data.header.stamp.toSec(),
            TR_Intruder_Data.Start_Flag, TR_Intruder_Data.Length, TR_Intruder_Data.Sequence, TR_Intruder_Data.System_ID,
            TR_Intruder_Data.Component, TR_Intruder_Data.Message_ID, TR_Intruder_Data.ICAO_address, TR_Intruder_Data.lat,
            TR_Intruder_Data.lon, TR_Intruder_Data.altitude, TR_Intruder_Data.heading, TR_Intruder_Data.hor_velocity,
            TR_Intruder_Data.ver_velocity, TR_Intruder_Data.validFlags, TR_Intruder_Data.squawk, TR_Intruder_Data.altitude_type,
            TR_Intruder_Data.emitter_type, TR_Intruder_Data.tslc, TR_Intruder_Data.CKA, TR_Intruder_Data.CKB,
            TR_Intruder_Data.CRC_PASS,
            TR_Intruder_Data.callsign[0], TR_Intruder_Data.callsign[1], TR_Intruder_Data.callsign[2], TR_Intruder_Data.callsign[3],
            TR_Intruder_Data.callsign[4], TR_Intruder_Data.callsign[5], TR_Intruder_Data.callsign[6], TR_Intruder_Data.callsign[7],
            TR_Intruder_Data.callsign[8]);
    fputs(TR_Intruder_String, TR_Intruder_File);
    fputs("\n", TR_Intruder_File);
}
