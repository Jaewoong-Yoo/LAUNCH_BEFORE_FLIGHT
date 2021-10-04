#include "header.h"

#include <stdio.h>
#include <boost/filesystem.hpp>

using namespace std;

ros::Subscriber MAVROS_GPS_sub;

void MAVROS_GPS_callback(const sensor_msgs::NavSatFix& msg_input);

uint32_t count_ros = 0;
FILE* MAVROS_GPS_File;
char MAVROS_GPS_FolderPath[100];
char MAVROS_GPS_FilePath[100];
char MAVROS_GPS_String[1000];

int main (int argc, char** argv){
    ros::init(argc, argv, "mavros_log_node");
    ros::NodeHandle nh;

    MAVROS_GPS_sub  = nh.subscribe("/mavros/global_position/global", 1000, MAVROS_GPS_callback);

    ros::Rate loop_rate(ROS_FREQ_HIGH); // 1000

    struct tm *date;
    const time_t t = time(NULL);
    date = localtime(&t);

    sprintf(MAVROS_GPS_FolderPath, "/home/usrg/Civil/ADS_B_txt_log");
    boost::filesystem::create_directories(MAVROS_GPS_FolderPath);

    sprintf(MAVROS_GPS_FilePath, "/home/usrg/Civil/ADS_B_txt_log/%d_%d_%d_%d_%d_%d_mavros_gps.txt",
            (date->tm_year+1900), (date->tm_mon+1), date->tm_mday, date->tm_hour, date->tm_min, date->tm_sec);
    cout << MAVROS_GPS_FilePath << endl;

    MAVROS_GPS_File = fopen(MAVROS_GPS_FilePath, "w");
    fputs("[mavros_gps_Data]\n", MAVROS_GPS_File);
    fputs("timestamp, altitude, latitude, longitude\n", MAVROS_GPS_File);
    ROS_INFO_STREAM("Start Logging GPS Data");

    while(ros::ok())
    {
        count_ros++;
        loop_rate.sleep();
        ros::spinOnce();
    }

    fclose(MAVROS_GPS_File);
}
void MAVROS_GPS_callback(const sensor_msgs::NavSatFix& msg_input)
{
    ROS_INFO_STREAM("Write [mavros gps Data] from /mavros/global_position/global to /home/usrg/ADS_B_txt_log/");
    sprintf(MAVROS_GPS_String, "%f, %f, %f, %f",
            msg_input.header.stamp.toSec(), msg_input.altitude, msg_input.latitude, msg_input.longitude);
    fputs(MAVROS_GPS_String, MAVROS_GPS_File);
    fputs("\n", MAVROS_GPS_File);
}
