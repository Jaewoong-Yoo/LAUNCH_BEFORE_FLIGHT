#include "header.h"

#include <ads_b_read/Traffic_Report.h>
#include <ads_b_read/Traffic_Report_Array.h>

struct struct_t_Traffic_Report     StrTraffic_Report;

ads_b_read::Traffic_Report       TR_Data_sub;
ads_b_read::Traffic_Report_Array TR_Data_Array_Sub;

ros::Publisher TR_Ownship_pub;
ros::Publisher TR_Intruder_pub;
ros::Subscriber TR_Data_Array_sub;

void TR_Data_Array_callback(const ads_b_read::Traffic_Report_Array::ConstPtr& msg_input);

uint32_t count_ros = 0;


int main (int argc, char** argv){
    ros::init(argc, argv, "parse_node");
    ros::NodeHandle nh;

    TR_Ownship_pub     = nh.advertise<ads_b_read::Traffic_Report>       ("/ADS_B/TR_Ownship",       1          );
    TR_Intruder_pub    = nh.advertise<ads_b_read::Traffic_Report>       ("/ADS_B/TR_Intruder",      1          );
    TR_Data_Array_sub  = nh.subscribe("/ADS_B/Traffic_Report_Array", 1000, TR_Data_Array_callback);

    ros::Rate loop_rate(ROS_FREQ_HIGH);

    while(ros::ok())
    {
        count_ros++;
        loop_rate.sleep();
        ros::spinOnce();

    }
}

void TR_Data_Array_callback(const ads_b_read::Traffic_Report_Array::ConstPtr& msg_input)
{
    TR_Data_Array_Sub.Traffic_Reports.clear();
    for (int i = 0; i<msg_input->Traffic_Reports.size();i++)
    {
        const ads_b_read::Traffic_Report& TR_Data_Sub = msg_input->Traffic_Reports[i];
        TR_Data_Array_Sub.Traffic_Reports.push_back(TR_Data_Sub);
        if (TR_Data_Sub.ICAO_address == 7463479) // Ownship 7463479
        {
            ROS_INFO_STREAM("Publish [Ownship  Data: " << TR_Data_Sub.ICAO_address << "] from /ADS_B/Traffic_Report_Array to /ADS_B/TR_Ownship");
            TR_Ownship_pub.publish(TR_Data_Sub);
        }
        if (TR_Data_Sub.ICAO_address == 7463475) // Intruder 7441001
        {
            ROS_INFO_STREAM("Publish [Intruder Data: " << TR_Data_Sub.ICAO_address << "] from /ADS_B/Traffic_Report_Array to /ADS_B/TR_Intruder");
            TR_Intruder_pub.publish(TR_Data_Sub);
        }
#if DEBUG_MODE
        ROS_INFO_STREAM("ICAO_address[" << i << "] : " << TR_Data_Sub.ICAO_address);
//        ROS_INFO_STREAM("ICAO_address_Arrary : " << TR_Data_Array_Sub.Traffic_Reports[0].ICAO_address);
#endif
    }
}
