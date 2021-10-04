
#include "DefineList.h"
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>

// setup the initial name
using namespace ros;
using namespace std;

#define ROS_FREQ  5.0    // while loop frequency [Hz]
#define DEBUG_MODE 1

#define DF_UDP_BUFFER_SIZE  3
#define DF_UDP_PORTNUM      14580 //14550

struct struct_t_UDP
{
    int Socket;
    struct sockaddr_in ServerAddr;
};

// Socket Variable
struct struct_t_UDP           StrUDP;
struct sockaddr_in            MyAddr;

// lat, lon, alt, heading, velx, vely, velz
float TX_buff[7];

void RX_Pos_Data_Callback(const sensor_msgs::NavSatFix::ConstPtr& msg_input);
void RX_Vel_Data_Callback(const geometry_msgs::TwistStamped::ConstPtr& msg_input);
void RX_Heading_Data_Callback(const nav_msgs::Odometry::ConstPtr& msg_input);

// Define Global Variable
float Cur_Att_rad[3];

// Define Ros Variable
sensor_msgs::NavSatFix      Local;
geometry_msgs::TwistStamped Localvel;
nav_msgs::Odometry          Localheading;

// node main loop, for ROS
int main(int argc, char** argv)
{
    // node name initialization
    init(argc, argv, "client_udp_node");

    // assign node handler
    ros::NodeHandle nh;

    // for debugging
    printf("Initiate: client_udp_node\n");

    // subscribing the flight state
    // ros::Subscriber RX_FCC_Data_sub = nh.subscribe("/mavros/global_position/global", 1, RX_FCC_Data_Callback);
    ros::Subscriber RX_Pos_Data_sub = nh.subscribe("/mavros/global_position/global", 1, RX_Pos_Data_Callback);
    ros::Subscriber RX_Heading_Data_sub = nh.subscribe("/mavros/global_position/local", 1, RX_Heading_Data_Callback);
    ros::Subscriber RX_Vel_Data_sub = nh.subscribe("/mavros/local_position/velocity", 1, RX_Vel_Data_Callback);

    // setup the loop speed, [Hz], synchronizing the hector slam loop
    ros::Rate loop_rate(ROS_FREQ);

    // Socket Creation
    StrUDP.Socket = socket(PF_INET, SOCK_DGRAM, 0);
    int enable = 1;
    setsockopt(StrUDP.Socket,SOL_SOCKET,SO_REUSEADDR,&enable,sizeof(enable));
    setsockopt(StrUDP.Socket,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof(tv));
    if(StrUDP.Socket == -1)
    {
            printf("[ERROR] 'socket()'\n");
            return -1;
    }
    else
    {
            printf("[DONE] UDP socket is created\n");
    }

    // UDP-IP Setting
    memset(&StrUDP.ServerAddr, 0, sizeof(StrUDP.ServerAddr)); // Clear to 0
    StrUDP.ServerAddr.sin_family      = PF_INET;
    StrUDP.ServerAddr.sin_port        = htons(DF_UDP_PORTNUM); // PORT#
    StrUDP.ServerAddr.sin_addr.s_addr = inet_addr(DF_UDP_SERVER_ADDR); // IP for Server (Normally PC IP)

    memset(&MyAddr, 0, sizeof(MyAddr));
    MyAddr.sin_family = PF_INET;
    MyAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    MyAddr.sin_port = htons(DF_UDP_PORTNUM);

    if(bind(StrUDP.Socket,(struct sockaddr *)&MyAddr, sizeof(MyAddr))!=0)
    {
        printf("bind() error!\n");
        return -1;
    }

    int size_addr = sizeof(StrUDP.ServerAddr);
    printf("[DONE] UDP initialized! size : %d\n",size_addr);

    int count_udp = 0;

    // node loop, for ROS, check ros status, ros::ok()
    while( ok() )
    {
        printf("[Waiting]\n");
        sendto(StrUDP.Socket, (char*)&TX_buff, sizeof(TX_buff), 0, (struct sockaddr *)(&StrUDP.ServerAddr), sizeof(StrUDP.ServerAddr));

#if DEBUG_MODE
	    system("clear");
// send
        printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
        printf("\t[USRG] Send the GPS data to Server\n");
        printf("[Lat] \t %2.7f \t[Lon] \t %2.7f \t[Alt] \t %3.2f \n", TX_buff[0], TX_buff[1], TX_buff[2]);
        printf("\n");
        printf("[Heading] \t %3.2f \n", TX_buff[3]);
        printf("\n");
        printf("[Vel_x] \t %2.2f \t[Vel_y] \t %2.2f \t[Vel_z] \t %2.2f \n", TX_buff[4], TX_buff[5], TX_buff[6]);
        printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
#endif

        // loop rate [Hz]
        loop_rate.sleep();
        count_udp++;

        // loop sampling, ros
        spinOnce();
    }

    // for debugging
    printf("Terminate: comm_udp_node\n");
    close(StrUDP.Socket);

    return 0;
}

void RX_Pos_Data_Callback(const sensor_msgs::NavSatFix::ConstPtr& msg_input)
{
    Local = *msg_input;

    TX_buff[0] = Local.latitude;
    TX_buff[1] = Local.longitude;
    TX_buff[2] = Local.altitude;
}

void RX_Vel_Data_Callback(const geometry_msgs::TwistStamped::ConstPtr& msg_input)
{
    Localvel = *msg_input;

    TX_buff[4] = Localvel.twist.linear.x;
    TX_buff[5] = Localvel.twist.linear.y;
    TX_buff[6] = Localvel.twist.linear.z;
}

void RX_Heading_Data_Callback(const nav_msgs::Odometry::ConstPtr& msg_input)
{
    Localheading = *msg_input;

    q[0] = Localheading.pose.pose.orientation.x;
    q[1] = Localheading.pose.pose.orientation.y;
    q[2] = Localheading.pose.pose.orientation.z;
    q[3] = Localheading.pose.pose.orientation.w;
    QuaterniontoEuler(Cur_Att_rad[0], Cur_Att_rad[1], Cur_Att_rad[2]);

    TX_buff[3] = Cur_Att_rad[2] * R2D;
}
