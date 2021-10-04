// essential header for ROS-OpenCV operation
#include <ros/ros.h>

// for using standard messages, float 32 type
// communicate to image processing algorithm result
#include <std_msgs/Float32MultiArray.h>


// for using serial communication
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <arpa/inet.h>


void arrayCallback(const std_msgs::Float32MultiArray& array);


// setup the initial name
using namespace ros;
using namespace std;

std_msgs::Float32MultiArray receive_data;

#define DF_UDP_BUFFER_SIZE  8192
#define DF_UDP_PORTNUM      31000				
#define DF_UDP_SERVER_ADDR  "127.0.0.1"				

float 	UAV[4];


struct struct_t_UDP
{
    int    Socket;
	struct sockaddr_in ServerAddr;
    float  TXBuffer[DF_UDP_BUFFER_SIZE];
    char   RXBuffer[DF_UDP_BUFFER_SIZE];
};

#pragma pack(1)
struct TX_message_data
{
    float UAV_N;
    float UAV_E;
    float UAV_D;
    float WP_case;
};
#pragma pack()

#pragma pack(2)
struct RX_message_data
{
    float R_UAV_N;
    float R_UAV_E;
    float R_UAV_D;
    float R_WP_case;
};
#pragma pack()

#pragma pack()
struct struct_udp_test
{
    // -------------
    // Packet Header
    // -------------

    uint32_t a0;
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
    uint32_t b4;
};
#pragma pack()

struct struct_t_UDP           StrUDP;
struct sockaddr_in            MyAddr;
struct TX_message_data        TX_buff;
struct RX_message_data        RX_buff;
struct struct_udp_test	      RX_test;

//struct struct_t_MainFuncArgs  StrMainFuncArgs;

// for publishing the kalman states
std_msgs::Float32MultiArray msg_image;


// node main loop, for ROS
int main(int argc, char** argv)
{
	// node name initialization
	init(argc, argv, "udp_server");

	// assign node handler
	ros::NodeHandle nh_;

	// for debugging
	printf("Initiate: vel_track_node\n");

	//Publish
	Publisher  P_UAV	= nh_.advertise<std_msgs::Float32MultiArray>("/UAV3", 100);			// UAV

	//Subscribe
	ros::Subscriber sub1 = nh_.subscribe("/UAV1",100,arrayCallback);						//UAV

	receive_data.data.resize(10);
	ros::Rate loop_rate(20);
	float fdt = (float)(1/20);


	// Socket Creation
	StrUDP.Socket = socket(PF_INET, SOCK_DGRAM, 0);
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

    char recvBuff;

	int size_addr = sizeof(StrUDP.ServerAddr);


	// node loop, for ROS, check ros status, ros::ok()
	while( ok() )
	{
        printf("Server!\n");

		recvfrom(StrUDP.Socket,(char*)&RX_test,sizeof(RX_test), 0, (struct sockaddr *)(&StrUDP.ServerAddr), (socklen_t *)&size_addr);

		printf( "print : %d", RX_test ); 


        //printf("recv data : %.3f  %.3f  %.3f %.3f\n",RX_buff.R_UAV_N, RX_buff.R_UAV_E, RX_buff.R_UAV_D,RX_buff.R_WP_case);

		TX_buff.UAV_N	=UAV[0];					// UAV
		TX_buff.UAV_E	=UAV[1];					// UAV
		TX_buff.UAV_D	=UAV[2];					// UAV
		TX_buff.WP_case	=UAV[3];					// UAV

        sendto(StrUDP.Socket, (char*)&TX_buff, sizeof(TX_buff), 0, (struct sockaddr *)(&StrUDP.ServerAddr), sizeof(StrUDP.ServerAddr));
        printf("send data : %.3f  %.3f  %.3f %.3f\n",TX_buff.UAV_N, TX_buff.UAV_E, TX_buff.UAV_D,TX_buff.WP_case);

    	// messages
    	std_msgs::Float32MultiArray UAV_msg;

    	UAV_msg.data.clear();
    	UAV_msg.data.resize(4);
    	UAV_msg.data[0] = RX_buff.R_UAV_N;
    	UAV_msg.data[1] = RX_buff.R_UAV_E;
    	UAV_msg.data[2] = RX_buff.R_UAV_D;
    	UAV_msg.data[3] = RX_buff.R_WP_case;

		P_UAV.publish(UAV_msg);

		// loop rate [Hz]
        loop_rate.sleep();

		// loop sampling, ros
		spinOnce();
	}

	// for debugging
	printf("Terminate: FCC_Serial_node\n");
    close(StrUDP.Socket);

	return 0;
}


void arrayCallback(const std_msgs::Float32MultiArray& array)
{
	UAV[0] = array.data[0];		//N
	UAV[1] = array.data[1];		//E
	UAV[2] = array.data[2];		//D
	UAV[3] = array.data[3];		//wp
	return;
}
