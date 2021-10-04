#include "DefineList.h"

// setup the initial name
using namespace ros;
using namespace std;

#define  ROS_FREQ  5.0    // while loop frequency [Hz]
#define DEBUG_MODE 1

#define DF_UDP_BUFFER_SIZE      3
// #define DF_UDP_PORTNUM      14550;


struct struct_t_UDP
{
    int Socket;
    struct sockaddr_in ServerAddr;
};

struct struct_t_UDP           StrUDP;
struct sockaddr_in            MyAddr;
struct struct_t_MsgUV         RX_buff;
//struct struct_t_MsgServer  TX_buff;

fcc_serial::MsgStrRXttyO FCC_Data_buff;
std_msgs::Float32MultiArray gps_data;

// node main loop, for ROS
int main(int argc, char** argv)
{
    // node name initialization
    init(argc, argv, "server_udp_node_uav1");

    // assign node handler
    ros::NodeHandle nh;

    // for debugging
    printf("Initiate: server_udp_node_uav1\n");

    // Publish Topic
    ros::Publisher FCC_Data_pub    = nh.advertise<fcc_serial::MsgStrRXttyO>("/UAV1/FCC", 1);
    ros::Publisher GPS_pub         = nh.advertise<std_msgs::Float32MultiArray> ("/UAV1/GPS"  ,1);
   

    // Subscribing the flight state

    // Get port number
    int UDP_PORTNUM;
    nh.getParam("udp_portnum_uav1", UDP_PORTNUM);

    // setup the loop speed, [Hz], synchronizing the hector slam loop
    ros::Rate loop_rate(ROS_FREQ);

    // Socket_Rear Creation
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

    //
    // UDP-IP Setting
    memset(&StrUDP.ServerAddr, 0, sizeof(StrUDP.ServerAddr)); // Clear to 0
    StrUDP.ServerAddr.sin_family      = PF_INET;
    StrUDP.ServerAddr.sin_port        = htons(UDP_PORTNUM); // PORT#
#ifdef REAR_SERVER
    StrUDP.ServerAddr.sin_addr.s_addr = inet_addr(DF_UDP_DRONE_ADDR_REAR);
#endif
#ifdef TOP_SERVER
    StrUDP.ServerAddr.sin_addr.s_addr = inet_addr(DF_UDP_DRONE_ADDR_TOP);
#endif

    memset(&MyAddr, 0, sizeof(MyAddr));
    MyAddr.sin_family = PF_INET;
    MyAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    MyAddr.sin_port = htons(UDP_PORTNUM);

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
        // recvfrom(StrUDP.Socket,(char*)&RX_buff,sizeof(RX_buff), 0, (struct sockaddr *)(&StrUDP.ServerAddr), (socklen_t *)&size_addr);
        // sendto(StrUDP.Socket, (char*)&TX_buff, sizeof(TX_buff), 0, (struct sockaddr *)(&StrUDP.ServerAddr), sizeof(StrUDP.ServerAddr));

        int recv_len;
        recv_len = recv(StrUDP.Socket,(char*)&RX_buff,sizeof(RX_buff), 0);

        if (recv_len >0)
        {
	    memcpy((void *)(&FCC_Data_buff), (void *)(&RX_buff), sizeof(RX_buff.FCC_Data));
            FCC_Data_pub.publish(FCC_Data_buff);

#if DEBUG_MODE
            system("clear");
// recv
            printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
            printf("\t[USRG] Receive the FCC message from the client [%d]\n", RX_buff.Agent_Number);
            printf("[Agent_Number] \t %d \t[count_udp] \t %d \t[Motor] \t %d \t[time_fcc] \t %.3f\n", RX_buff.Agent_Number, count_udp, RX_buff.FCC_Data.Mode_MotorAct, RX_buff.FCC_Data.Cur_Time_sec);
//             printf("[FlightMode] \t %d \t[VFlyingStat] \t %d \t[Voltage] \t %.2f \t[Current] \t %.2f\n", RX_buff.FCC_Data.Mode_FlightMode, RX_buff.FCC_Data.Mode_VehicleFlyingStatus, RX_buff.FCC_Data.ADC_volt[0], RX_buff.FCC_Data.ADC_volt[1]);
            printf("\n");
            printf("[Sat_Number] \t %2.f \t[Lon] \t %3.7f \t[Lat] \t %2.7f \t[Height_GPS] \t %3.3f\n", RX_buff.FCC_Data.GNSS_SatNum, RX_buff.FCC_Data.GNSS_LLH_degdegm[0], RX_buff.FCC_Data.GNSS_LLH_degdegm[1], RX_buff.FCC_Data.GNSS_LLH_degdegm[2]);
            printf("\n");
            printf("[roll]  \t %2.2f \t[pitch] \t %2.2f \t[yaw] \t %3.2f\n", RX_buff.FCC_Data.Cur_Att_deg[0], RX_buff.FCC_Data.Cur_Att_deg[1], RX_buff.FCC_Data.Cur_Att_deg[2]);
//             printf("[roll_rate]  \t %2.2f \t[pitch_rate] \t %2.2f \t[yaw_rate] \t %3.2f\n", RX_buff.FCC_Data.Cur_AngularRate_dps[0], RX_buff.FCC_Data.Cur_AngularRate_dps[1], RX_buff.FCC_Data.Cur_AngularRate_dps[2]);
//             printf("\n");
//             printf("[LinAcc_0]  \t %2.2f \t[LinAcc_1] \t %2.2f \t[LinAcc_1] \t %3.2f\n", RX_buff.FCC_Data.Cur_LinAccAED_mpss[0], RX_buff.FCC_Data.Cur_LinAccAED_mpss[1], RX_buff.FCC_Data.Cur_LinAccAED_mpss[2]);
            printf("[Vel_for]  \t %2.2f \t[Vel_side] \t %2.2f \t[Vel_down] \t %3.2f\n", RX_buff.FCC_Data.Cur_VelAED_mps[0], RX_buff.FCC_Data.Cur_VelAED_mps[1], RX_buff.FCC_Data.Cur_VelAED_mps[2]);
            printf("\n");
            printf("[North_m]  \t %2.2f \t[East_m] \t %2.2f \t[Down_m] \t %3.2f \t[LIDAR_m] \t %3.2f\n", RX_buff.FCC_Data.Cur_PosNED_m[0], RX_buff.FCC_Data.Cur_PosNED_m[1], RX_buff.FCC_Data.Cur_PosNED_m[2], RX_buff.FCC_Data.LidarPosDown_m);
            printf("[Vel_N_m]  \t %2.2f \t[Vel_E_m] \t %2.2f \t[Vel_D_m] \t %3.2f\n", RX_buff.FCC_Data.Cur_VelNED_mps[0], RX_buff.FCC_Data.Cur_VelNED_mps[1], RX_buff.FCC_Data.Cur_VelNED_mps[2]);
            printf("\n");
//             printf("[SensorStatus] 0x%X\n", RX_buff.FCC_Data.SensorStatus);
            printf("[stick_Ail] \t %4.f \t[stick_Ele] \t %4.f \t[stick_Thr] \t %4.f \t[stick_Rud] \t %4.f\n", RX_buff.FCC_Data.TXStickPosAil_usec, RX_buff.FCC_Data.TXStickPosEle_usec, RX_buff.FCC_Data.TXStickPosThr_usec, RX_buff.FCC_Data.TXStickPosRud_usec);
//             printf("[TX_CH_8] \t %4.f \t[TX_CH_9] \t %4.f \t[TX_CH_13] \t %4.f \t[TX_CH_14] \t %4.f \t[TX_CH_15] \t %4.f \t[TX_CH_16] \t %4.f\n", RX_buff.FCC_Data.TX_CH_8_9_13_14_15_16_usec[0], RX_buff.FCC_Data.TX_CH_8_9_13_14_15_16_usec[1], RX_buff.FCC_Data.TX_CH_8_9_13_14_15_16_usec[2], RX_buff.FCC_Data.TX_CH_8_9_13_14_15_16_usec[3], RX_buff.FCC_Data.TX_CH_8_9_13_14_15_16_usec[4], RX_buff.FCC_Data.TX_CH_8_9_13_14_15_16_usec[5]);
            printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
            printf("\n");
#endif
        }

        float heading = 0.0;

        if( RX_buff.FCC_Data.Cur_Att_deg[2]>=0){
            heading = RX_buff.FCC_Data.Cur_Att_deg[2];
        }
        else{
            heading = 360.0 + RX_buff.FCC_Data.Cur_Att_deg[2]; 
        }

        gps_data.data.resize(4);
        gps_data.data[0] = RX_buff.FCC_Data.GNSS_LLH_degdegm[1]; // Lat
        gps_data.data[1] = RX_buff.FCC_Data.GNSS_LLH_degdegm[0]; // Lon
        gps_data.data[2] = RX_buff.FCC_Data.GNSS_LLH_degdegm[2]; // Alt
        gps_data.data[3] = heading;
        GPS_pub.publish(gps_data);
        

        // loop rate [Hz]
        loop_rate.sleep();

        count_udp++;

        // loop sampling, ros
        spinOnce();
    }

    // for debugging
    printf("Terminate: comm_udp_node_uav1\n");
    close(StrUDP.Socket);

    return 0;
}
