

#include "DefineList.h"
#include "CommModule.h"
#include "SystemModule.h"

// setup the initial name
using namespace ros;
using namespace std;

//void RX_FCC_Data_Publish(void);

#define  ROS_FREQ  20.0    // while loop frequency [Hz]
#define  _Agent_Number_ 101
#define  DEBUG_MODE 1
#define  HS_MODE 0

void UpdateCommand(void);
void FlightState_Publish(void);

int CLOCK()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

// ROS topic define
#if HS_MODE
	Subscriber msg_flag_sub;
	Subscriber msg_cmd_sub;

	Publisher  mode_motor_pub;
	Publisher  mode_flight_pub;

	Publisher  global_LLH_pub;
	Publisher  global_ECEF_pub;
	Publisher  global_sat_pub;

	Publisher  local_pos_pub;
	Publisher  local_vel_pub;
	Publisher  local_alt_pub;

	Publisher  imu_att_pub;
	Publisher  imu_pqr_pub;
	Publisher  imu_accel_pub;

	Publisher  sys_time_pub;


	std_msgs::UInt8             Mode_Motor;
	std_msgs::UInt8             Mode_Flightmode;

	std_msgs::Float64MultiArray GNSS_LLH;
	std_msgs::Float64MultiArray GNSS_ECEF;
	std_msgs::UInt8             GNSS_satnum;

	std_msgs::Float32MultiArray Local_Pos;
	std_msgs::Float32MultiArray Local_Vel;
	std_msgs::Float32           Local_Alt;

	std_msgs::Float32MultiArray IMU_Att;
	std_msgs::Float32MultiArray IMU_pqr;
	std_msgs::Float32MultiArray IMU_Accel;

	std_msgs::Float32           SYS_Time;
#else
	ros::Publisher RX_FCC_Data_pub;

	fcc_serial::MsgStrRXttyO RX_FCC_Data;
//	std_msgs::Float32MultiArray RX_FCC_Data;
#endif


// node main loop, for ROS
int main(int argc, char** argv)
{
//    System_Initialization(argc, argv);       // System initialization

    init(argc, argv, "fcc_serial_node");          // node    name initialization
    NodeHandle nh;                           // assign node handler

    printf("Initiate: fcc_serial_node\n");   // for debugging

    // Publish Topic
#if HS_MODE
    mode_motor_pub    = nh.advertise<std_msgs::UInt8>("/UAV_101/mode/MotorAct",10);
    mode_flight_pub   = nh.advertise<std_msgs::UInt8>("/UAV_101/mode/Flightmode",10);

    global_sat_pub    = nh.advertise<std_msgs::UInt8>("/UAV_101/global_position/sat",10);
    global_LLH_pub    = nh.advertise<std_msgs::Float64MultiArray>("/UAV_101/global_position/LLH",10);
    global_ECEF_pub   = nh.advertise<std_msgs::Float64MultiArray>("/UAV_101/global_position/ECEF",10);

    local_pos_pub     = nh.advertise<std_msgs::Float32MultiArray>("/UAV_101/local_position/pos",10);
    local_vel_pub     = nh.advertise<std_msgs::Float32MultiArray>("/UAV_101/local_position/vel",10);
    local_alt_pub     = nh.advertise<std_msgs::Float32>("/UAV_101/local_position/alt",10);

    imu_att_pub       = nh.advertise<std_msgs::Float32MultiArray>("/UAV_101/imu/attitude",10);
    imu_pqr_pub       = nh.advertise<std_msgs::Float32MultiArray>("/UAV_101/imu/pqr",10);
    imu_accel_pub     = nh.advertise<std_msgs::Float32MultiArray>("/UAV_101/imu/accel",10);

    sys_time_pub      = nh.advertise<std_msgs::Float32>("/system/time",10);
#else
    RX_FCC_Data_pub = nh.advertise<fcc_serial::MsgStrRXttyO>("/UAV/101/FCC", 1);
    printf("Initiate: publish rostopic </UAV/001/FCC>\n");   // for debugging
//    RX_FCC_Data_pub = nh.advertise<std_msgs::Float32MultiArray>("/UAV_101/FCC", 30);
#endif

    FdPort1 = OpenSerial(PORT1);             // Open Serial
//    FdPort1 = OpenSerial("/dev/tty");             // Open Serial
    SerialReceive(FdPort1);                  // Serial Receive (pthread)
    Rate loop_rate(ROS_FREQ);                      // setup the loop speed, [Hz]

    Initialization();                        // Mission Initialization

    // node loop, for ROS, check ros status, ros::ok()
    while( ok() )
    {
        clock_t start = CLOCK();
        sys_time = (float)start/CLOCKS_PER_SEC*100.0;
		
        UpdateCommand();

        // Serial TX part
        SerialSend(FdPort1);

        FlightState_Publish();

        count_ros = count_ros + 1;
        t_cur = count_ros/ROS_FREQ;

#if DEBUG_MODE
        printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
        printf("\t[USRG] Serial data from FCC\n");
        printf("[count_ros] \t %d \t[time_ros] \t %.2f\n", count_ros, count_ros/20.0);
        printf("[Agent_Number] \t %d \t[time_fcc] \t %.3f \t[Motor] \t %d\n", _Agent_Number_, StrRXttyO.Cur_Time_sec, StrRXttyO.Mode_MotorAct);
        printf("[FlightMode] \t %d \t[VFlyingStat] \t %d \t[Voltage] \t %.2f \t[Current] \t %.2f\n", StrRXttyO.Mode_FlightMode, StrRXttyO.Mode_VehicleFlyingStatus, StrRXttyO.ADC_volt[0], StrRXttyO.ADC_volt[1]);
        printf("\n");
        printf("[Sat_Number] \t %2.f \t[Lon] \t %3.7f \t[Lat] \t %2.7f \t[Height] \t %3.3f\n", StrRXttyO.GNSS_SatNum, StrRXttyO.GNSS_LLH_degdegm[0], StrRXttyO.GNSS_LLH_degdegm[1], StrRXttyO.GNSS_LLH_degdegm[2]);
        printf("\n");
        printf("[roll_rate]  \t %2.3f \t[pitch_rate] \t %2.3f \t[yaw_rate] \t %3.3f\n", StrRXttyO.Cur_AngularRate_dps[0], StrRXttyO.Cur_AngularRate_dps[1], StrRXttyO.Cur_AngularRate_dps[2]);
        printf("[roll]  \t %2.3f \t[pitch] \t %2.3f \t[yaw] \t %3.3f\n", StrRXttyO.Cur_Att_deg[0], StrRXttyO.Cur_Att_deg[1], StrRXttyO.Cur_Att_deg[2]);
        printf("\n");
        printf("[LinAcc_0]  \t %2.3f \t[LinAcc_1] \t %2.3f \t[LinAcc_1] \t %3.3f\n", StrRXttyO.Cur_LinAccAED_mpss[0], StrRXttyO.Cur_LinAccAED_mpss[1], StrRXttyO.Cur_LinAccAED_mpss[2]);
        printf("[Vel_0]  \t %2.3f \t[Vel_1] \t %2.3f \t[Vel_2] \t %3.3f\n", StrRXttyO.Cur_VelAED_mps[0], StrRXttyO.Cur_VelAED_mps[1], StrRXttyO.Cur_VelAED_mps[2]);
        printf("\n");
        printf("[North_m]  \t %2.3f \t[East_m] \t %2.3f \t[Down_m] \t %3.3f\n", StrRXttyO.Cur_PosNED_m[0], StrRXttyO.Cur_PosNED_m[1], StrRXttyO.Cur_PosNED_m[2]);
        printf("[Vel_N_m]  \t %2.3f \t[Vel_E_m] \t %2.3f \t[Vel_D_m] \t %3.3f\n", StrRXttyO.Cur_VelNED_mps[0], StrRXttyO.Cur_VelNED_mps[1], StrRXttyO.Cur_VelNED_mps[2]);
        printf("\n");
        printf("[SensorStatus] %d\n", StrRXttyO.SensorStatus);
        printf("[stick_Ail] \t %4.f \t[stick_Ele] \t %4.f \t[stick_Thr] \t %4.f \t[stick_Rud] \t %4.f\n", StrRXttyO.TXStickPosAil_usec, StrRXttyO.TXStickPosEle_usec, StrRXttyO.TXStickPosThr_usec, StrRXttyO.TXStickPosRud_usec);
        printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
        printf("\n");
#endif

        // loop rate [Hz]
        loop_rate.sleep();

	// loop sampling, ros
        spinOnce();
    }

    // for debugging
    printf("Terminate: FCC_Serial_node\n");

    return 0;
}




void Initialization(void)
{

}


void UpdateCommand(void)
{

    // Command  TX2 ---> DS_Board

    // flagA = 0   : default        (None)
    // flagA = 1   : Auto take-off  (be careful!!!!!)
    // flagA = 4   : Motor Cut-off

    // velx        : forward(+) backward(-) (Body-axis)
    // vely        : right  (+) left    (-) (Body-axis)
    // velz        : down   (+) up      (-) (Body-axis)
    // cmdr        : right  (+) left    (-) (Body-axis)  **yaw rate command

    flagA = 0;
    flagB = 0;
    flagC = 0;
    flagD = 0;

    velx = 3.0;
    vely = 0.0;
    velz = 0.0;
    cmdr = 0.0;


    // tx_data update
    tx_data.FlagA      = flagA;
    tx_data.FlagB      = flagB;
    tx_data.FlagC      = flagC;
    tx_data.FlagD      = flagD;

    tx_data.CmdVelAil  = satmax(vely,Vx_max);
    tx_data.CmdVelEle  = satmax(velx,Vx_max);
    tx_data.CmdVelDown = satmax(velz,Vz_max);
    tx_data.CmdR_dps   = cmdr;

    unsigned char *data = (unsigned char *)&tx_data;
    memcpy((void *)(tx.Data),(void *)(data),sizeofdata);

}


void FlightState_Publish(void)
{
#if HS_MODE
    Mode_Motor.data = StrRXttyO.Mode_MotorAct;
    mode_motor_pub.publish(Mode_Motor);

    Mode_Flightmode.data = StrRXttyO.Mode_FlightMode;
    mode_flight_pub.publish(Mode_Flightmode);

    GNSS_satnum.data = StrRXttyO.GNSS_SatNum;
    global_sat_pub.publish(GNSS_satnum);

    GNSS_LLH.data.resize(3);
    GNSS_LLH.data[0] = StrRXttyO.GNSS_LLH_degdegm[0];
    GNSS_LLH.data[1] = StrRXttyO.GNSS_LLH_degdegm[1];
    GNSS_LLH.data[2] = StrRXttyO.GNSS_LLH_degdegm[2];
    global_LLH_pub.publish(GNSS_LLH);

    GNSS_ECEF.data.resize(3);
    GNSS_ECEF.data[0] = StrRXttyO.GNSS_ECEF_m[0];
    GNSS_ECEF.data[1] = StrRXttyO.GNSS_ECEF_m[1];
    GNSS_ECEF.data[2] = StrRXttyO.GNSS_ECEF_m[2];
    global_ECEF_pub.publish(GNSS_ECEF);

    Local_Pos.data.resize(3);
    Local_Pos.data[0] = StrRXttyO.Cur_PosNED_m[0];
    Local_Pos.data[1] = StrRXttyO.Cur_PosNED_m[1];
    Local_Pos.data[2] = StrRXttyO.Cur_PosNED_m[2];
    local_pos_pub.publish(Local_Pos);

    Local_Vel.data.resize(3);
    Local_Vel.data[0] = StrRXttyO.Cur_VelNED_mps[0];
    Local_Vel.data[1] = StrRXttyO.Cur_VelNED_mps[1];
    Local_Vel.data[2] = StrRXttyO.Cur_VelNED_mps[2];
    local_vel_pub.publish(Local_Vel);

    Local_Alt.data = StrRXttyO.BaroPosDown_m;
    local_alt_pub.publish(Local_Alt);

    IMU_Att.data.resize(3);
    IMU_Att.data[0] = StrRXttyO.Cur_Att_deg[0];
    IMU_Att.data[1] = StrRXttyO.Cur_Att_deg[1];
    IMU_Att.data[2] = StrRXttyO.Cur_Att_deg[2];
    imu_att_pub.publish(IMU_Att);

    IMU_pqr.data.resize(3);
    IMU_pqr.data[0] = StrRXttyO.Cur_AngularRate_dps[0];
    IMU_pqr.data[1] = StrRXttyO.Cur_AngularRate_dps[1];
    IMU_pqr.data[2] = StrRXttyO.Cur_AngularRate_dps[2];
    imu_pqr_pub.publish(IMU_pqr);

    IMU_Accel.data.resize(3);
    IMU_Accel.data[0] = StrRXttyO.Cur_LinAccAED_mpss[0];
    IMU_Accel.data[1] = StrRXttyO.Cur_LinAccAED_mpss[1];
    IMU_Accel.data[2] = StrRXttyO.Cur_LinAccAED_mpss[2];
    imu_accel_pub.publish(IMU_Accel);

    SYS_Time.data = sys_time;
    sys_time_pub.publish(SYS_Time);
#else
	RX_FCC_Data.header.stamp = ros::Time::now();
	RX_FCC_Data.header.frame_id = "/UAV/101/FCC";
	RX_FCC_Data.Agent_Number = _Agent_Number_;
	memcpy((void *)(&RX_FCC_Data.HeaderA), (void *)(&StrRXttyO), sizeof(StrRXttyO));
	RX_FCC_Data_pub.publish(RX_FCC_Data);
#endif
}

