#include "header.h"

#include <ads_b_read/DataStream_Request.h>
#include <ads_b_read/Traffic_Report.h>
#include <ads_b_read/Traffic_Report_Array.h>
#include <ads_b_read/Status.h>
#include <ads_b_read/Dynamic_Ownship.h>
#include <ads_b_read/Static.h>

serial::Serial ser;


struct struct_t_DataStream_Request StrDataStreamRequest;
struct struct_t_Traffic_Report     StrTraffic_Report;
struct struct_t_Status             StrStatus;
struct struct_t_Dynamic_Ownship    StrOwnship;
struct struct_t_Dynamic_Ownship    StrDynamic;
struct struct_t_Static             StrStatic;

ads_b_read::DataStream_Request   DSR_Data;
ads_b_read::Traffic_Report       TR_Data;
ads_b_read::Traffic_Report       TR_Data_sub;
ads_b_read::Traffic_Report_Array TR_Data_Array;
ads_b_read::Traffic_Report_Array TR_Data_Array_Sub;
ads_b_read::Status               Status_Data;
ads_b_read::Dynamic_Ownship      Ownship_Data;
ads_b_read::Dynamic_Ownship      Dynamic_Data;
ads_b_read::Static               Static_Data;


void TR_Data_Array_callback(const ads_b_read::Traffic_Report_Array::ConstPtr& msg_input);
int update_checksum(int CK_calc, int data);

uint8_t  recvData[50][100]; // 50 planes
uint8_t  sendData_Dynamic[50];
uint8_t  sendData_Static[27];
uint8_t  count_send = 0;
uint32_t count_ros = 0;
uint32_t count_TR = 0;
uint32_t count_DSR = 0;
uint32_t count_Status = 0;
uint32_t count_Ownship = 0;
uint8_t  flag_Parsing = 0;
int      CK_calc = 0xffff;



int main (int argc, char** argv){
    ros::init(argc, argv, "ads_b_read_node");
    ros::NodeHandle nh;

    ros::Publisher DSR_pub      = nh.advertise<ads_b_read::DataStream_Request>   ("/ADS_B/DataStream_Request",   1          );
//    ros::Publisher TR_pub       = nh.advertise<ads_b_read::Traffic_Report>       ("/ADS_B/Traffic_Report",       1          );
    ros::Publisher TR_Array_pub = nh.advertise<ads_b_read::Traffic_Report_Array> ("/ADS_B/Traffic_Report_Array", MAX_PLANES );
    ros::Publisher Status_pub   = nh.advertise<ads_b_read::Status>               ("/ADS_B/Status",               1          );
    ros::Publisher Ownship_pub  = nh.advertise<ads_b_read::Dynamic_Ownship>      ("/ADS_B/Ownship",              1          );
#if SEND_DYNAMIC
    ros::Publisher Dynamic_pub  = nh.advertise<ads_b_read::Dynamic_Ownship>      ("/ADS_B/Dynamic",              1          );
#endif
#if SEND_STATIC
    ros::Publisher Static_pub   = nh.advertise<ads_b_read::Static>               ("/ADS_B/Static",               1          );
#endif
        ros::Subscriber TR_Data_Array_sub = nh.subscribe("/ADS_B/Traffic_Report_Array", 1000, TR_Data_Array_callback);


    try
    {
        ser.setPort(PORT1);
        ser.setBaudrate(BAUDRATE);
        serial::Timeout to = serial::Timeout::simpleTimeout(1000);
        ser.setTimeout(to);
        ser.open();
    }
    catch (serial::IOException& e)
    {
        ROS_ERROR_STREAM("Unable to open port ");
        return -1;
    }

    if(ser.isOpen()){
        ROS_INFO_STREAM("Serial Port initialized");
    }else{
        return -1;
    }

    ros::Rate loop_rate(ROS_FREQ);

    while(ros::ok())
    {

        uint8_t recvData[MAX_PLANES][100] = {0};
        int msg = 0;
        int index = 0;


        //
        // Read Serial Data

        if(ser.available()){
            uint8_t temp;
            msg = 0;
            index = 0;

            flag_Parsing = 1;

            printf("[Read]\n");
            while (ser.available())
            {
                ser.read(&temp, 1);
#if DEBUG_MODE
                printf("%X", temp);
#endif
                if (temp == 0xFE)
                {
                    msg++;
                    index = 0;
                }
                recvData[msg][index] = temp;
                index++;
            }
            printf("\n");
#if DEBUG_MODE
            for (int i = 0; i <= msg; i++)
            {
                for (int j = 0; j <= index; j++)
                {
                    printf("%X,", recvData[i][j]);
                }
                printf("\n");
            }
#endif
            printf("\n");
        }
        else
        {
            flag_Parsing = 0;
        }

        //
        // Parsing
        TR_Data_Array.Traffic_Reports.clear();
        for(int i = 0; i <= msg; i++)
        {
            if((recvData[i][0] == 0xFE) && (recvData[i][5] == 0x42))
            {
                CK_calc = 0xffff;
                for (int j = 1;j <= 5 + recvData[i][1]; j++)
                {
                    CK_calc = update_checksum(CK_calc, recvData[i][j]);
                }
                CK_calc = update_checksum(CK_calc, StrDataStreamRequest.CRC_EXTRA);

                memcpy((void *)(&StrDataStreamRequest), (void *)(recvData[i]), sizeof(StrDataStreamRequest)-1);
                StrDataStreamRequest.CRC_PASS = ((CK_calc & 0xff) == StrDataStreamRequest.checksum.CKA) && ((CK_calc >> 8) == StrDataStreamRequest.checksum.CKB);

                if (StrDataStreamRequest.CRC_PASS == 1)
                {
                    printf("\t[DataStream Request]\n");
                    printf("_ROSTOPIC_ \t /ADS_B/DataStream_Request\n");
#if DEBUG_MODE
                    printf("   <Header>\n");
                    printf("[Start_Flag] \t 0x%X \t\t[Length] \t %d \t\t[Sequence] \t %d \t\t[System ID] \t %d \t\t[Component] \t %d \t\t[Message ID] \t %d\n", StrDataStreamRequest.header.Start_Flag, StrDataStreamRequest.header.Length, StrDataStreamRequest.header.Sequence, StrDataStreamRequest.header.System_ID, StrDataStreamRequest.header.Component, StrDataStreamRequest.header.Message_ID);
                    printf("   <Data>\n");
                    printf("[req_msg_rate] \t %d \t\t[tgt_system] \t %d \t\t[tgt_component]  %d \t\t[req_stream_id]  %d \t\t[start_stop] \t %d\n", StrDataStreamRequest.req_message_rate, StrDataStreamRequest.target_system, StrDataStreamRequest.target_component, StrDataStreamRequest.req_stream_id, StrDataStreamRequest.start_stop);
                    printf("   <Checksum>\n");
                    printf("[CKA] \t\t 0x%X \t\t[CKB] \t\t 0x%X \t\t[CK_calc] \t 0x%X   \t[CRC_EXTRA] \t %d \t\t[CRC_PASS] \t %d\n", StrDataStreamRequest.checksum.CKA, StrDataStreamRequest.checksum.CKB, CK_calc, StrDataStreamRequest.CRC_EXTRA, StrDataStreamRequest.CRC_PASS);
                    printf("\n");
#endif
                    memcpy((void *)(&DSR_Data), (void *)(&StrDataStreamRequest), sizeof(StrDataStreamRequest)-1);
                    DSR_Data.header.seq = count_DSR;
                    DSR_Data.header.stamp = ros::Time::now();
                    DSR_Data.header.frame_id = 1;
                    DSR_pub.publish(DSR_Data);
                    count_DSR++;


                }
            }
            if((recvData[i][0] == 0xFE) && (recvData[i][5] == 0xF6))
            {
                CK_calc = 0xffff;
                for (int j = 1;j <= 5 + recvData[i][1]; j++)
                {
                    CK_calc = update_checksum(CK_calc, recvData[i][j]);
                }
                CK_calc = update_checksum(CK_calc, StrTraffic_Report.CRC_EXTRA);

                memcpy((void *)(&StrTraffic_Report), (void *)(recvData[i]), sizeof(StrTraffic_Report)-1);
                StrTraffic_Report.CRC_PASS = ((CK_calc & 0xff) == StrTraffic_Report.checksum.CKA) && ((CK_calc >> 8) == StrTraffic_Report.checksum.CKB);

                if (StrTraffic_Report.CRC_PASS == 1)
                {
                    printf("\t[Traffic Report]\n");
                    printf("_ROSTOPIC_ \t /ADS_B/Traffic_Report \t\t /ADS_B/Traffic_Report/Callsign\n");
#if DEBUG_MODE
                    printf("   <Header>\n");
                    printf("[Start_Flag] \t 0x%X \t\t[Length] \t %d \t\t[Sequence] \t %d \t\t[System ID] \t %d \t\t[Component] \t %d \t\t[Message ID] \t %d\n", StrTraffic_Report.header.Start_Flag, StrTraffic_Report.header.Length, StrTraffic_Report.header.Sequence, StrTraffic_Report.header.System_ID, StrTraffic_Report.header.Component, StrTraffic_Report.header.Message_ID);
                    printf("   <Data>\n");
                    printf("[ICAO_address] \t %u \t[callsign] \t %s\n", StrTraffic_Report.ICAO_address, StrTraffic_Report.callsign);
                    printf("[lat] \t\t %d \t[lon] \t\t %d \t[altitude] \t %d \t[heading] \t %u \t\t[hor_velocity] \t %u \t\t[ver_velocity] \t %d\n", StrTraffic_Report.lat, StrTraffic_Report.lon, StrTraffic_Report.altitude, StrTraffic_Report.heading, StrTraffic_Report.hor_velocity, StrTraffic_Report.ver_velocity);
                    printf("[validFlags] \t 0x%X \t\t[squawk] \t %u \t\t[altitude_type]  0x%X \t\t[emitter_type] \t 0x%X \t\t[tslc] \t\t %u\n", StrTraffic_Report.validFlags, StrTraffic_Report.squawk, StrTraffic_Report.altitude_type, StrTraffic_Report.emitter_type, StrTraffic_Report.tslc);
                    printf("   <Checksum>\n");
                    printf("[CKA] \t\t 0x%X \t\t[CKB] \t\t 0x%X \t\t[CK_calc] \t 0x%X   \t[CRC_EXTRA] \t %d \t\t[CRC_PASS] \t %d \t\t[count] \t %d\n", StrTraffic_Report.checksum.CKA, StrTraffic_Report.checksum.CKB, CK_calc, StrTraffic_Report.CRC_EXTRA, StrTraffic_Report.CRC_PASS, count_TR);
                    printf("\n");
#endif
                    TR_Data.Start_Flag = StrTraffic_Report.header.Start_Flag;
                    TR_Data.Length = StrTraffic_Report.header.Length;
                    TR_Data.Sequence = StrTraffic_Report.header.Sequence;
                    TR_Data.System_ID = StrTraffic_Report.header.System_ID;
                    TR_Data.Component = StrTraffic_Report.header.Component;
                    TR_Data.Message_ID = StrTraffic_Report.header.Message_ID;
                    TR_Data.ICAO_address = StrTraffic_Report.ICAO_address;      // 4bytes
                    TR_Data.lat = StrTraffic_Report.lat;               // 4bytes
                    TR_Data.lon = StrTraffic_Report.lon;               // 4bytes
                    TR_Data.altitude = StrTraffic_Report.altitude;          // 4bytes
                    TR_Data.heading = StrTraffic_Report.heading;          // 2bytes
                    TR_Data.hor_velocity = StrTraffic_Report.hor_velocity;     // 2bytes
                    TR_Data.ver_velocity = StrTraffic_Report.ver_velocity;     // 2bytes
                    TR_Data.validFlags = StrTraffic_Report.validFlags;       // 2bytes
                    TR_Data.squawk = StrTraffic_Report.squawk;           // 2bytes
                    TR_Data.altitude_type = StrTraffic_Report.altitude_type;    // 1bytes
                    memcpy((void *)(&TR_Data.callsign), (void *)(&StrTraffic_Report.callsign), sizeof(TR_Data.callsign));
//                    TR_Data.callsign = StrTraffic_Report.callsign;         // 9bytes
                    TR_Data.emitter_type = StrTraffic_Report.emitter_type;     // 1bytes
                    TR_Data.tslc = StrTraffic_Report.tslc;             // 1bytes
                    TR_Data.CKA = StrTraffic_Report.checksum.CKA;
                    TR_Data.CKB = StrTraffic_Report.checksum.CKB;
                    TR_Data.CRC_PASS = StrTraffic_Report.CRC_PASS;
//                    memcpy((void *)(&TR_Data), (void *)(&StrTraffic_Report), sizeof(TR_Data));
//                    TR_pub.publish(TR_Data);
                    TR_Data.header.seq = count_TR;
                    TR_Data.header.stamp = ros::Time::now();
                    TR_Data.header.frame_id = 1;
                    TR_Data_Array.Traffic_Reports.push_back(TR_Data);
                    count_TR++;
                }
            }
            if((recvData[i][0] == 0xFE) && (recvData[i][5] == 0xCB))
            {
                CK_calc = 0xffff;
                for (int j = 1;j <= 5 + recvData[i][1]; j++)
                {
                    CK_calc = update_checksum(CK_calc, recvData[i][j]);
                }
                CK_calc = update_checksum(CK_calc, StrStatus.CRC_EXTRA);

                memcpy((void *)(&StrStatus), (void *)(recvData[i]), sizeof(StrStatus)-1);
//                StrStatus.status.OK = ((StrStatus.status.raw & 0x01) == 0x01);
//                StrStatus.status.TX_FAIL_1090ES = ((StrStatus.status.raw & 0x02) == 0x02);
//                StrStatus.status.RX_FAIL_1090ES = ((StrStatus.status.raw & 0x04) == 0x04);
//                StrStatus.status.TX_FAIL_UAT = ((StrStatus.status.raw & 0x08) == 0x08);
//                StrStatus.status.RX_FAIL_UAT = ((StrStatus.status.raw & 0x10) == 0x10);
                StrStatus.CRC_PASS = ((CK_calc & 0xff) == StrStatus.checksum.CKA) && ((CK_calc >> 8) == StrStatus.checksum.CKB);

                if (StrStatus.CRC_PASS == 1)
                {
                    printf("\t[Status]\n");
                    printf("_ROSTOPIC_ \t /ADS_B/Status\n");
#if DEBUG_MODE
                    printf("   <Header>\n");
                    printf("[Start_Flag] \t 0x%X \t\t[Length] \t %d \t\t[Sequence] \t %d \t\t[System ID] \t %d \t\t[Component] \t %d \t\t[Message ID] \t %d\n", StrStatus.header.Start_Flag, StrStatus.header.Length, StrStatus.header.Sequence, StrStatus.header.System_ID, StrStatus.header.Component, StrStatus.header.Message_ID);
                    printf("   <Data>\n");
                    printf("[status] \t 0x%X\n", StrStatus.status);
//                    printf("_initializing_ \t %d \t_OK_ \t\t %d \t_TX_FAIL_1090ES_ \t %d \t_RX_FAIL_1090ES_ \t %d \t_TX_FAIL_UAT_ \t %d \t_RX_FAIL_UAT_ \t %d\n", StrStatus.status.INITIALIZING, StrStatus.status.OK, StrStatus.status.TX_FAIL_1090ES, StrStatus.status.RX_FAIL_1090ES, StrStatus.status.TX_FAIL_UAT, StrStatus.status.RX_FAIL_UAT);
                    printf("   <Checksum>\n");
                    printf("[CKA] \t\t 0x%X \t\t[CKB] \t\t 0x%X \t\t[CK_calc] \t 0x%X   \t[CRC_EXTRA] \t %d \t\t[CRC_PASS] \t %d\n", StrStatus.checksum.CKA, StrStatus.checksum.CKB, CK_calc, StrStatus.CRC_EXTRA, StrStatus.CRC_PASS);
                    printf("\n");
#endif
                    memcpy((void *)(&Status_Data), (void *)(&StrStatus), sizeof(StrStatus)-1);
                    Status_Data.header.seq = count_Status;
                    Status_Data.header.stamp = ros::Time::now();
                    Status_Data.header.frame_id = 1;
                    Status_pub.publish(Status_Data);
                    count_Status++;


                }
            }
            if((recvData[i][0] == 0xFE) && (recvData[i][5] == 0xCA))
            {
                CK_calc = 0xffff;
                for (int j = 1;j <= 5 + recvData[i][1]; j++)
                {
                    CK_calc = update_checksum(CK_calc, recvData[i][j]);
                }
                CK_calc = update_checksum(CK_calc, StrOwnship.CRC_EXTRA);

                memcpy((void *)(&StrOwnship), (void *)(recvData[i]), sizeof(StrOwnship)-1);
                StrOwnship.CRC_PASS = ((CK_calc & 0xff) == StrOwnship.checksum.CKA) && ((CK_calc >> 8) == StrOwnship.checksum.CKB);

                if (StrOwnship.CRC_PASS == 1)
                {
                    printf("\t[Ownship]\n");
                    printf("_ROSTOPIC_ \t /ADS_B/Ownship\n");
#if DEBUG_MODE
                    printf("   <Header>\n");
                    printf("[Start_Flag] \t 0x%X \t\t[Length] \t %d \t\t[Sequence] \t %d \t\t[System ID] \t %d \t\t[Component] \t %d \t\t[Message ID] \t %d\n", StrOwnship.header.Start_Flag, StrOwnship.header.Length, StrOwnship.header.Sequence, StrOwnship.header.System_ID, StrOwnship.header.Component, StrOwnship.header.Message_ID);
                    printf("   <Data>\n");
                    printf("[utcTime] \t %d \t\t[latitude] \t %d \t\t[longitude] \t %d \t\t[altPres] \t %d \t\t[altGNSS] \t %d\n", StrOwnship.utcTime, StrOwnship.latitude, StrOwnship.longitude, StrOwnship.altPres, StrOwnship.altGNSS);
                    printf("[accHoriz] \t %u \t[accVert] \t %u \t\t[accVel] \t %u \t\t[velVert] \t %d \t\t[nsVog] \t %d \t\t[ewVog] \t %d\n", StrOwnship.accHoriz, StrOwnship.accVert, StrOwnship.accVel, StrOwnship.velVert, StrOwnship.nsVog, StrOwnship.ewVog);
                    printf("[state] \t 0x%X \t\t[squawk] \t %u \t\t[fixType] \t 0x%X \t\t[numSats] \t %u \t\t[emStatus] \t %u \t\t[control] \t 0x%X\n", StrOwnship.state, StrOwnship.squawk, StrOwnship.fixType, StrOwnship.numSats, StrOwnship.emStatus, StrOwnship.control);
                    printf("   <Checksum>\n");
                    printf("[CKA] \t\t 0x%X \t\t[CKB] \t\t 0x%X \t\t[CK_calc] \t 0x%X   \t[CRC_EXTRA] \t %d \t\t[CRC_PASS] \t %d\n", StrOwnship.checksum.CKA, StrOwnship.checksum.CKB, CK_calc, StrOwnship.CRC_EXTRA, StrOwnship.CRC_PASS);
                    printf("\n");
#endif
                    Ownship_Data.Start_Flag = StrOwnship.header.Start_Flag;
                    Ownship_Data.Length     = StrOwnship.header.Length;
                    Ownship_Data.Sequence   = StrOwnship.header.Sequence;
                    Ownship_Data.System_ID  = StrOwnship.header.System_ID;
                    Ownship_Data.Component  = StrOwnship.header.Component;
                    Ownship_Data.Message_ID = StrOwnship.header.Message_ID;
                    Ownship_Data.utcTime    = StrOwnship.utcTime;           // 4bytes
                    Ownship_Data.latitude   = StrOwnship.latitude;          // 4bytes
                    Ownship_Data.longitude  = StrOwnship.longitude;         // 4bytes
                    Ownship_Data.altPres    = StrOwnship.altPres;           // 4bytes
                    Ownship_Data.altGNSS    = StrOwnship.altGNSS;          // 4bytes
                    Ownship_Data.accHoriz   = StrOwnship.accHoriz;         // 4bytes
                    Ownship_Data.accVert    = StrOwnship.accVert;          // 2bytes
                    Ownship_Data.accVel     = StrOwnship.accVel;           // 2bytes
                    Ownship_Data.velVert    = StrOwnship.velVert;          // 2bytes
                    Ownship_Data.nsVog      = StrOwnship.nsVog;            // 2bytes
                    Ownship_Data.ewVog      = StrOwnship.ewVog;            // 2bytes
                    Ownship_Data.state      = StrOwnship.state;            // 2bytes
                    Ownship_Data.squawk     = StrOwnship.squawk;           // 2bytes
                    Ownship_Data.fixType    = StrOwnship.fixType;          // 1bytes
                    Ownship_Data.numSats    = StrOwnship.numSats;          // 1bytes
                    Ownship_Data.emStatus   = StrOwnship.emStatus;         // 1bytes
                    Ownship_Data.control    = StrOwnship.control;          // 1bytes
                    Ownship_Data.CKA        = StrOwnship.checksum.CKA;
                    Ownship_Data.CKB        = StrOwnship.checksum.CKB;
                    Ownship_Data.CRC_PASS   = StrOwnship.CRC_PASS;
//                    memcpy((void *)(&Ownship_Data), (void *)(&StrOwnship), sizeof(Ownship_Data));
                    Ownship_Data.header.seq = count_Ownship;
                    Ownship_Data.header.stamp = ros::Time::now();
                    Ownship_Data.header.frame_id = 1;
                    Ownship_pub.publish(Ownship_Data);
                    count_Ownship++;
                }
            }
        }

        if (TR_Data_Array.Traffic_Reports.size() == 0 && flag_Parsing == 1)
        {
            TR_Data.Start_Flag = 255;
            TR_Data.Length = 0;
            TR_Data.Sequence = 0;
            TR_Data.System_ID = 255;
            TR_Data.Component = 0;
            TR_Data.Message_ID = 255;
            TR_Data.ICAO_address = 0;      // 4bytes
            TR_Data.lat = 0;               // 4bytes
            TR_Data.lon = 0;               // 4bytes
            TR_Data.altitude = 0;          // 4bytes
            TR_Data.heading = 0;          // 2bytes
            TR_Data.hor_velocity = 0;     // 2bytes
            TR_Data.ver_velocity = 0;     // 2bytes
            TR_Data.validFlags = 0;       // 2bytes
            TR_Data.squawk = 0;           // 2bytes
            TR_Data.altitude_type = 0;    // 1bytes
            memcpy((void *)(&TR_Data.callsign), (void *)(&""), sizeof(TR_Data.callsign));
            TR_Data.emitter_type = 0;     // 1bytes
            TR_Data.tslc = 0;             // 1bytes
            TR_Data.CKA = 0;
            TR_Data.CKB = 0;
            TR_Data.CRC_PASS = 1;
            TR_Data.header.seq = count_TR;
            TR_Data.header.stamp = ros::Time::now();
            TR_Data.header.frame_id = 1;
            TR_Data_Array.Traffic_Reports.push_back(TR_Data);
            TR_Array_pub.publish(TR_Data_Array);
        }
        else if (flag_Parsing == 1)
        {
            TR_Array_pub.publish(TR_Data_Array);
        }

        //
        // Send Serial Data
        if (count_ros % ROS_FREQ == 0)
        {
#if SEND_DYNAMIC
            StrDynamic.header.Start_Flag    = 0xFE;
            StrDynamic.header.Length        = 42;
            StrDynamic.header.Sequence      = count_send;
            StrDynamic.header.System_ID     = 1;
            StrDynamic.header.Component     = 156;
            StrDynamic.header.Message_ID    = 202;
            StrDynamic.utcTime              = UINT32_MAX;
            StrDynamic.latitude             = INT32_MAX;
            StrDynamic.longitude            = INT32_MAX;
            StrDynamic.altPres              = INT32_MAX;
            StrDynamic.altGNSS              = INT32_MAX;
            StrDynamic.accHoriz             = UINT32_MAX;
            StrDynamic.accVert              = UINT16_MAX;
            StrDynamic.accVel               = UINT16_MAX;
            StrDynamic.velVert              = INT16_MAX;
            StrDynamic.nsVog                = INT16_MAX;
            StrDynamic.ewVog                = INT16_MAX;
            StrDynamic.state;
            StrDynamic.squawk               = 1200;
            StrDynamic.fixType;
            StrDynamic.numSats              = UINT8_MAX;
            StrDynamic.emStatus;
            StrDynamic.control;

            memcpy((void *)(sendData_Dynamic), (void *)(&StrDynamic), sizeof(StrDynamic)-1);

            CK_calc = 0xffff;
            for (int j = 1;j <= 5 + sendData_Dynamic[1]; j++)
            {
                CK_calc = update_checksum(CK_calc, sendData_Dynamic[j]);
            }
            CK_calc = update_checksum(CK_calc, StrDynamic.CRC_EXTRA);

            StrDynamic.checksum.CKA = (CK_calc & 0xff);
            StrDynamic.checksum.CKB = (CK_calc >> 8);
            sendData_Dynamic[sizeof(sendData_Dynamic)-2] = StrDynamic.checksum.CKA;
            sendData_Dynamic[sizeof(sendData_Dynamic)-1] = StrDynamic.checksum.CKB;

            ser.write(sendData_Dynamic, sizeof(sendData_Dynamic));
            count_send++;

            printf("\t[Dynamic]\n");
            printf("_ROSTOPIC_ \t /ADS_B/Dynamic\n");
#if DEBUG_MODE
            printf("   <Header>\n");
            printf("[Start_Flag] \t 0x%X \t\t[Length] \t %d \t\t[Sequence] \t %d \t\t[System ID] \t %d \t\t[Component] \t %d \t\t[Message ID] \t %d\n", StrDynamic.header.Start_Flag, StrDynamic.header.Length, StrDynamic.header.Sequence, StrDynamic.header.System_ID, StrDynamic.header.Component, StrDynamic.header.Message_ID);
            printf("   <Data>\n");
            printf("[utcTime] \t %d \t\t[latitude] \t %d \t\t[longitude] \t %d \t\t[altPres] \t %d \t\t[altGNSS] \t %d\n", StrDynamic.utcTime, StrDynamic.latitude, StrDynamic.longitude, StrDynamic.altPres, StrDynamic.altGNSS);
            printf("[accHoriz] \t %u \t[accVert] \t %u \t\t[accVel] \t %u \t\t[velVert] \t %d \t\t[nsVog] \t %d \t\t[ewVog] \t %d\n", StrDynamic.accHoriz, StrDynamic.accVert, StrDynamic.accVel, StrDynamic.velVert, StrDynamic.nsVog, StrDynamic.ewVog);
            printf("[state] \t 0x%X \t\t[squawk] \t %u \t\t[fixType] \t 0x%X \t\t[numSats] \t %u \t\t[emStatus] \t %u \t\t[control] \t 0x%X\n", StrDynamic.state, StrDynamic.squawk, StrDynamic.fixType, StrDynamic.numSats, StrDynamic.emStatus, StrDynamic.control);
            printf("   <Checksum>\n");
            printf("[CKA] \t\t 0x%X \t\t[CKB] \t\t 0x%X \t\t[CK_calc] \t 0x%X   \t[CRC_EXTRA] \t %d\n", StrDynamic.checksum.CKA, StrDynamic.checksum.CKB, CK_calc, StrDynamic.CRC_EXTRA);
            printf("\n");
#endif
            memcpy((void *)(&Dynamic_Data), (void *)(&StrDynamic), sizeof(StrDynamic)-1);
            Dynamic_pub.publish(Dynamic_Data);
#endif
#if SEND_STATIC
            StrStatic.header.Start_Flag = 0xFE;
            StrStatic.header.Length = 19;
            StrStatic.header.Sequence = count_send;
            StrStatic.header.System_ID = 1;
            StrStatic.header.Component = 156;
            StrStatic.header.Message_ID = 201;
            StrStatic.ICAO[0];
            StrStatic.ICAO[1];
            StrStatic.ICAO[2];
            StrStatic.integrity;
            StrStatic.Callsign_Flt_Plan[0] = 'U';
            StrStatic.Callsign_Flt_Plan[1] = 'S';
            StrStatic.Callsign_Flt_Plan[2] = 'R';
            StrStatic.Callsign_Flt_Plan[3] = '9';
            StrStatic.Callsign_Flt_Plan[4] = '0';
            StrStatic.Callsign_Flt_Plan[5] = '9';
            StrStatic.Callsign_Flt_Plan[6] = ' ';
            StrStatic.Callsign_Flt_Plan[7] = ' ';
            StrStatic.capability;
            StrStatic.emitter;
            StrStatic.alwEncode;
            StrStatic.gpsLatOffs;
            StrStatic.gpsLonOffs;

            memcpy((void *)(sendData_Static), (void *)(&StrStatic), sizeof(StrStatic)-1);

            CK_calc = 0xffff;
            for (int j = 1;j <= 5 + sendData_Static[1]; j++)
            {
                CK_calc = update_checksum(CK_calc, sendData_Static[j]);
            }
            CK_calc = update_checksum(CK_calc, StrStatic.CRC_EXTRA);

            StrStatic.checksum.CKA = (CK_calc & 0xff);
            StrStatic.checksum.CKB = (CK_calc >> 8);
            sendData_Static[sizeof(sendData_Static)-2] = StrStatic.checksum.CKA;
            sendData_Static[sizeof(sendData_Static)-1] = StrStatic.checksum.CKB;

            ser.write(sendData_Static, sizeof(sendData_Static));
            count_send++;

            printf("\t[Static]\n");
            printf("_ROSTOPIC_ \t /ADS_B/Static \t\t\t /ADS_B/Static/Callsign\n");
#if DEBUG_MODE
            printf("   <Header>\n");
            printf("[Start_Flag] \t 0x%X \t\t[Length] \t %d \t\t[Sequence] \t %d \t\t[System ID] \t %d \t\t[Component] \t %d \t\t[Message ID] \t %d\n", StrStatic.header.Start_Flag, StrStatic.header.Length, StrStatic.header.Sequence, StrStatic.header.System_ID, StrStatic.header.Component, StrStatic.header.Message_ID);
            printf("   <Data>\n");
            printf("[ICAO_address] \t %u \t[callsign] \t %s\n", StrStatic.ICAO, StrStatic.Callsign_Flt_Plan);
            printf("[integrity] \t %u \t\t[stallSpeed] \t %u \t\t[capability] \t %u \t\t[emitter] \t %u\n", StrStatic.integrity, StrStatic.stallSpeed, StrStatic.capability, StrStatic.emitter);
            printf("[alwEncode] \t %u \t\t[gpsLatOffs] \t %u \t\t[gpsLonOffs] \t %u\n", StrStatic.alwEncode, StrStatic.gpsLatOffs, StrStatic.gpsLonOffs);
            printf("   <Checksum>\n");
            printf("[CKA] \t\t 0x%X \t\t[CKB] \t\t 0x%X \t\t[CK_calc] \t 0x%X   \t[CRC_EXTRA] \t %d\n", StrStatic.checksum.CKA, StrStatic.checksum.CKB, CK_calc, StrStatic.CRC_EXTRA);
            printf("\n");
#endif
            memcpy((void *)(&Static_Data), (void *)(&StrStatic), sizeof(StrStatic)-1);
            Static_pub.publish(Static_Data);
#endif
        }
        if (count_send > 255)
        {
            count_send = 0;
        }

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
#if DEBUG_MODE
        ROS_INFO_STREAM("ICAO_address : " << TR_Data_Sub.ICAO_address);
        ROS_INFO_STREAM("ICAO_address_Arrary : " << TR_Data_Array_Sub.Traffic_Reports[0].ICAO_address);
#endif
    }
}

int update_checksum(int CK_calc, int data)
{
    int temp;
    data = data & 0xff;
    temp = data ^ (CK_calc & 0xff);
    temp ^= (temp << 4) & 0xff;
    CK_calc = ((CK_calc >> 8) & 0xff) ^ (temp << 8) ^ (temp << 3) ^ ((temp >> 4) & 0xf);
    return CK_calc;
}
